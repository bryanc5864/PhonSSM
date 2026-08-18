"""
Train the REAL PhonSSM (AGAN + BiSSM, PDM/HPC-bypass option) on the official
27-joint HRNet WLASL splits — the novel-architecture push to beat DSTA-SLR SOTA.

BiSSM (selective state-space, Mamba-style) is PhonSSM's original differentiator
vs. attention/GCN baselines like DSTA-Net; it was never actually benchmarked
before this. PDM's unsupervised phonological decomposition is bypassed by
default (config.use_pdm=False) since it measurably hurt on real official-split
data (~40% vs ~70% test top-1 with a plain head, same features).
"""
import os, sys, math, time, argparse, copy, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.phonssm import PhonSSM, PhonSSMConfig
from training.train_sign27 import load_split, normalize_sign27
from training.train_strong27 import make_streams, DS
from training.train_official import EMA, param_groups, evaluate


def run(subset, epochs, frames, streams, device, tag, seed=0, width=256, ssm_layers=2,
        use_pdm=False, save_logits=False, batch_size=64):
    torch.manual_seed(seed); np.random.seed(seed)
    st = streams.split(',')
    def prep(sp):
        X, y = load_split(sp, subset, frames); return make_streams(normalize_sign27(X), st), y
    Xtr, ytr = prep('train'); Xva, yva = prep('val'); Xte, yte = prep('test')
    in_ch = Xtr.shape[-1]
    print(f"[PhonSSM WLASL{subset} {streams}] train {len(ytr)} val {len(yva)} test {len(yte)} "
          f"in_ch={in_ch} use_pdm={use_pdm}", flush=True)

    tr = DataLoader(DS(Xtr, ytr, True), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(DS(Xva, yva), batch_size=128); te = DataLoader(DS(Xte, yte), batch_size=128)

    cfg = PhonSSMConfig(num_signs=subset, input_mode='sign27', num_landmarks=27, num_frames=frames,
                        coord_dim=in_ch, use_multistream=False, use_pdm=use_pdm,
                        spatial_out=width, d_model=width, num_ssm_layers=ssm_layers,
                        component_dim=48, dropout=0.3)
    model = PhonSSM(cfg).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"({model.count_parameters()})", flush=True)
    # SSM-specific params (A_log, dt_proj, D) need their own (higher, no-decay) LR —
    # they parameterize continuous-time dynamics and converge far slower than
    # ordinary weights under a single shared AdamW LR (empirically: shared-LR
    # BiSSM plateaus at ~33% vs GRU's 69% on identical data/epochs).
    ssm_params, other_decay, other_nodecay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in ('A_log', 'dt_proj', '.D')):
            ssm_params.append(p)
        elif p.ndim <= 1 or 'prototype' in n or 'logit_scale' in n:
            other_nodecay.append(p)
        else:
            other_decay.append(p)
    opt = torch.optim.AdamW([
        {'params': other_decay, 'weight_decay': 5e-2, 'lr': 8e-4},
        {'params': other_nodecay, 'weight_decay': 0.0, 'lr': 8e-4},
        {'params': ssm_params, 'weight_decay': 0.0, 'lr': 4e-3},
    ])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, [8e-4, 8e-4, 4e-3], epochs=epochs, steps_per_epoch=len(tr), pct_start=0.1)
    ema = EMA(model, 0.99)

    best_val = -1; best_state = None
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for Xb, yb in tr:
            opt.zero_grad()
            out = model(Xb.to(device))
            model.compute_loss(out, yb.to(device), label_smoothing=0.1)['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); ema.update(model)
        bk = copy.deepcopy(model.state_dict()); model.load_state_dict(ema.shadow)
        v1, _, _ = evaluate(model, va, device, subset)
        model.load_state_dict(bk)
        if v1 > best_val:
            best_val = v1; best_state = {k: t.clone() for k, t in ema.shadow.items()}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d} val {v1:.2f} (best {best_val:.2f}) [{(time.time()-t0)/60:.1f}min]", flush=True)
    model.load_state_dict(best_state)
    t1, t5, pc = evaluate(model, te, device, subset)
    print(f">>> PhonSSM WLASL{subset}[{tag}] top1={t1:.2f}% top5={t5:.2f}% P-C={pc:.2f}% "
          f"(val {best_val:.2f}) n={len(yte)}", flush=True)
    rd = ROOT / 'benchmarks' / 'phonssm27'; rd.mkdir(parents=True, exist_ok=True)
    res = {'subset': subset, 'streams': streams, 'tag': tag, 'use_pdm': use_pdm,
           'test_top1': round(t1, 2), 'test_top5': round(t5, 2), 'test_perclass': round(pc, 2),
           'best_val_top1': round(best_val, 2), 'test_n': int(len(yte)), 'width': width, 'ssm_layers': ssm_layers}
    (rd / f'wlasl{subset}_{tag}.json').write_text(json.dumps(res, indent=2))
    if save_logits:
        model.eval(); L = []
        with torch.no_grad():
            for Xb, yb in te: L.append(model(Xb.to(device))['logits'].cpu())
        np.save(rd / f'logits_wlasl{subset}_{tag}.npy', torch.cat(L).numpy())
        np.save(rd / f'ytest_wlasl{subset}.npy', np.array(yte))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--streams', type=str, default='joint')
    ap.add_argument('--tag', type=str, default='v1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--ssm_layers', type=int, default=2)
    ap.add_argument('--use_pdm', action='store_true')
    ap.add_argument('--logits', action='store_true')
    ap.add_argument('--batch_size', type=int, default=64)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[device {dev}]", flush=True)
    t0 = time.time()
    run(a.subset, a.epochs, a.frames, a.streams, dev, a.tag, a.seed, a.width, a.ssm_layers,
        a.use_pdm, a.logits, a.batch_size)
    print(f"[done {(time.time()-t0)/60:.1f} min]", flush=True)
