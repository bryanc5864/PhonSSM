"""
Pretrain the (proven) GRU+cosine-head encoder on the FULL WLASL2000 official
training set (14,289 videos, 2000-way classification — ~10x more data than the
WLASL100 subset's 1,442 videos), then fine-tune on a target subset (100/300/1000).

Rationale: every more-sophisticated architecture we tried (graph-smoothing,
transformer, BiSSM, anatomical dual-stream) UNDERPERFORMED a plain GRU on
WLASL100, consistently — strong evidence the bottleneck is training-set size,
not architecture. WLASL100's 100 classes are literally the first 100 (by gloss
order) of WLASL2000's vocabulary, so pretraining on the full 2000-way task is a
legitimate, leakage-free way to give the encoder far more signal before
specializing it to a smaller subset.
"""
import os, sys, math, time, argparse, copy, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from training.train_sign27 import load_split, normalize_sign27
from training.train_strong27 import make_streams, DS
from training.train_official import EMA, evaluate
from training.simple_baseline27 import GRUNet


def prep(sp, subset, frames, st):
    X, y = load_split(sp, subset, frames)
    return make_streams(normalize_sign27(X), st), y


def pretrain(epochs, frames, streams, device, h, tag, batch_size=128):
    st = streams.split(',')
    Xtr, ytr = prep('train', 2000, frames, st)
    Xva, yva = prep('val', 2000, frames, st)
    in_ch = Xtr.shape[-1]
    print(f"[PRETRAIN 2000-way] train {len(ytr)} val {len(yva)} in_ch={in_ch}", flush=True)
    tr = DataLoader(DS(Xtr, ytr, True), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(DS(Xva, yva), batch_size=256)
    m = GRUNet(27 * in_ch, 2000, h=h).to(device)
    print(f"  params {sum(p.numel() for p in m.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, epochs=epochs, steps_per_epoch=len(tr), pct_start=0.1)
    ema = EMA(m, 0.995)
    best = -1; best_state = None
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        for Xb, yb in tr:
            opt.zero_grad()
            F.cross_entropy(m(Xb.to(device))['logits'], yb.to(device), label_smoothing=0.1).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step(); ema.update(m)
        bk = copy.deepcopy(m.state_dict()); m.load_state_dict(ema.shadow)
        v1, _, _ = evaluate(m, va, device, 2000)
        m.load_state_dict(bk)
        if v1 > best: best = v1; best_state = {k: t.clone() for k, t in ema.shadow.items()}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d} val {v1:.2f} (best {best:.2f}) [{(time.time()-t0)/60:.1f}min]", flush=True)
    ckpt_dir = ROOT / 'benchmarks' / 'pretrain27'; ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f'encoder_{tag}.pt'
    torch.save({'state_dict': best_state, 'h': h, 'in_ch': in_ch, 'val2000': best}, path)
    print(f">>> PRETRAIN done best_val2000={best:.2f}% saved to {path}", flush=True)
    return path


def finetune(subset, epochs, frames, streams, device, ckpt_path, tag, freeze_epochs=0, batch_size=64, lr=3e-4, seed=0, save_logits=False):
    torch.manual_seed(seed); np.random.seed(seed)
    st = streams.split(',')
    Xtr, ytr = prep('train', subset, frames, st); Xva, yva = prep('val', subset, frames, st)
    Xte, yte = prep('test', subset, frames, st)
    in_ch = Xtr.shape[-1]
    print(f"[FINETUNE WLASL{subset}] train {len(ytr)} val {len(yva)} test {len(yte)}", flush=True)
    tr = DataLoader(DS(Xtr, ytr, True), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(DS(Xva, yva), batch_size=256); te = DataLoader(DS(Xte, yte), batch_size=256)

    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    m = GRUNet(27 * ck['in_ch'], subset, h=ck['h']).to(device)   # fresh proto/scale for new class count
    sd = ck['state_dict']
    own = m.state_dict()
    transferred = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape and 'proto' not in k}
    own.update(transferred); m.load_state_dict(own)
    print(f"  transferred {len(transferred)}/{len(own)} tensors from pretrained encoder", flush=True)

    if freeze_epochs > 0:
        for n, p in m.named_parameters():
            if 'proto' not in n and 'scale' not in n:
                p.requires_grad = False

    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=2e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs=epochs, steps_per_epoch=len(tr), pct_start=0.1)
    ema = EMA(m, 0.99); best = -1; best_state = None
    t0 = time.time()
    for ep in range(epochs):
        if freeze_epochs > 0 and ep == freeze_epochs:
            for p in m.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW(m.parameters(), lr=lr * 0.3, weight_decay=2e-2)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr * 0.3, epochs=epochs - freeze_epochs,
                                                        steps_per_epoch=len(tr), pct_start=0.1)
        m.train()
        for Xb, yb in tr:
            opt.zero_grad()
            F.cross_entropy(m(Xb.to(device))['logits'], yb.to(device), label_smoothing=0.1).backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); ema.update(m)
        bk = copy.deepcopy(m.state_dict()); m.load_state_dict(ema.shadow)
        v1, _, _ = evaluate(m, va, device, subset)
        m.load_state_dict(bk)
        if v1 > best: best = v1; best_state = {k: t.clone() for k, t in ema.shadow.items()}
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d} val {v1:.2f} (best {best:.2f}) [{(time.time()-t0)/60:.1f}min]", flush=True)
    m.load_state_dict(best_state)
    t1, t5, pc = evaluate(m, te, device, subset)
    print(f">>> FINETUNE WLASL{subset}[{tag}] top1={t1:.2f}% top5={t5:.2f}% P-C={pc:.2f}% (val {best:.2f}) n={len(yte)}", flush=True)
    rd = ROOT / 'benchmarks' / 'pretrain27'; rd.mkdir(parents=True, exist_ok=True)
    (rd / f'ft_wlasl{subset}_{tag}.json').write_text(json.dumps(
        {'subset': subset, 'tag': tag, 'test_top1': round(t1, 2), 'test_top5': round(t5, 2),
         'test_perclass': round(pc, 2), 'best_val': round(best, 2), 'test_n': len(yte)}, indent=2))
    if save_logits:
        m.eval(); L = []
        with torch.no_grad():
            for Xb, yb in te: L.append(m(Xb.to(device))['logits'].cpu())
        np.save(rd / f'logits_wlasl{subset}_{tag}.npy', torch.cat(L).numpy())
        np.save(rd / f'ytest_wlasl{subset}.npy', np.array(yte))
    return {'top1': round(t1, 2)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['pretrain', 'finetune'])
    ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--streams', type=str, default='joint,bone,motion')
    ap.add_argument('--h', type=int, default=256)
    ap.add_argument('--tag', type=str, default='v1')
    ap.add_argument('--ckpt', type=str, default='')
    ap.add_argument('--freeze_epochs', type=int, default=0)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--logits', action='store_true')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    if a.mode == 'pretrain':
        pretrain(a.epochs, a.frames, a.streams, dev, a.h, a.tag, batch_size=max(a.batch_size, 128))
    else:
        finetune(a.subset, a.epochs, a.frames, a.streams, dev, Path(a.ckpt), a.tag,
                 a.freeze_epochs, a.batch_size, a.lr, a.seed, a.logits)
    print(f"[done {(time.time()-t0)/60:.1f} min]", flush=True)
