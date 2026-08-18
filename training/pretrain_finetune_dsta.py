"""
Apply the proven pretrain-on-WLASL2000-then-finetune improvement to the faithful
DSTA-Net port. Pretrain the backbone on the full 2000-way official train set
(14,289 videos -- 10x WLASL100's 1,442), then transfer everything except the
final classifier (`fc`, which is class-count-dependent) to a fresh model sized
for the target subset and continue training with the same SGD/nesterov recipe
at a reduced schedule.
"""
import os, sys, json, time, argparse, random, copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.dstanet import Model
from training.train_dstanet import (load_split_raw, DSTAFeeder, evaluate, adjust_lr)


def pretrain(streams, epochs, tag, device, batch_size=24, base_lr=0.1,
            warmup=5, step=(36, 48), keep_rate=0.9, window_size=120, eval_every=5, seed=0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    bone = streams in ('bone', 'bonemotion')
    motion = streams in ('motion', 'bonemotion')

    Xtr, ytr = load_split_raw('train', 2000)
    Xva, yva = load_split_raw('val', 2000)
    print(f"[PRETRAIN 2000-way {streams}] train {len(ytr)} val {len(yva)}", flush=True)

    tr = DataLoader(DSTAFeeder(Xtr, ytr, True, window_size, bone, motion),
                    batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    va = DataLoader(DSTAFeeder(Xva, yva, False, window_size, bone, motion), batch_size=batch_size, num_workers=2)

    model = Model(num_class=2000, num_point=27, num_person=1, groups=16, block_size=41,
                 inner_dim=64, drop_layers=2, depth=4, window_size=window_size).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1; best_state = None
    t0 = time.time()
    for ep in range(epochs):
        lr = adjust_lr(opt, ep, base_lr, warmup, step)
        keep_prob = -(1 - keep_rate) / 100 * ep + 1.0 if ep < 100 else keep_rate
        model.train()
        for Xb, yb in tr:
            opt.zero_grad()
            loss = loss_fn(model(Xb.to(device), keep_prob), yb.to(device))
            loss.backward(); opt.step()
        if ep % eval_every == 0 or ep == epochs - 1:
            v1, _, _ = evaluate(model, va, device, 2000)
            if v1 > best_val:
                best_val = v1
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            print(f"  ep{ep:3d} lr {lr:.4f} keep_prob {keep_prob:.3f} val {v1:.2f} "
                  f"(best {best_val:.2f}) [{(time.time()-t0)/60:.1f}min]", flush=True)
    ckpt_dir = ROOT / 'benchmarks' / 'dstanet_pretrain'; ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f'encoder_{streams}_{tag}.pt'
    torch.save({'state_dict': best_state, 'val2000': best_val}, path)
    print(f">>> PRETRAIN done best_val2000={best_val:.2f}% saved to {path}", flush=True)
    return path


def finetune(subset, streams, ckpt_path, epochs, tag, device, batch_size=24, base_lr=0.02,
            warmup=5, step=(60, 80), keep_rate=0.9, window_size=120, eval_every=5, seed=0,
            save_logits=False):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    bone = streams in ('bone', 'bonemotion')
    motion = streams in ('motion', 'bonemotion')

    Xtr, ytr = load_split_raw('train', subset)
    Xva, yva = load_split_raw('val', subset)
    Xte, yte = load_split_raw('test', subset)
    print(f"[FINETUNE WLASL{subset} {streams}] train {len(ytr)} val {len(yva)} test {len(yte)}", flush=True)

    tr = DataLoader(DSTAFeeder(Xtr, ytr, True, window_size, bone, motion),
                    batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    va = DataLoader(DSTAFeeder(Xva, yva, False, window_size, bone, motion), batch_size=batch_size, num_workers=2)
    te = DataLoader(DSTAFeeder(Xte, yte, False, window_size, bone, motion), batch_size=batch_size, num_workers=2)

    model = Model(num_class=subset, num_point=27, num_person=1, groups=16, block_size=41,
                 inner_dim=64, drop_layers=2, depth=4, window_size=window_size).to(device)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    own = model.state_dict()
    transferred = {k: v for k, v in ck['state_dict'].items() if k in own and own[k].shape == v.shape and not k.startswith('fc.')}
    own.update(transferred); model.load_state_dict(own)
    print(f"  transferred {len(transferred)}/{len(own)} tensors from pretrained encoder", flush=True)

    opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1; best_state = None
    t0 = time.time()
    for ep in range(epochs):
        lr = adjust_lr(opt, ep, base_lr, warmup, step)
        keep_prob = -(1 - keep_rate) / 100 * ep + 1.0 if ep < 100 else keep_rate
        model.train()
        for Xb, yb in tr:
            opt.zero_grad()
            loss = loss_fn(model(Xb.to(device), keep_prob), yb.to(device))
            loss.backward(); opt.step()
        if ep % eval_every == 0 or ep == epochs - 1:
            v1, _, _ = evaluate(model, va, device, subset)
            if v1 > best_val:
                best_val = v1
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            print(f"  ep{ep:3d} lr {lr:.4f} keep_prob {keep_prob:.3f} val {v1:.2f} "
                  f"(best {best_val:.2f}) [{(time.time()-t0)/60:.1f}min]", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    t1, t5, pc = evaluate(model, te, device, subset)
    print(f">>> FT-DSTA WLASL{subset}[{streams}/{tag}] top1={t1:.2f}% top5={t5:.2f}% "
          f"P-C={pc:.2f}% (val {best_val:.2f}) n={len(yte)}", flush=True)
    rd = ROOT / 'benchmarks' / 'dstanet_pretrain'; rd.mkdir(parents=True, exist_ok=True)
    res = {'subset': subset, 'streams': streams, 'tag': tag, 'test_top1': round(t1, 2),
           'test_top5': round(t5, 2), 'test_perclass': round(pc, 2), 'best_val': round(best_val, 2), 'test_n': len(yte)}
    (rd / f'ft_wlasl{subset}_{streams}_{tag}.json').write_text(json.dumps(res, indent=2))
    if save_logits:
        model.eval(); L = []
        with torch.no_grad():
            for Xb, yb in te: L.append(model(Xb.to(device)).cpu())
        np.save(rd / f'logits_wlasl{subset}_{streams}_{tag}.npy', torch.cat(L).numpy())
        np.save(rd / f'ytest_wlasl{subset}.npy', np.array(yte))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['pretrain', 'finetune'])
    ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--streams', type=str, default='joint', choices=['joint', 'bone', 'motion', 'bonemotion'])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--tag', type=str, default='v1')
    ap.add_argument('--ckpt', type=str, default='')
    ap.add_argument('--base_lr', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--logits', action='store_true')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[device {dev}]", flush=True)
    t0 = time.time()
    if a.mode == 'pretrain':
        pretrain(a.streams, a.epochs, a.tag, dev, base_lr=a.base_lr or 0.1, eval_every=a.eval_every, seed=a.seed)
    else:
        finetune(a.subset, a.streams, Path(a.ckpt), a.epochs, a.tag, dev,
                 base_lr=a.base_lr or 0.02, eval_every=a.eval_every, seed=a.seed, save_logits=a.logits)
    print(f"[done {(time.time()-t0)/60:.1f} min]", flush=True)
