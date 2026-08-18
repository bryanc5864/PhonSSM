"""
Official-split PhonSSM trainer (workhorse for the SOTA push).

Loads leakage-free official splits from data/processed/wlasl_official/wlasl{N}
(built by build_wlasl_official.py after the normalization fix), and trains the
revised model. Incorporates the recipe fixes from the diagnosis:
  - learnable logit scale (temperature bug fixed in hpc.py)
  - model.compute_loss (CE + PDM orthogonality + prototype diversity)
  - AdamW with decay/no-decay param groups (no WD on norms/biases/prototypes/scale)
  - linear warmup + cosine decay
  - deep-cloned best-val checkpoint (fixes the shallow .copy() bug)
  - optional on-the-fly geometric + temporal augmentation (--aug)
  - optional weight EMA (--ema)
  - top-1/top-5, per-class (P-C) accuracy, seed control
"""
import os, sys, json, time, math, argparse, copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.phonssm import PhonSSM, PhonSSMConfig

SPLIT_ROOT = ROOT / 'data' / 'processed' / 'wlasl_official'


# data + augmentation
class SeqDataset(Dataset):
    """(N, T, 225) float32 sequences; optional train-time augmentation.
    Feature layout per frame: 75 landmarks x 3 (x,y,z), flattened."""
    def __init__(self, X, y, augment=False, num_landmarks=75):
        self.X = X.astype(np.float32); self.y = y.astype(np.int64)
        self.augment = augment; self.N = num_landmarks

    def __len__(self):
        return len(self.y)

    def _aug(self, seq):
        T = seq.shape[0]
        s = seq.reshape(T, self.N, 3).copy()
        valid = ~np.all(s == 0.0, axis=2)                       # keep missing joints at 0
        # random 2D rotation in image plane (+/- ~13 deg)
        th = np.random.uniform(-0.23, 0.23)
        c, sn = math.cos(th), math.sin(th)
        x, y = s[..., 0].copy(), s[..., 1].copy()
        s[..., 0] = c * x - sn * y
        s[..., 1] = sn * x + c * y
        # random isotropic scale + small translation
        s[..., :2] *= np.random.uniform(0.9, 1.1)
        s[..., :2] += np.random.uniform(-0.05, 0.05, size=(1, 1, 2))
        # per-joint gaussian jitter
        s += np.random.normal(0, 0.01, size=s.shape).astype(np.float32)
        s[~valid] = 0.0
        # temporal: random resample to same length (mild speed warp) or frame dropout
        if np.random.rand() < 0.5:
            n = max(4, int(T * np.random.uniform(0.8, 1.0)))
            idx = np.sort(np.random.choice(T, n, replace=False))
            s = s[idx]
            # pad back to T by repeating last frame
            if len(s) < T:
                s = np.concatenate([s, np.repeat(s[-1:], T - len(s), axis=0)], 0)
        return np.clip(s.reshape(T, self.N * 3), -5, 5).astype(np.float32)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = self._aug(x)
        return torch.from_numpy(x), int(self.y[i])


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()


# eval
@torch.no_grad()
def evaluate(model, dl, device, num_classes):
    model.eval()
    tot = 0; c1 = 0; c5 = 0
    per_c_correct = np.zeros(num_classes); per_c_total = np.zeros(num_classes)
    for Xb, yb in dl:
        logits = model(Xb.to(device))['logits']
        yb = yb.to(device)
        _, top5 = logits.topk(min(5, logits.shape[1]), dim=1)
        c1 += (top5[:, 0] == yb).sum().item()
        c5 += (top5 == yb[:, None]).any(1).sum().item()
        tot += yb.size(0)
        for t, p in zip(yb.cpu().numpy(), top5[:, 0].cpu().numpy()):
            per_c_total[t] += 1; per_c_correct[t] += (t == p)
    seen = per_c_total > 0
    pc = float((per_c_correct[seen] / per_c_total[seen]).mean() * 100) if seen.any() else 0.0
    return c1 / tot * 100, c5 / tot * 100, pc


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or 'prototype' in n or 'logit_scale' in n or 'A_learn' in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': wd}, {'params': no_decay, 'weight_decay': 0.0}]


def load(d, name):
    return np.load(d / f'X_{name}.npy'), np.load(d / f'y_{name}.npy')


def run(subset, epochs, seed, device, aug, ema_decay, warmup, tag, width=128, cdim=32, frames=30):
    torch.manual_seed(seed); np.random.seed(seed)
    split_root = SPLIT_ROOT if frames == 30 else SPLIT_ROOT.parent / f'wlasl_official_f{frames}'
    d = split_root / f'wlasl{subset}'
    meta = json.load(open(d / 'meta.json')) if (d / 'meta.json').exists() else {}
    Xtr, ytr = load(d, 'train'); Xva, yva = load(d, 'val'); Xte, yte = load(d, 'test')
    print(f"[WLASL{subset}] train {len(ytr)} val {len(yva)} test {len(yte)} | aug={aug} ema={ema_decay}", flush=True)

    tr = DataLoader(SeqDataset(Xtr, ytr, augment=aug), batch_size=128, shuffle=True, num_workers=4, drop_last=False)
    va = DataLoader(SeqDataset(Xva, yva), batch_size=256) if len(yva) else None
    te = DataLoader(SeqDataset(Xte, yte), batch_size=256)

    cfg = PhonSSMConfig(num_signs=subset, input_mode='pose_hands', num_landmarks=75,
                        spatial_out=width, d_model=width, component_dim=cdim)
    model = PhonSSM(cfg).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(param_groups(model, 1e-2), lr=3e-4, betas=(0.9, 0.999))

    def lr_at(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        t = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    ema = EMA(model, ema_decay) if ema_decay > 0 else None
    best_val = -1; best_state = None
    for ep in range(epochs):
        model.train()
        for Xb, yb in tr:
            opt.zero_grad()
            out = model(Xb.to(device))
            losses = model.compute_loss(out, yb.to(device), label_smoothing=0.1)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema: ema.update(model)
        sched.step()
        # validate (EMA weights if enabled)
        if va is not None:
            if ema:
                backup = copy.deepcopy(model.state_dict()); model.load_state_dict(ema.shadow)
            v1, _, _ = evaluate(model, va, device, subset)
            if ema: model.load_state_dict(backup)
            if v1 > best_val:
                best_val = v1
                src = ema.shadow if ema else model.state_dict()
                best_state = {k: t.detach().clone() for k, t in src.items()}
            if ep % 10 == 0 or ep == epochs - 1:
                sc = model.hpc.logit_scale.exp().clamp(4, 64).item()
                print(f"  ep{ep:3d} val {v1:.2f} (best {best_val:.2f}) lr {opt.param_groups[0]['lr']:.2e} scale {sc:.1f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    t1, t5, pc = evaluate(model, te, device, subset)
    out = {'subset': subset, 'protocol': 'official_split', 'tag': tag,
           'test_top1': round(t1, 2), 'test_top5': round(t5, 2), 'test_perclass': round(pc, 2),
           'best_val_top1': round(best_val, 2), 'test_n': int(len(yte)),
           'aug': aug, 'ema': ema_decay, 'epochs': epochs, 'seed': seed,
           'test_videos': meta.get('test_videos')}
    print(f">>> WLASL{subset} OFFICIAL[{tag}] top1={t1:.2f}% top5={t5:.2f}% P-C={pc:.2f}% (val {best_val:.2f}) n={len(yte)}", flush=True)
    resd = ROOT / 'benchmarks' / 'official'; resd.mkdir(parents=True, exist_ok=True)
    (resd / f'wlasl{subset}_{tag}.json').write_text(json.dumps(out, indent=2))
    ckd = ROOT / 'models' / 'phonssm' / 'checkpoints'; ckd.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg.__dict__, 'result': out},
               ckd / f'official_wlasl{subset}_{tag}.pt')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--aug', action='store_true')
    ap.add_argument('--ema', type=float, default=0.0)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--cdim', type=int, default=32)
    ap.add_argument('--frames', type=int, default=30)
    ap.add_argument('--tag', type=str, default='baseline')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[device {dev}]", flush=True)
    t0 = time.time()
    run(a.subset, a.epochs, a.seed, dev, a.aug, a.ema, a.warmup, a.tag, a.width, a.cdim, a.frames)
    print(f"[done {(time.time()-t0)/60:.1f} min]", flush=True)
