"""
Train the revised PhonSSM on the SAM-SLR / DSTA-SLR **27-joint HRNet whole-body**
skeletons — the exact feature basis behind pose-only WLASL SOTA (DSTA-SLR 82.4%
on WLASL100). Apples-to-apples with the baseline we are trying to beat.

Data (SAM-SLR-v2 format): {split}_data_joint.npy = (N, C=3, T=150, V=27, M=1),
{split}_label.pkl = (video_id_names, labels). Provided for WLASL2000; we subset
to 100/300/1000 by class id (WLASL_v0.3 top-K gloss order).
"""
import os, sys, json, time, math, argparse, pickle, copy
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
from training.train_official import EMA, param_groups, evaluate

DATA = ROOT / 'data' / 'sota_skeleton' / 'wlasl27'
_V03 = json.load(open(ROOT / 'data' / 'wlasl-processed' / 'WLASL_v0.3.json'))
_VID2SPLIT = {inst['video_id']: inst['split'] for e in _V03 for inst in e['instances']}


def load_split(split, subset, max_frames):
    """Return X (N, T, 27, 3), y (N,) for the OFFICIAL split, subset to the
    top-`subset` classes. SAM-SLR-v2 ships train-file = official train, and
    val-file = official (val+test) merged; we reconstruct val vs test by the
    WLASL_v0.3 per-video_id split so the TEST set is the real official one."""
    src = 'train' if split == 'train' else 'val'
    joint = np.load(DATA / f'{src}_data_joint.npy')            # (N,3,T,27,1)
    names, labels = pickle.load(open(DATA / f'{src}_label.pkl', 'rb'))
    labels = np.asarray(labels)
    sel = np.array([_VID2SPLIT.get(str(n)) == split for n in names])
    joint, labels = joint[sel], labels[sel]
    X = joint[:, :, :, :, 0].transpose(0, 2, 3, 1)             # (N, T, 27, 3)
    # temporal resample to max_frames (linear)
    if X.shape[1] != max_frames:
        T = X.shape[1]
        src = np.linspace(0, T - 1, max_frames)
        lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, T - 1)
        fr = (src - lo)[None, :, None, None]
        X = X[:, lo] * (1 - fr) + X[:, hi] * fr
    if subset < 2000:
        keep = labels < subset
        X, labels = X[keep], labels[keep]
    return X.astype(np.float32), labels.astype(np.int64)


def normalize_sign27(X, conf_gate=0.1):
    """Per-sequence center (neck = shoulder midpoint, joints 1&2) + scale
    (shoulder width). Uses ONLY (x, y) — the HRNet confidence (channel 2) is a
    quality score, not a spatial coordinate, so feeding it as a 3rd 'coord' made
    the bone/motion streams compute meaningless 'confidence velocity'. Instead we
    gate: joints below `conf_gate` confidence are zeroed (treated as unreliable).
    X: (N,T,27,3) pixel coords + score. Returns (N,T,27,2)."""
    xy = X[..., :2].astype(np.float32).copy()
    conf = X[..., 2]
    lowconf = conf < conf_gate                                 # (N,T,27) unreliable
    for i in range(len(xy)):
        sh = xy[i, :, 1], xy[i, :, 2]                          # L/R shoulder over time
        center = np.median((sh[0] + sh[1]) / 2.0, axis=0)
        scale = float(np.median(np.linalg.norm(sh[0] - sh[1], axis=1)))
        radius = float(np.median(np.linalg.norm(xy[i].reshape(-1, 2) - center, axis=1)))
        if not np.isfinite(scale) or scale < max(0.2 * radius, 1e-3):
            scale = max(0.2 * radius, 1e-3)
        xy[i] = (xy[i] - center) / scale
    xy[lowconf] = 0.0
    return np.clip(xy, -5, 5).astype(np.float32)


class S27(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self):
        return len(self.y)

    def _aug(self, s):
        s = s.copy(); T = s.shape[0]
        valid = ~np.all(s == 0, axis=2)
        th = np.random.uniform(-0.23, 0.23); c, sn = math.cos(th), math.sin(th)
        x, y = s[..., 0].copy(), s[..., 1].copy()
        s[..., 0] = c * x - sn * y; s[..., 1] = sn * x + c * y
        s[..., :2] *= np.random.uniform(0.9, 1.1)
        s[..., :2] += np.random.uniform(-0.05, 0.05, (1, 1, 2))
        s[..., :2] += np.random.normal(0, 0.01, s[..., :2].shape).astype(np.float32)
        s[~valid] = 0.0
        return np.clip(s, -5, 5).astype(np.float32)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = self._aug(x)
        return torch.from_numpy(x.reshape(x.shape[0], -1)), int(self.y[i])


def run(subset, epochs, seed, device, aug, ema_decay, warmup, width, cdim, frames, tag):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = load_split('train', subset, frames)
    Xva, yva = load_split('val', subset, frames)
    Xte, yte = load_split('test', subset, frames)
    Xtr, Xva, Xte = normalize_sign27(Xtr), normalize_sign27(Xva), normalize_sign27(Xte)
    print(f"[WLASL{subset} sign27] train {len(ytr)} val {len(yva)} test {len(yte)} | "
          f"classes {len(set(ytr.tolist()))} | aug={aug} ema={ema_decay}", flush=True)

    tr = DataLoader(S27(Xtr, ytr, aug), batch_size=64, shuffle=True, num_workers=4)
    va = DataLoader(S27(Xva, yva), batch_size=128) if len(yva) else None
    te = DataLoader(S27(Xte, yte), batch_size=128)

    cfg = PhonSSMConfig(num_signs=subset, input_mode='sign27', num_landmarks=27,
                        num_frames=frames, coord_dim=2,
                        spatial_out=width, d_model=width, component_dim=cdim)
    model = PhonSSM(cfg).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(param_groups(model, 1e-2), lr=3e-4)
    def lr_at(ep):
        if ep < warmup: return (ep + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    ema = EMA(model, ema_decay) if ema_decay > 0 else None

    best_val = -1; best_state = None
    for ep in range(epochs):
        model.train()
        for Xb, yb in tr:
            opt.zero_grad()
            out = model(Xb.to(device))
            model.compute_loss(out, yb.to(device), label_smoothing=0.1)['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema: ema.update(model)
        sched.step()
        if va is not None:
            if ema:
                bk = copy.deepcopy(model.state_dict()); model.load_state_dict(ema.shadow)
            v1, _, _ = evaluate(model, va, device, subset)
            if ema: model.load_state_dict(bk)
            if v1 > best_val:
                best_val = v1
                best_state = {k: t.detach().clone() for k, t in (ema.shadow if ema else model.state_dict()).items()}
            if ep % 10 == 0 or ep == epochs - 1:
                print(f"  ep{ep:3d} val {v1:.2f} (best {best_val:.2f})", flush=True)
    if best_state: model.load_state_dict(best_state)
    t1, t5, pc = evaluate(model, te, device, subset)
    out = {'subset': subset, 'protocol': 'official_sign27_HRNet', 'tag': tag,
           'test_top1': round(t1, 2), 'test_top5': round(t5, 2), 'test_perclass': round(pc, 2),
           'best_val_top1': round(best_val, 2), 'test_n': int(len(yte)), 'width': width}
    print(f">>> WLASL{subset} sign27[{tag}] top1={t1:.2f}% top5={t5:.2f}% P-C={pc:.2f}% (val {best_val:.2f}) n={len(yte)}", flush=True)
    rd = ROOT / 'benchmarks' / 'sign27'; rd.mkdir(parents=True, exist_ok=True)
    (rd / f'wlasl{subset}_{tag}.json').write_text(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--aug', action='store_true')
    ap.add_argument('--ema', type=float, default=0.0)
    ap.add_argument('--warmup', type=int, default=15)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--cdim', type=int, default=48)
    ap.add_argument('--frames', type=int, default=100)
    ap.add_argument('--tag', type=str, default='v1')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[device {dev}]", flush=True)
    t0 = time.time()
    run(a.subset, a.epochs, a.seed, dev, a.aug, a.ema, a.warmup, a.width, a.cdim, a.frames, a.tag)
    print(f"[done {(time.time()-t0)/60:.1f} min]", flush=True)
