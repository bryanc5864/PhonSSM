"""
Improved training recipe for DSTA-Net v2, trained and validated on a single
WLASL subset only (NO external-data pretraining). Reuses the faithful feeder
mechanics from train_dstanet.py (bone/motion construction, resample, mirror,
shift, normalization) and adds:

- random small-angle 2D rotation + isotropic scale jitter (skeleton-level,
  applied around the frame centroid) -- standard augmentation in skeleton
  action recognition (ST-GCN++/CTR-GCN-style) absent from the original recipe.
- random joint masking (zero out a few joints per clip) -- simulates the
  transient occlusion/detector-dropout that's already present in real HRNet
  keypoints, forces the spatial mixing unit to not over-rely on any single
  joint.
- label smoothing (0.1).
- cosine LR schedule w/ linear warmup, replacing the step decay tuned for a
  250-epoch/2000-class schedule -- better suited to a small (1-2k video)
  single-subset training set.
- EMA of weights (decay 0.999), evaluated instead of the raw weights.
- flip test-time augmentation (average softmax of the clip and its mirrored
  version) at both validation-model-selection and final test time.
"""
import os, sys, json, time, argparse, random, math, copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.dstanet import ModelV2
from training.train_dstanet import (load_split_raw, random_sample_np, uniform_sample_np,
                                    BONE_EDGES, FLIP_INDEX)


def rotate_scale(data, max_deg=8.0, scale_range=(0.9, 1.1)):
    """data: (C,T,V,M) float32, in-place-ish rotate+scale around the per-clip xy centroid."""
    C, T, V, M = data.shape
    theta = math.radians(random.uniform(-max_deg, max_deg))
    s = random.uniform(*scale_range)
    cos_t, sin_t = math.cos(theta) * s, math.sin(theta) * s
    x, y = data[0].copy(), data[1].copy()
    cx, cy = x.mean(), y.mean()
    xc, yc = x - cx, y - cy
    data[0] = xc * cos_t - yc * sin_t + cx
    data[1] = xc * sin_t + yc * cos_t + cy
    return data


def joint_mask(data, p_joint=0.06):
    C, T, V, M = data.shape
    mask = np.random.random(V) > p_joint
    if mask.all():
        return data
    data[:, :, ~mask, :] = 0.0
    return data


class DSTAFeederV2(torch.utils.data.Dataset):
    def __init__(self, X, y, train, window_size=120, bone=False, motion=False, strong_aug=True):
        self.X, self.y = X, y
        self.train = train
        self.window_size = window_size
        self.bone = bone
        self.motion = motion
        self.strong_aug = strong_aug

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        data = np.array(self.X[index])
        label = int(self.y[index])
        data[np.isinf(data)] = 0.0

        if self.train and self.strong_aug and random.random() > 0.3:
            data = rotate_scale(data)
        if self.train and self.strong_aug and random.random() > 0.7:
            data = joint_mask(data)

        if self.bone:
            ori = data.copy()
            for v1, v2 in BONE_EDGES:
                data[:, :, v2 - 5, :] = ori[:, :, v2 - 5, :] - ori[:, :, v1 - 5, :]

        if self.motion:
            T = data.shape[1]
            ori = data.copy()
            for t in range(T - 1):
                data[:, t, :, :] = ori[:, t + 1, :, :] - ori[:, t, :, :]
            data[:, T - 1, :, :] = 0

        data = random_sample_np(data, self.window_size) if self.train else uniform_sample_np(data, self.window_size)

        if self.train and random.random() > 0.5:
            data = data[:, :, FLIP_INDEX, :]
            data[0, :, :, :] = 512 - data[0, :, :, :]

        data[0, :, :, :] = data[0, :, :, :] - data[0, :, 0, 0].mean(axis=0)
        data[1, :, :, :] = data[1, :, :, :] - data[1, :, 0, 0].mean(axis=0)

        if self.train and not self.bone:
            data[0, :, :, :] += random.random() * 20 - 10.0
            data[1, :, :, :] += random.random() * 20 - 10.0

        return torch.from_numpy(data.copy()), label


def flip_clip(x):
    """x: (B,C,T,V,M) tensor already normalized -> mirrored version for TTA."""
    xf = x.clone()
    xf = xf[:, :, :, FLIP_INDEX, :]
    xf[:, 0] = -xf[:, 0]
    return xf


@torch.no_grad()
def evaluate_tta(model, loader, device, num_classes, tta=True):
    model.eval()
    c1 = c5 = tot = 0
    per_c_c = np.zeros(num_classes); per_c_t = np.zeros(num_classes)
    for Xb, yb in loader:
        Xb = Xb.to(device)
        logits = F.softmax(model(Xb), dim=1)
        if tta:
            logits = logits + F.softmax(model(flip_clip(Xb)), dim=1)
        yb = yb.to(device)
        _, top5 = logits.topk(min(5, logits.shape[1]), dim=1)
        c1 += (top5[:, 0] == yb).sum().item()
        c5 += (top5 == yb[:, None]).any(1).sum().item()
        tot += yb.size(0)
        for t, p in zip(yb.cpu().numpy(), top5[:, 0].cpu().numpy()):
            per_c_t[t] += 1; per_c_c[t] += (t == p)
    seen = per_c_t > 0
    pc = float((per_c_c[seen] / per_c_t[seen]).mean() * 100) if seen.any() else 0.0
    return c1 / tot * 100, c5 / tot * 100, pc


def cosine_lr(optimizer, epoch, epochs, base_lr, warmup, min_lr_ratio=0.02):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        prog = (epoch - warmup) / max(1, epochs - warmup)
        lr = base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model):
        model.load_state_dict(self.shadow)


def run(subset, streams, epochs, tag, device, batch_size=24, base_lr=0.1,
        warmup=15, keep_rate=0.9, window_size=120, save_logits=False,
        eval_every=5, seed=0, label_smoothing=0.1, ema_decay=0.999, strong_aug=True):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    bone = streams in ('bone', 'bonemotion')
    motion = streams in ('motion', 'bonemotion')

    Xtr, ytr = load_split_raw('train', subset)
    Xva, yva = load_split_raw('val', subset)
    Xte, yte = load_split_raw('test', subset)
    print(f"[DSTAv2 WLASL{subset} {streams} seed{seed}] train {len(ytr)} val {len(yva)} test {len(yte)}", flush=True)

    tr = DataLoader(DSTAFeederV2(Xtr, ytr, True, window_size, bone, motion, strong_aug),
                    batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    va = DataLoader(DSTAFeederV2(Xva, yva, False, window_size, bone, motion), batch_size=batch_size, num_workers=2)
    te = DataLoader(DSTAFeederV2(Xte, yte, False, window_size, bone, motion), batch_size=batch_size, num_workers=2)

    model = ModelV2(num_class=subset, num_point=27, num_person=1, groups=16, block_size=41,
                    inner_dim=64, drop_layers=2, depth=4, window_size=window_size).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    ema = EMA(model, ema_decay)

    best_val = -1; best_state = None
    t0 = time.time()
    for ep in range(epochs):
        lr = cosine_lr(opt, ep, epochs, base_lr, warmup)
        keep_prob = -(1 - keep_rate) / 100 * ep + 1.0 if ep < 100 else keep_rate
        model.train()
        for Xb, yb in tr:
            opt.zero_grad()
            out = model(Xb.to(device), keep_prob)
            loss = loss_fn(out, yb.to(device))
            loss.backward()
            opt.step()
            ema.update(model)
        if (ep % eval_every == 0 or ep == epochs - 1) and len(yva):
            snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.apply_to(model)
            v1, _, _ = evaluate_tta(model, va, device, subset, tta=False)
            if v1 > best_val:
                best_val = v1
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            model.load_state_dict(snap)
            print(f"  ep{ep:3d} lr {lr:.4f} keep_prob {keep_prob:.3f} val(ema) {v1:.2f} "
                  f"(best {best_val:.2f}) [{(time.time()-t0)/60:.1f}min]", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    t1, t5, pc = evaluate_tta(model, te, device, subset, tta=True)
    print(f">>> DSTAv2 WLASL{subset}[{streams}/{tag}/seed{seed}] top1={t1:.2f}% top5={t5:.2f}% P-C={pc:.2f}% "
          f"(val {best_val:.2f}) n={len(yte)}", flush=True)
    rd = ROOT / 'benchmarks' / 'dstanet_v2'; rd.mkdir(parents=True, exist_ok=True)
    res = {'subset': subset, 'streams': streams, 'tag': tag, 'seed': seed, 'test_top1': round(t1, 2),
           'test_top5': round(t5, 2), 'test_perclass': round(pc, 2), 'best_val': round(best_val, 2),
           'test_n': len(yte), 'epochs': epochs}
    (rd / f'wlasl{subset}_{streams}_{tag}_seed{seed}.json').write_text(json.dumps(res, indent=2))
    if save_logits:
        model.eval(); L = []
        with torch.no_grad():
            for Xb, yb in te:
                Xb = Xb.to(device)
                p = F.softmax(model(Xb), dim=1) + F.softmax(model(flip_clip(Xb)), dim=1)
                L.append(p.cpu())
        np.save(rd / f'probs_wlasl{subset}_{streams}_{tag}_seed{seed}.npy', torch.cat(L).numpy())
        np.save(rd / f'ytest_wlasl{subset}.npy', np.array(yte))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--streams', type=str, default='joint', choices=['joint', 'bone', 'motion', 'bonemotion'])
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--batch_size', type=int, default=24)
    ap.add_argument('--tag', type=str, default='v2')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--logits', action='store_true')
    ap.add_argument('--base_lr', type=float, default=0.1)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[device {dev}]", flush=True)
    t0 = time.time()
    run(a.subset, a.streams, a.epochs, a.tag, dev, a.batch_size, base_lr=a.base_lr,
        save_logits=a.logits, eval_every=a.eval_every, seed=a.seed)
    print(f"[done {(time.time()-t0)/60:.1f} min]", flush=True)
