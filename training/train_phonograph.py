"""
Train PhonoGraph on a single WLASL subset, from scratch, NO external pretraining.

Recipe (all evidence-backed small-data levers, no extra data):
- strong skeleton augmentation: global rotation + anisotropic scale + shear,
  temporal crop-resample, frame reversal, per-joint gaussian jitter, joint
  masking, bone-length perturbation (built into the feeder).
- multi-stream input (joint / bone / joint-motion / bone-motion), one model per
  stream, softmax-averaged at test time.
- label smoothing 0.1.
- FR-Head-style auxiliary supervised-contrastive loss on the fused feature +
  prototype-orthogonality penalty (both from the model's return_aux path).
- SGD nesterov, cosine LR w/ warmup, EMA weights, flip TTA at eval.

Reuses the official-split loader and the (verified, leakage-free) 27-joint data
from train_dstanet.py.
"""
import os, sys, json, time, argparse, random, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.phonograph import PhonoGraph
from training.train_dstanet import (load_split_raw, random_sample_np, uniform_sample_np,
                                    BONE_EDGES, FLIP_INDEX)


def rand_transform(data, max_deg=12.0, scale=(0.85, 1.15), shear=0.12):
    """Global 2D rotation + anisotropic scale + shear around the xy centroid."""
    theta = math.radians(random.uniform(-max_deg, max_deg))
    sx, sy = random.uniform(*scale), random.uniform(*scale)
    shx, shy = random.uniform(-shear, shear), random.uniform(-shear, shear)
    c, s = math.cos(theta), math.sin(theta)
    # M = shear @ scale @ rot
    a11 = sx * c + shx * sy * s
    a12 = -sx * s + shx * sy * c
    a21 = shy * sx * c + sy * s
    a22 = -shy * sx * s + sy * c
    x, y = data[0].copy(), data[1].copy()
    cx, cy = x.mean(), y.mean()
    xc, yc = x - cx, y - cy
    data[0] = a11 * xc + a12 * yc + cx
    data[1] = a21 * xc + a22 * yc + cy
    return data


def joint_jitter(data, sigma=2.0):
    data[:2] += np.random.randn(*data[:2].shape).astype(np.float32) * sigma
    return data


def joint_mask(data, p=0.08):
    V = data.shape[2]
    m = np.random.random(V) > p
    if not m.all():
        data[:, :, ~m, :] = 0.0
    return data


def uniform_sample_offset(data, size, offset_frac):
    """Deterministic uniform temporal sample with a fractional phase shift, for
    multi-crop test-time augmentation."""
    C, T, V, M = data.shape
    if T == size:
        return data
    interval = T / size
    idx = [min(T - 1, int((i + offset_frac) * interval)) for i in range(size)]
    return data[:, idx]


class PhonoFeeder(Dataset):
    def __init__(self, X, y, train, window_size=120, bone=False, motion=False, strong=True,
                 eval_offset=0.0):
        self.X, self.y = X, y
        self.train, self.window_size = train, window_size
        self.bone, self.motion, self.strong = bone, motion, strong
        self.eval_offset = eval_offset

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        data = np.array(self.X[i]); label = int(self.y[i])
        data[np.isinf(data)] = 0.0

        if self.train and self.strong:
            if random.random() > 0.2:
                data = rand_transform(data)
            if random.random() > 0.5:
                data = joint_jitter(data)
            if random.random() > 0.7:
                data = joint_mask(data)

        if self.bone:
            ori = data.copy()
            for v1, v2 in BONE_EDGES:
                data[:, :, v2 - 5, :] = ori[:, :, v2 - 5, :] - ori[:, :, v1 - 5, :]
        if self.motion:
            T = data.shape[1]; ori = data.copy()
            for t in range(T - 1):
                data[:, t, :, :] = ori[:, t + 1, :, :] - ori[:, t, :, :]
            data[:, T - 1, :, :] = 0

        if self.train:
            data = random_sample_np(data, self.window_size)
        elif self.eval_offset:
            data = uniform_sample_offset(data, self.window_size, self.eval_offset)
        else:
            data = uniform_sample_np(data, self.window_size)

        if self.train and random.random() > 0.5:
            data = data[:, :, FLIP_INDEX, :]
            data[0, :, :, :] = 512 - data[0, :, :, :]

        data[0, :, :, :] = data[0, :, :, :] - data[0, :, 0, 0].mean(axis=0)
        data[1, :, :, :] = data[1, :, :, :] - data[1, :, 0, 0].mean(axis=0)

        if self.train and not self.bone and random.random() > 0.5:
            data[0] += random.random() * 20 - 10.0
            data[1] += random.random() * 20 - 10.0

        return torch.from_numpy(data.copy()), label


def flip_clip(x):
    xf = x[:, :, :, FLIP_INDEX, :].clone()
    xf[:, 0] = -xf[:, 0]
    return xf


@torch.no_grad()
def evaluate(model, loader, device, num_classes, tta=True):
    model.eval()
    c1 = c5 = tot = 0
    pcc = np.zeros(num_classes); pct = np.zeros(num_classes)
    probs = []
    for Xb, yb in loader:
        Xb = Xb.to(device)
        p = F.softmax(model(Xb), dim=1)
        if tta:
            p = p + F.softmax(model(flip_clip(Xb)), dim=1)
        probs.append(p.cpu())
        yb = yb.to(device)
        _, t5 = p.topk(min(5, p.shape[1]), dim=1)
        c1 += (t5[:, 0] == yb).sum().item()
        c5 += (t5 == yb[:, None]).any(1).sum().item()
        tot += yb.size(0)
        for t, pr in zip(yb.cpu().numpy(), t5[:, 0].cpu().numpy()):
            pct[t] += 1; pcc[t] += (t == pr)
    seen = pct > 0
    pc = float((pcc[seen] / pct[seen]).mean() * 100) if seen.any() else 0.0
    return c1 / tot * 100, c5 / tot * 100, pc, torch.cat(probs).numpy()


@torch.no_grad()
def multicrop_eval(model, X, y, device, num_classes, subset, bone, motion, window_size,
                   batch_size, crops=(0.0, 0.33, 0.66)):
    """Average softmax over K temporal crops x {flip} for a lower-variance test score."""
    model.eval()
    acc = None
    for off in crops:
        feeder = PhonoFeeder(X, y, False, window_size, bone, motion, eval_offset=off)
        loader = DataLoader(feeder, batch_size=batch_size, num_workers=2)
        parts = []
        for Xb, _ in loader:
            Xb = Xb.to(device)
            p = F.softmax(model(Xb), dim=1) + F.softmax(model(flip_clip(Xb)), dim=1)
            parts.append(p.cpu())
        probs = torch.cat(parts).numpy()
        acc = probs if acc is None else acc + probs
    acc = acc / (len(crops))
    pred = acc.argmax(1)
    t1 = float((pred == y).mean() * 100)
    pcc = np.zeros(num_classes); pct = np.zeros(num_classes)
    for t, p in zip(y, pred):
        pct[t] += 1; pcc[t] += (t == p)
    seen = pct > 0
    pc = float((pcc[seen] / pct[seen]).mean() * 100)
    t5 = float((np.argsort(-acc, 1)[:, :5] == y[:, None]).any(1).mean() * 100)
    return t1, t5, pc, acc


@torch.no_grad()
def recalibrate_bn(model, loader, device, max_batches=40):
    """Recompute BatchNorm running stats for SWA-averaged weights."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    for i, (Xb, _) in enumerate(loader):
        if i >= max_batches:
            break
        model(Xb.to(device))
    model.eval()


def supcon_loss(feat, labels, temp=0.1):
    """Supervised contrastive loss on L2-normalized features (FR-Head spirit)."""
    f = F.normalize(feat, dim=1)
    sim = f @ f.t() / temp
    n = f.size(0)
    sim.fill_diagonal_(-1e9)
    lab = labels.view(-1, 1)
    pos = (lab == lab.t()).float()
    pos.fill_diagonal_(0)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_cnt = pos.sum(1)
    valid = pos_cnt > 0
    if valid.sum() == 0:
        return torch.zeros((), device=feat.device)
    loss = -(pos * logp).sum(1)[valid] / pos_cnt[valid]
    return loss.mean()


def cosine_lr(opt, ep, epochs, base_lr, warmup, min_ratio=0.02):
    if ep < warmup:
        lr = base_lr * (ep + 1) / warmup
    else:
        prog = (ep - warmup) / max(1, epochs - warmup)
        lr = base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))
    for g in opt.param_groups:
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


def run(subset, streams, epochs, tag, device, batch_size=32, base_lr=0.1, warmup=15,
        window_size=120, seed=0, eval_every=5, label_smoothing=0.1, base=64,
        num_proto=64, drop=0.2, w_supcon=0.1, w_ortho=0.01, ema_decay=0.999,
        save_probs=False, strong=True, swa=False, swa_start_frac=0.6, multicrop=False):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    bone = streams in ('bone', 'bonemotion')
    motion = streams in ('motion', 'bonemotion')

    Xtr, ytr = load_split_raw('train', subset)
    Xva, yva = load_split_raw('val', subset)
    Xte, yte = load_split_raw('test', subset)
    print(f"[PhonoGraph WLASL{subset} {streams} seed{seed}] train {len(ytr)} val {len(yva)} test {len(yte)}", flush=True)

    tr = DataLoader(PhonoFeeder(Xtr, ytr, True, window_size, bone, motion, strong),
                    batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    va = DataLoader(PhonoFeeder(Xva, yva, False, window_size, bone, motion), batch_size=batch_size, num_workers=2)
    te = DataLoader(PhonoFeeder(Xte, yte, False, window_size, bone, motion), batch_size=batch_size, num_workers=2)

    model = PhonoGraph(num_class=subset, base=base, num_proto=num_proto, drop=drop).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, nesterov=True, weight_decay=2e-4)
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    ema = EMA(model, ema_decay)

    best_val = -1; best_state = None
    swa_state = None; swa_n = 0; swa_start = int(epochs * swa_start_frac)
    t0 = time.time()
    for ep in range(epochs):
        lr = cosine_lr(opt, ep, epochs, base_lr, warmup)
        model.train()
        for Xb, yb in tr:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, feat, ortho = model(Xb, return_aux=True)
            loss = ce(logits, yb) + w_supcon * supcon_loss(feat, yb) + w_ortho * ortho
            loss.backward()
            opt.step()
            ema.update(model)
        if ep % eval_every == 0 or ep == epochs - 1:
            snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.apply_to(model)
            v1, _, _, _ = evaluate(model, va, device, subset, tta=False)
            if v1 > best_val:
                best_val = v1
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            model.load_state_dict(snap)
            print(f"  ep{ep:3d} lr {lr:.4f} val(ema) {v1:.2f} (best {best_val:.2f}) "
                  f"[{(time.time()-t0)/60:.1f}min]", flush=True)
        if swa and ep >= swa_start:
            if swa_state is None:
                swa_state = {k: v.detach().clone().float() for k, v in ema.shadow.items()}
                swa_n = 1
            else:
                for k, v in ema.shadow.items():
                    if v.dtype.is_floating_point:
                        swa_state[k].mul_(swa_n / (swa_n + 1)).add_(v.detach().float() / (swa_n + 1))
                    else:
                        swa_state[k] = v.detach().clone().float()
                swa_n += 1

    # choose final weights by VALIDATION (no test peeking): best-val-EMA vs SWA
    if swa and swa_state is not None:
        model.load_state_dict({k: v.to(next(model.parameters()).dtype) for k, v in swa_state.items()})
        recalibrate_bn(model, tr, device)
        swa_val, _, _, _ = evaluate(model, va, device, subset, tta=False)
        swa_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  [SWA] val {swa_val:.2f} vs best-val-EMA {best_val:.2f}", flush=True)
        if swa_val >= best_val:
            best_state = swa_weights; best_val = swa_val
    if best_state is not None:
        model.load_state_dict(best_state)
    if multicrop:
        t1, t5, pc, probs = multicrop_eval(model, Xte, yte, device, subset, subset,
                                           bone, motion, window_size, batch_size)
    else:
        t1, t5, pc, probs = evaluate(model, te, device, subset, tta=True)
    print(f">>> PhonoGraph WLASL{subset}[{streams}/{tag}/seed{seed}] top1={t1:.2f}% top5={t5:.2f}% "
          f"P-C={pc:.2f}% (val {best_val:.2f}) n={len(yte)}", flush=True)
    rd = ROOT / 'benchmarks' / 'phonograph'; rd.mkdir(parents=True, exist_ok=True)
    res = {'subset': subset, 'streams': streams, 'tag': tag, 'seed': seed, 'test_top1': round(t1, 2),
           'test_top5': round(t5, 2), 'test_perclass': round(pc, 2), 'best_val': round(best_val, 2),
           'test_n': len(yte), 'epochs': epochs}
    (rd / f'wlasl{subset}_{streams}_{tag}_seed{seed}.json').write_text(json.dumps(res, indent=2))
    if save_probs:
        np.save(rd / f'probs_wlasl{subset}_{streams}_{tag}_seed{seed}.npy', probs)
        np.save(rd / f'ytest_wlasl{subset}.npy', np.array(yte))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--streams', type=str, default='joint', choices=['joint', 'bone', 'motion', 'bonemotion'])
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--tag', type=str, default='v1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--base', type=int, default=64)
    ap.add_argument('--base_lr', type=float, default=0.1)
    ap.add_argument('--w_supcon', type=float, default=0.1)
    ap.add_argument('--probs', action='store_true')
    ap.add_argument('--swa', action='store_true')
    ap.add_argument('--multicrop', action='store_true')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[device {dev}]", flush=True)
    t0 = time.time()
    run(a.subset, a.streams, a.epochs, a.tag, dev, a.batch_size, base_lr=a.base_lr,
        seed=a.seed, eval_every=a.eval_every, base=a.base, w_supcon=a.w_supcon, save_probs=a.probs,
        swa=a.swa, multicrop=a.multicrop)
    print(f"[done {(time.time()-t0)/60:.1f} min]", flush=True)
