"""
Faithful reproduction of DSTA-SLR's training recipe on our official-split WLASL
27-joint data, to (1) verify the port matches published numbers, then (2) serve
as the base to beat.

Feeder logic ported verbatim from feeders/feeder.py: bone/motion stream
construction, random_sample_np (train) / uniform_sample_np (test) resampling to
window_size=120, random_mirror (p=0.5, flip_index, x -> 512-x), normalization
(subtract the TIME-MEAN of joint-0's x/y from all frames/joints; NOT a
per-frame shoulder-width scale — the network's own data_bn handles scale), and
random_shift (+/-10px translation jitter, non-bone streams only).

Training recipe from config/train.yaml + main.py: SGD momentum=0.9 nesterov,
weight_decay=1e-4 (uniform, incl. bias -- decay_mult=0 in the source is a dead
param-group key PyTorch's SGD ignores), base_lr=0.1, linear warmup 20 epochs,
step decay x0.1 at [150,200], 250 epochs, batch_size 24, DropBlock keep_prob
linearly annealed 1.0->0.9 over epochs 0-100 then held at 0.9, plain
CrossEntropyLoss (no label smoothing).
"""
import os, sys, json, time, argparse, random, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.dstanet import Model

DATA = ROOT / 'data' / 'sota_skeleton' / 'wlasl27'
_V03 = json.load(open(ROOT / 'data' / 'raw' / 'wlasl' / 'start_kit' / 'WLASL_v0.3.json'))
_VID2SPLIT = {inst['video_id']: inst['split'] for e in _V03 for inst in e['instances']}

BONE_EDGES = [
    (5, 6), (5, 7), (6, 8), (8, 10), (7, 9), (9, 11),
    (12, 13), (12, 14), (12, 16), (12, 18), (12, 20),
    (14, 15), (16, 17), (18, 19), (20, 21),
    (22, 23), (22, 24), (22, 26), (22, 28), (22, 30),
    (24, 25), (26, 27), (28, 29), (30, 31),
    (10, 12), (11, 22),
]
FLIP_INDEX = np.concatenate((
    [0, 2, 1, 4, 3, 6, 5],
    [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
), axis=0)


def load_split_raw(split, subset):
    """Return X (N,3,150,27,1) float32, y (N,) for the OFFICIAL split, un-resampled."""
    src = 'train' if split == 'train' else 'val'
    joint = np.load(DATA / f'{src}_data_joint.npy')          # (N,3,150,27,1)
    names, labels = pickle.load(open(DATA / f'{src}_label.pkl', 'rb'))
    labels = np.asarray(labels)
    sel = np.array([_VID2SPLIT.get(str(n)) == split for n in names])
    joint, labels = joint[sel], labels[sel]
    if subset < 2000:
        keep = labels < subset
        joint, labels = joint[keep], labels[keep]
    return joint.astype(np.float32), labels.astype(np.int64)


def random_sample_np(data, size):
    C, T, V, M = data.shape
    if T == size:
        return data
    interval = int(np.ceil(size / T))
    idx = sorted(random.sample(list(range(T)) * interval, size))
    return data[:, idx]


def uniform_sample_np(data, size):
    C, T, V, M = data.shape
    if T == size:
        return data
    interval = T / size
    idx = [int(i * interval) for i in range(size)]
    return data[:, idx]


class DSTAFeeder(Dataset):
    def __init__(self, X, y, train, window_size=120, bone=False, motion=False):
        self.X, self.y = X, y
        self.train = train
        self.window_size = window_size
        self.bone = bone
        self.motion = motion

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        data = np.array(self.X[index])  # (3,150,27,1)
        label = int(self.y[index])
        data[np.isinf(data)] = 0.0

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
            assert data.shape[2] == 27
            data = data[:, :, FLIP_INDEX, :]
            data[0, :, :, :] = 512 - data[0, :, :, :]

        # normalization: subtract time-mean of joint-0 (nose) x/y from all x/y
        data[0, :, :, :] = data[0, :, :, :] - data[0, :, 0, 0].mean(axis=0)
        data[1, :, :, :] = data[1, :, :, :] - data[1, :, 0, 0].mean(axis=0)

        if self.train and not self.bone:
            data[0, :, :, :] += random.random() * 20 - 10.0
            data[1, :, :, :] += random.random() * 20 - 10.0

        return torch.from_numpy(data.copy()), label


def evaluate(model, loader, device, num_classes):
    model.eval()
    c1 = c5 = tot = 0
    per_c_c = np.zeros(num_classes); per_c_t = np.zeros(num_classes)
    with torch.no_grad():
        for Xb, yb in loader:
            logits = model(Xb.to(device))
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


def adjust_lr(optimizer, epoch, base_lr, warmup, step):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        lr = base_lr * (0.1 ** sum(epoch >= s for s in step))
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


def run(subset, streams, epochs, tag, device, batch_size=24, base_lr=0.1,
        warmup=20, step=(150, 200), keep_rate=0.9, window_size=120, save_logits=False,
        eval_every=5, seed=0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    bone = streams in ('bone', 'bonemotion')
    motion = streams in ('motion', 'bonemotion')

    Xtr, ytr = load_split_raw('train', subset)
    Xva, yva = load_split_raw('val', subset)
    Xte, yte = load_split_raw('test', subset)
    print(f"[DSTA WLASL{subset} {streams}] train {len(ytr)} val {len(yva)} test {len(yte)}", flush=True)

    tr = DataLoader(DSTAFeeder(Xtr, ytr, True, window_size, bone, motion),
                    batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    va = DataLoader(DSTAFeeder(Xva, yva, False, window_size, bone, motion), batch_size=batch_size, num_workers=2)
    te = DataLoader(DSTAFeeder(Xte, yte, False, window_size, bone, motion), batch_size=batch_size, num_workers=2)

    model = Model(num_class=subset, num_point=27, num_person=1, groups=16, block_size=41,
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
            out = model(Xb.to(device), keep_prob)
            loss = loss_fn(out, yb.to(device))
            loss.backward()
            opt.step()
        if (ep % eval_every == 0 or ep == epochs - 1) and len(yva):
            v1, _, _ = evaluate(model, va, device, subset)
            if v1 > best_val:
                best_val = v1
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            print(f"  ep{ep:3d} lr {lr:.4f} keep_prob {keep_prob:.3f} val {v1:.2f} "
                  f"(best {best_val:.2f}) [{(time.time()-t0)/60:.1f}min]", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    t1, t5, pc = evaluate(model, te, device, subset)
    print(f">>> DSTA WLASL{subset}[{tag}] top1={t1:.2f}% top5={t5:.2f}% P-C={pc:.2f}% "
          f"(val {best_val:.2f}) n={len(yte)}", flush=True)
    rd = ROOT / 'benchmarks' / 'dstanet'; rd.mkdir(parents=True, exist_ok=True)
    res = {'subset': subset, 'streams': streams, 'tag': tag, 'test_top1': round(t1, 2),
           'test_top5': round(t5, 2), 'test_perclass': round(pc, 2), 'best_val': round(best_val, 2),
           'test_n': len(yte), 'epochs': epochs}
    (rd / f'wlasl{subset}_{streams}_{tag}.json').write_text(json.dumps(res, indent=2))
    if save_logits:
        model.eval(); L = []
        with torch.no_grad():
            for Xb, yb in te:
                L.append(model(Xb.to(device)).cpu())
        np.save(rd / f'logits_wlasl{subset}_{streams}_{tag}.npy', torch.cat(L).numpy())
        np.save(rd / f'ytest_wlasl{subset}.npy', np.array(yte))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--streams', type=str, default='joint', choices=['joint', 'bone', 'motion', 'bonemotion'])
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--batch_size', type=int, default=24)
    ap.add_argument('--tag', type=str, default='v1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--logits', action='store_true')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[device {dev}]", flush=True)
    t0 = time.time()
    run(a.subset, a.streams, a.epochs, a.tag, dev, a.batch_size, save_logits=a.logits,
        eval_every=a.eval_every, seed=a.seed)
    print(f"[done {(time.time()-t0)/60:.1f} min]", flush=True)
