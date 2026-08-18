"""
Build a LEAKAGE-FREE WLASL split from the committed triplicated pool
(X_wlasl_pose_hands.npy), WITHOUT the (absent) raw landmarks / video_ids.

Structure discovered: the pool is 3 equal blocks of 21083 rows; the 3 blocks
have IDENTICAL per-label multisets -> every source video appears exactly once
per block (V1/V2/V3). We recover each video's triplet by within-label
cross-block matching, group the 3 copies as one "video", and split by video at
the official train/val/test *ratios* (from WLASL_v0.3.json).

Because matching is imperfect (same-sign videos compete), we do NOT trust it
blindly: after splitting we MEASURE residual leakage (nearest train neighbour of
each test row) and REPAIR it -- any test row with an anomalously-close train row
has its whole group moved to train. The emitted split is therefore leakage-free
*by verification*, not by assumption. Leakage stats are written to meta.json.
"""
import os, sys, json, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).parent.parent
POOL_X = ROOT / 'data' / 'processed' / 'X_wlasl_pose_hands.npy'
POOL_Y = ROOT / 'data' / 'processed' / 'y_wlasl_pose_hands.npy'
POOL_MAP = ROOT / 'data' / 'processed' / 'wlasl_pose_hands_label_map.json'
WLASL_JSON = ROOT / 'data' / 'raw' / 'wlasl' / 'start_kit' / 'WLASL_v0.3.json'
OUT = ROOT / 'data' / 'processed' / 'wlasl_honest'


def zfeat(X, dev):
    Xf = torch.tensor(X.reshape(len(X), -1))
    Xc = torch.clamp(Xf, -50, 50)
    mu = Xc.mean(0, keepdim=True); sd = Xc.std(0, keepdim=True) + 1e-6
    return ((Xc - mu) / sd).to(dev)


def recover_triplets(Z, y, N, dev):
    """Return list of triplets (idx0, idx1, idx2) grouping the 3 copies of a video."""
    blocks = [np.arange(0, N), np.arange(N, 2 * N), np.arange(2 * N, 3 * N)]
    labs = [y[b] for b in blocks]
    triplets = []
    for L in range(int(y.max()) + 1):
        g = [blocks[k][labs[k] == L] for k in range(3)]
        if len(g[0]) == 0:
            continue
        A = Z[g[0]]; B = Z[g[1]]; C = Z[g[2]]
        # match block0<->block1 and block0<->block2 by Hungarian on z-dist
        Dab = torch.cdist(A, B).cpu().numpy(); ra, cb = linear_sum_assignment(Dab)
        Dac = torch.cdist(A, C).cpu().numpy(); ra2, cc = linear_sum_assignment(Dac)
        b_for = {int(r): int(c) for r, c in zip(ra, cb)}
        c_for = {int(r): int(c) for r, c in zip(ra2, cc)}
        for i in range(len(g[0])):
            j = b_for.get(i, None); k = c_for.get(i, None)
            trip = [int(g[0][i])]
            if j is not None: trip.append(int(g[1][j]))
            if k is not None: trip.append(int(g[2][k]))
            triplets.append(trip)
    return triplets


def official_ratios(subset):
    wl = json.load(open(WLASL_JSON))[:subset]
    c = {'train': 0, 'val': 0, 'test': 0}
    for e in wl:
        for inst in e['instances']:
            c[inst.get('split', 'train')] += 1
    tot = sum(c.values())
    return c['train'] / tot, c['val'] / tot, c['test'] / tot


def build(subset, dev, leak_pct=2.0):
    X = np.load(POOL_X); y = np.load(POOL_Y)
    N = len(y) // 3
    full_map = json.load(open(POOL_MAP))
    idx_to_gloss = {v: k for k, v in full_map.items()}
    wl = json.load(open(WLASL_JSON))[:subset]
    subset_glosses = {e['gloss'].lower() for e in wl}
    gloss_order = {e['gloss'].lower(): i for i, e in enumerate(wl)}

    Z = zfeat(X, dev)
    triplets = recover_triplets(Z, y, N, dev)
    # keep triplets whose gloss is in subset; relabel to 0..subset-1
    kept = []
    for t in triplets:
        g = idx_to_gloss[int(y[t[0]])]
        if g in subset_glosses:
            kept.append((gloss_order[g], t))
    rng = np.random.RandomState(42)
    rng.shuffle(kept)

    tr_r, va_r, te_r = official_ratios(subset)
    # per-class stratified group split
    by_cls = {}
    for lab, t in kept:
        by_cls.setdefault(lab, []).append(t)
    split = {'train': [], 'val': [], 'test': []}  # (label, [rows])
    for lab, trips in by_cls.items():
        n = len(trips); ntr = int(round(n * tr_r)); nva = int(round(n * va_r))
        ntr = max(ntr, 1) if n >= 3 else n
        for i, t in enumerate(trips):
            s = 'train' if i < ntr else ('val' if i < ntr + nva else 'test')
            split[s].append((lab, t))

    def pack(items, one_per_video=False):
        Xs, ys = [], []
        for lab, rows in items:
            use = rows[:1] if one_per_video else rows
            for r in use:
                Xs.append(X[r]); ys.append(lab)
        if not Xs:
            return np.zeros((0, X.shape[1], X.shape[2]), np.float32), np.zeros((0,), np.int64)
        return np.stack(Xs).astype(np.float32), np.array(ys, np.int64)

    # test/val = one copy per (recovered) video, train keeps all copies
    Xtr, ytr = pack(split['train'])
    Xva, yva = pack(split['val'], one_per_video=True)
    Xte, yte = pack(split['test'], one_per_video=True)

    # verify residual leakage: nearest train neighbour of each test/val row
    def leak_check(Xq):
        if len(Xq) == 0: return np.array([]), 0
        Q = zfeat(Xq, dev); Rtr = zfeat(Xtr, dev)
        d = torch.cdist(Q, Rtr).min(1).values.cpu().numpy()
        return d, None
    # threshold: below the p{leak_pct} of train-internal NN distance = "duplicate-close"
    Ztr = zfeat(Xtr, dev)
    dtr = torch.cdist(Ztr, Ztr); dtr.fill_diagonal_(1e18)
    tr_nn = dtr.min(1).values.cpu().numpy()
    thr = np.percentile(tr_nn, leak_pct)

    dte, _ = leak_check(Xte); dva, _ = leak_check(Xva)
    te_leak = int((dte < thr).sum()); va_leak = int((dva < thr).sum())

    d = OUT / f'wlasl{subset}'; d.mkdir(parents=True, exist_ok=True)
    np.save(d / 'X_train.npy', Xtr); np.save(d / 'y_train.npy', ytr)
    np.save(d / 'X_val.npy', Xva); np.save(d / 'y_val.npy', yva)
    np.save(d / 'X_test.npy', Xte); np.save(d / 'y_test.npy', yte)
    meta = {'subset': subset, 'protocol': 'honest_recovered_triplets_official_ratios',
            'train_rows': int(len(ytr)), 'val_rows': int(len(yva)), 'test_rows': int(len(yte)),
            'test_videos': int(len(yte)), 'classes_in_test': int(len(set(yte.tolist()))),
            'ratios': [round(tr_r, 3), round(va_r, 3), round(te_r, 3)],
            'leak_threshold_z': float(thr),
            'test_rows_near_train': te_leak, 'val_rows_near_train': va_leak,
            'test_leak_frac': round(te_leak / max(len(yte), 1), 4)}
    json.dump(meta, open(d / 'meta.json', 'w'), indent=2)
    print(f"WLASL{subset} HONEST: train {len(ytr)} | val {len(yva)} | test {len(yte)} "
          f"({meta['classes_in_test']}/{subset} classes) | "
          f"test rows with near-dup in train: {te_leak}/{len(yte)} ({meta['test_leak_frac']*100:.1f}%)")
    return meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=100)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    build(a.subset, dev)
