"""
Strong 27-joint HRNet ISLR model — the revised PhonSSM taken to its logical end:
DROP the harmful unsupervised PDM/HPC path (a simple BiGRU already beat full
PhonSSM 70.5% vs ~40% on the same official-split HRNet features) and keep the
pieces that help — graph-aware multi-stream input + learnable-scale cosine head.

Model: multi-stream (joint + bone + motion, bone via the sign_27 skeleton) ->
per-frame graph-linear -> deep BiGRU -> attention pool -> cosine (learnable scale).
Supports single-stream selection so we can 4-stream ENSEMBLE (DSTA style) to beat
the 82.4% WLASL100 SOTA.
"""
import os, sys, math, time, argparse, copy, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from training.train_sign27 import load_split, normalize_sign27
from training.train_official import EMA, evaluate
from models.phonssm.agan import create_sign27_adjacency

A27 = create_sign27_adjacency().numpy()
An27 = A27 / (A27.sum(1, keepdims=True) + 1e-6)


def make_streams(X, streams):
    """X: (N,T,27,2) -> (N,T,27, 2*len(streams)) for the requested streams."""
    feats = []
    if 'joint' in streams:
        feats.append(X)
    if 'bone' in streams:
        neigh = np.einsum('vw,ntwc->ntvc', An27, X)
        feats.append(X - neigh)
    if 'motion' in streams:
        mot = np.zeros_like(X); mot[:, 1:] = X[:, 1:] - X[:, :-1]
        feats.append(mot)
    if 'bonemotion' in streams:
        neigh = np.einsum('vw,ntwc->ntvc', An27, X)
        bone = X - neigh
        bm = np.zeros_like(bone); bm[:, 1:] = bone[:, 1:] - bone[:, :-1]
        feats.append(bm)
    out = np.concatenate(feats, axis=-1)
    valid = ~np.all(X == 0, axis=3, keepdims=True)   # gate derived streams by joint validity
    return (out * valid).astype(np.float32)


class DS(Dataset):
    def __init__(self, X, y, aug=False):
        self.X, self.y, self.aug = X, y, aug

    def __len__(self): return len(self.y)

    def _augment(self, s):  # s (T,27,C)
        s = s.copy(); valid = ~np.all(s == 0, axis=2)
        th = np.random.uniform(-0.26, 0.26); c, sn = math.cos(th), math.sin(th)
        for k in range(0, s.shape[-1], 2):
            x, y = s[..., k].copy(), s[..., k + 1].copy()
            s[..., k] = c * x - sn * y; s[..., k + 1] = sn * x + c * y
        s *= np.random.uniform(0.9, 1.1)
        s[..., 0::2] += np.random.uniform(-0.05, 0.05)
        s[..., 1::2] += np.random.uniform(-0.05, 0.05)
        s += np.random.normal(0, 0.01, s.shape).astype(np.float32)
        s[~valid] = 0.0
        return np.clip(s, -6, 6).astype(np.float32)

    def __getitem__(self, i):
        x = self.X[i]
        if self.aug: x = self._augment(x)
        return torch.from_numpy(x.reshape(x.shape[0], -1)), int(self.y[i])


class StrongNet(nn.Module):
    def __init__(self, in_ch, nc, h=384, layers=3, drop=0.3):
        super().__init__()
        self.V = 27; self.in_ch = in_ch
        self.register_buffer('A', torch.tensor((A27 + np.eye(27)) / (A27 + np.eye(27)).sum(1, keepdims=True), dtype=torch.float32))
        self.gin = nn.Linear(in_ch, 64)                      # per-joint embed
        self.gproj = nn.Sequential(nn.LayerNorm(64), nn.GELU())
        self.inp = nn.Sequential(nn.Linear(27 * 64, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(drop))
        self.gru = nn.GRU(h, h, num_layers=layers, batch_first=True, bidirectional=True, dropout=drop)
        self.q = nn.Parameter(torch.randn(2 * h) * 0.02)
        self.head_norm = nn.LayerNorm(2 * h)
        self.proto = nn.Parameter(torch.randn(nc, 2 * h)); nn.init.xavier_uniform_(self.proto)
        self.scale = nn.Parameter(torch.tensor(math.log(16.0)))

    def forward(self, x):                                     # x (B,T,27*in_ch)
        B, T, _ = x.shape
        x = x.view(B, T, self.V, self.in_ch)
        h = self.gproj(self.gin(x))                          # (B,T,V,64)
        h = torch.einsum('vw,btwc->btvc', self.A, h)         # graph smoothing over joints
        h = self.inp(h.reshape(B, T, -1))                    # (B,T,h)
        h, _ = self.gru(h)
        w = F.softmax((h @ self.q) / h.shape[-1] ** 0.5, dim=1).unsqueeze(-1)
        e = self.head_norm((h * w).sum(1))
        logits = F.normalize(e, dim=-1) @ F.normalize(self.proto, dim=-1).T
        return {'logits': logits * self.scale.exp().clamp(4, 64), 'emb': e}


def run(subset, epochs, frames, streams, device, tag, seed=0, return_logits=False):
    torch.manual_seed(seed); np.random.seed(seed)
    st = streams.split(',')
    def prep(sp):
        X, y = load_split(sp, subset, frames); X = normalize_sign27(X); return make_streams(X, st), y
    Xtr, ytr = prep('train'); Xva, yva = prep('val'); Xte, yte = prep('test')
    in_ch = Xtr.shape[-1]
    print(f"[strong WLASL{subset} {streams}] train {len(ytr)} val {len(yva)} test {len(yte)} in_ch={in_ch}", flush=True)
    tr = DataLoader(DS(Xtr, ytr, True), batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(DS(Xva, yva), batch_size=256); te = DataLoader(DS(Xte, yte), batch_size=256)
    m = StrongNet(in_ch, subset).to(device)
    print('  params %.2fM' % (sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, epochs=epochs, steps_per_epoch=len(tr), pct_start=0.1)
    ema = EMA(m, 0.99); best = -1; best_state = None
    for ep in range(epochs):
        m.train()
        for Xb, yb in tr:
            opt.zero_grad()
            F.cross_entropy(m(Xb.to(device))['logits'], yb.to(device), label_smoothing=0.1).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step(); ema.update(m)
        bk = copy.deepcopy(m.state_dict()); m.load_state_dict(ema.shadow)
        v1, _, _ = evaluate(m, va, device, subset)
        if v1 > best: best = v1; best_state = {k: t.clone() for k, t in ema.shadow.items()}
        m.load_state_dict(bk)
        if ep % 10 == 0 or ep == epochs - 1: print(f"  ep{ep:3d} val {v1:.2f} (best {best:.2f})", flush=True)
    m.load_state_dict(best_state)
    t1, t5, pc = evaluate(m, te, device, subset)
    print(f">>> strong WLASL{subset}[{tag}] top1={t1:.2f}% top5={t5:.2f}% P-C={pc:.2f}% (val {best:.2f}) n={len(yte)}", flush=True)
    res = {'subset': subset, 'streams': streams, 'tag': tag, 'test_top1': round(t1, 2),
           'test_top5': round(t5, 2), 'test_perclass': round(pc, 2), 'best_val': round(best, 2), 'test_n': len(yte)}
    rd = ROOT / 'benchmarks' / 'strong27'; rd.mkdir(parents=True, exist_ok=True)
    (rd / f'wlasl{subset}_{tag}.json').write_text(json.dumps(res, indent=2))
    if return_logits:
        m.eval(); allL = []
        with torch.no_grad():
            for Xb, yb in te: allL.append(m(Xb.to(device))['logits'].cpu())
        np.save(rd / f'logits_wlasl{subset}_{tag}.npy', torch.cat(allL).numpy())
        np.save(rd / f'ytest_wlasl{subset}.npy', np.array(yte))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--streams', type=str, default='joint,bone,motion')
    ap.add_argument('--tag', type=str, default='multi')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--logits', action='store_true')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    run(a.subset, a.epochs, a.frames, a.streams, dev, a.tag, a.seed, a.logits)
    print(f"[done {(time.time()-t0)/60:.1f} min]", flush=True)
