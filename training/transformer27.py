"""
Spatial-Temporal Transformer for 27-joint HRNet ISLR — the architecture class
behind pose-only WLASL SOTA (decoupled spatial then temporal attention).
Joint-embed (+joint pos) -> spatial self-attn over 27 joints -> attn-pool joints
-> (+temporal pos) temporal self-attn over frames -> attn-pool time -> cosine head.
Strong regularization for the small WLASL train sets.
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


class Block(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d), nn.Dropout(drop))

    def forward(self, x):
        h = self.n1(x); x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class STTransformer(nn.Module):
    def __init__(self, in_ch, nc, V=27, d=128, heads=8, s_layers=2, t_layers=4, drop=0.3):
        super().__init__()
        self.V, self.d = V, d
        self.embed = nn.Linear(in_ch, d)
        self.joint_pos = nn.Parameter(torch.randn(1, V, d) * 0.02)
        self.spatial = nn.ModuleList([Block(d, heads, drop) for _ in range(s_layers)])
        self.sq = nn.Parameter(torch.randn(d) * 0.02)
        self.temp_pos = nn.Parameter(torch.randn(1, 400, d) * 0.02)
        self.temporal = nn.ModuleList([Block(d, heads, drop) for _ in range(t_layers)])
        self.tq = nn.Parameter(torch.randn(d) * 0.02)
        self.norm = nn.LayerNorm(d); self.drop = nn.Dropout(drop)
        self.proto = nn.Parameter(torch.randn(nc, d)); nn.init.xavier_uniform_(self.proto)
        self.scale = nn.Parameter(torch.tensor(math.log(16.0)))
        self.in_ch = in_ch

    def forward(self, x):                                 # x (B,T,V*in_ch)
        B, T, _ = x.shape
        x = x.view(B, T, self.V, self.in_ch)
        h = self.embed(x) + self.joint_pos               # (B,T,V,d)
        h = h.view(B * T, self.V, self.d)
        for blk in self.spatial:
            h = blk(h)
        w = F.softmax((h @ self.sq) / self.d ** 0.5, 1).unsqueeze(-1)
        h = (h * w).sum(1).view(B, T, self.d)            # (B,T,d) spatial-pooled
        h = h + self.temp_pos[:, :T]
        h = self.drop(h)
        for blk in self.temporal:
            h = blk(h)
        w = F.softmax((h @ self.tq) / self.d ** 0.5, 1).unsqueeze(-1)
        e = self.norm((h * w).sum(1))
        logits = F.normalize(e, dim=-1) @ F.normalize(self.proto, dim=-1).T
        return {'logits': logits * self.scale.exp().clamp(4, 64)}


def run(subset, epochs, frames, streams, device, tag, seed=0, save_logits=False):
    torch.manual_seed(seed); np.random.seed(seed)
    st = streams.split(',')
    def prep(sp):
        X, y = load_split(sp, subset, frames); return make_streams(normalize_sign27(X), st), y
    Xtr, ytr = prep('train'); Xva, yva = prep('val'); Xte, yte = prep('test')
    in_ch = Xtr.shape[-1]
    print(f"[transf WLASL{subset} {streams}] train {len(ytr)} val {len(yva)} test {len(yte)} in_ch={in_ch}", flush=True)
    tr = DataLoader(DS(Xtr, ytr, True), batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(DS(Xva, yva), batch_size=256); te = DataLoader(DS(Xte, yte), batch_size=256)
    m = STTransformer(in_ch, subset).to(device)
    print('  params %.2fM' % (sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=8e-4, weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 8e-4, epochs=epochs, steps_per_epoch=len(tr), pct_start=0.15)
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
    print(f">>> transf WLASL{subset}[{tag}] top1={t1:.2f}% top5={t5:.2f}% P-C={pc:.2f} (val {best:.2f}) n={len(yte)}", flush=True)
    rd = ROOT / 'benchmarks' / 'strong27'; rd.mkdir(parents=True, exist_ok=True)
    (rd / f'wlasl{subset}_{tag}.json').write_text(json.dumps(
        {'subset': subset, 'streams': streams, 'tag': tag, 'test_top1': round(t1, 2),
         'test_top5': round(t5, 2), 'test_perclass': round(pc, 2), 'best_val': round(best, 2), 'test_n': len(yte)}, indent=2))
    if save_logits:
        m.eval(); L = []
        with torch.no_grad():
            for Xb, yb in te: L.append(m(Xb.to(device))['logits'].cpu())
        np.save(rd / f'logits_wlasl{subset}_{tag}.npy', torch.cat(L).numpy())
        np.save(rd / f'ytest_wlasl{subset}.npy', np.array(yte))
    return {'top1': round(t1, 2)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=100); ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--frames', type=int, default=120); ap.add_argument('--streams', type=str, default='joint,bone,motion')
    ap.add_argument('--tag', type=str, default='tf'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--logits', action='store_true')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    run(a.subset, a.epochs, a.frames, a.streams, dev, a.tag, a.seed, a.logits)
    print(f"[done {(time.time()-t0)/60:.1f} min]", flush=True)
