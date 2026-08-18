"""
Minimal strong skeleton baseline on the 27-joint HRNet features — a diagnostic to
separate "features vs model". If this simple BiGRU beats PhonSSM, the PhonSSM
PDM/HPC path is the bottleneck, not the features.
Per-frame MLP -> 2-layer BiGRU -> attention pool -> cosine (learnable-scale) head.
"""
import os, sys, math, time, argparse, copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from training.train_sign27 import load_split, normalize_sign27, S27
from training.train_strong27 import make_streams, DS
from training.train_official import EMA, evaluate


class GRUNet(nn.Module):
    def __init__(self, in_dim, nc, h=256, drop=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(drop))
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=drop)
        self.q = nn.Parameter(torch.randn(2 * h) * 0.02)
        self.proto = nn.Parameter(torch.randn(nc, 2 * h)); nn.init.xavier_uniform_(self.proto)
        self.scale = nn.Parameter(torch.tensor(math.log(16.0)))

    def forward(self, x):
        h = self.inp(x)
        h, _ = self.gru(h)
        w = F.softmax((h @ self.q) / h.shape[-1] ** 0.5, dim=1).unsqueeze(-1)
        e = (h * w).sum(1)
        logits = F.normalize(e, dim=-1) @ F.normalize(self.proto, dim=-1).T
        return {'logits': logits * self.scale.exp().clamp(4, 64)}


def main(subset, epochs, frames, device, streams='joint', tag='simple', seed=0, save_logits=False):
    torch.manual_seed(seed); np.random.seed(seed)
    st = streams.split(',')
    def prep(sp):
        X, y = load_split(sp, subset, frames); return make_streams(normalize_sign27(X), st), y
    Xtr, ytr = prep('train'); Xva, yva = prep('val'); Xte, yte = prep('test')
    in_ch = Xtr.shape[-1]
    print(f"[simple WLASL{subset} {streams}] train {len(ytr)} val {len(yva)} test {len(yte)} in_ch={in_ch}", flush=True)
    tr = DataLoader(DS(Xtr, ytr, True), batch_size=64, shuffle=True, num_workers=4)
    va = DataLoader(DS(Xva, yva), batch_size=256); te = DataLoader(DS(Xte, yte), batch_size=256)
    m = GRUNet(27 * in_ch, subset).to(device)
    print('  params %.2fM' % (sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, epochs=epochs, steps_per_epoch=len(tr))
    ema = EMA(m, 0.99); best = -1; best_state = None
    for ep in range(epochs):
        m.train()
        for Xb, yb in tr:
            opt.zero_grad()
            loss = F.cross_entropy(m(Xb.to(device))['logits'], yb.to(device), label_smoothing=0.1)
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step(); ema.update(m)
        bk = copy.deepcopy(m.state_dict()); m.load_state_dict(ema.shadow)
        v1, _, _ = evaluate(m, va, device, subset)
        if v1 > best: best = v1; best_state = {k: t.clone() for k, t in ema.shadow.items()}
        m.load_state_dict(bk)
        if ep % 10 == 0 or ep == epochs - 1: print(f"  ep{ep:3d} val {v1:.2f} (best {best:.2f})", flush=True)
    m.load_state_dict(best_state)
    t1, t5, pc = evaluate(m, te, device, subset)
    print(f">>> simple WLASL{subset}[{tag}] test top1={t1:.2f}% top5={t5:.2f}% P-C={pc:.2f} (val {best:.2f}) n={len(yte)}", flush=True)
    import json
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
    return {'test_top1': round(t1, 2)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=100); ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--streams', type=str, default='joint'); ap.add_argument('--tag', type=str, default='simple')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--logits', action='store_true')
    a = ap.parse_args()
    main(a.subset, a.epochs, a.frames, 'cuda' if torch.cuda.is_available() else 'cpu',
         a.streams, a.tag, a.seed, a.logits)
