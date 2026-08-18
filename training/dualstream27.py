"""
Anatomical dual-stream + dual-path temporal architecture for 27-joint HRNet ISLR.

Novel over both DSTA-Net (single unified graph-attention stream) and our earlier
PhonSSM: splits joints along a REAL anatomical/phonological line — a body-pose
stream (nose+shoulders+elbows+wrists: carries sign LOCATION and ORIENTATION) and
a hands stream (20 hand joints: carries HANDSHAPE and MOVEMENT) — each with its
own per-frame encoder, fused per-frame via cross-attention (not concatenation),
then a dual-path temporal encoder: multi-scale dilated causal conv (local detail,
e.g. abrupt handshape transitions) run in parallel with a BiGRU (global context,
the ingredient already proven to work at 69-74%), gated together. Cosine head
with learnable scale (kept from the earlier fix).

sign27 layout: 0 nose, 1/2 shoulders, 3/4 elbows, 5/6 wrists (body, 7 joints),
7-16 left hand, 17-26 right hand (hands, 20 joints).
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

BODY_IDX = list(range(0, 7))
HAND_IDX = list(range(7, 27))


class CrossAttnFuse(nn.Module):
    def __init__(self, d, heads=4, drop=0.2):
        super().__init__()
        self.b2h = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.h2b = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.nb = nn.LayerNorm(d); self.nh = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))

    def forward(self, body, hands):                       # each (B,V,d) per-frame node embeddings
        b2 = self.nb(body + self.b2h(body, hands, hands, need_weights=False)[0])
        h2 = self.nh(hands + self.h2b(hands, body, body, need_weights=False)[0])
        return self.out(torch.cat([b2.mean(1), h2.mean(1)], dim=-1))  # (B,d) per-frame fused


class DilatedTemporal(nn.Module):
    """Multi-scale dilated conv over time — local motion detail."""
    def __init__(self, d, drop=0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d, d // 2, kernel_size=k, padding=(k - 1) * dil // 2, dilation=dil)
            for k, dil in [(3, 1), (3, 2), (5, 2)]
        ])
        self.proj = nn.Sequential(nn.Linear(3 * (d // 2), d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))

    def forward(self, x):                                  # (B,T,d)
        xt = x.transpose(1, 2)                              # (B,d,T)
        outs = [F.gelu(c(xt))[..., :x.shape[1]].transpose(1, 2) for c in self.convs]
        return self.proj(torch.cat(outs, dim=-1))            # (B,T,d)


class DualPathTemporal(nn.Module):
    def __init__(self, d, drop=0.2):
        super().__init__()
        self.conv_path = DilatedTemporal(d, drop)
        self.gru = nn.GRU(d, d, num_layers=2, batch_first=True, bidirectional=True, dropout=drop)
        self.gru_proj = nn.Linear(2 * d, d)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.norm = nn.LayerNorm(d)

    def forward(self, x):                                  # (B,T,d)
        c = self.conv_path(x)
        g, _ = self.gru(x); g = self.gru_proj(g)
        gate = self.gate(torch.cat([c, g], dim=-1))
        return self.norm(x + gate * c + (1 - gate) * g)


class DualStreamNet(nn.Module):
    def __init__(self, in_ch, nc, d=192, temporal_layers=2, drop=0.3):
        super().__init__()
        self.in_ch = in_ch
        self.body_embed = nn.Sequential(nn.Linear(in_ch, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.hand_embed = nn.Sequential(nn.Linear(in_ch, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.body_pos = nn.Parameter(torch.randn(1, len(BODY_IDX), d) * 0.02)
        self.hand_pos = nn.Parameter(torch.randn(1, len(HAND_IDX), d) * 0.02)
        self.fuse = CrossAttnFuse(d, drop=drop)
        self.temporal = nn.ModuleList([DualPathTemporal(d, drop) for _ in range(temporal_layers)])
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.out_norm = nn.LayerNorm(d)
        self.proto = nn.Parameter(torch.randn(nc, d)); nn.init.xavier_uniform_(self.proto)
        self.scale = nn.Parameter(torch.tensor(math.log(16.0)))

    def forward(self, x):                                  # x (B,T,27*in_ch)
        B, T, _ = x.shape
        x = x.view(B, T, 27, self.in_ch)
        body = self.body_embed(x[:, :, BODY_IDX]) + self.body_pos   # (B,T,7,d)
        hand = self.hand_embed(x[:, :, HAND_IDX]) + self.hand_pos   # (B,T,20,d)
        d = body.shape[-1]
        fused = self.fuse(body.reshape(B * T, 7, d), hand.reshape(B * T, 20, d))  # (B*T,d)
        h = fused.view(B, T, d)
        for layer in self.temporal:
            h = layer(h)
        w = F.softmax((h @ self.q) / d ** 0.5, dim=1).unsqueeze(-1)
        e = self.out_norm((h * w).sum(1))
        logits = F.normalize(e, dim=-1) @ F.normalize(self.proto, dim=-1).T
        return {'logits': logits * self.scale.exp().clamp(4, 64)}


def run(subset, epochs, frames, streams, device, tag, seed=0, d=192, t_layers=2, save_logits=False, batch_size=64):
    torch.manual_seed(seed); np.random.seed(seed)
    st = streams.split(',')
    def prep(sp):
        X, y = load_split(sp, subset, frames); return make_streams(normalize_sign27(X), st), y
    Xtr, ytr = prep('train'); Xva, yva = prep('val'); Xte, yte = prep('test')
    in_ch = Xtr.shape[-1]
    print(f"[dual WLASL{subset} {streams}] train {len(ytr)} val {len(yva)} test {len(yte)} in_ch={in_ch}", flush=True)
    tr = DataLoader(DS(Xtr, ytr, True), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(DS(Xva, yva), batch_size=128); te = DataLoader(DS(Xte, yte), batch_size=128)
    m = DualStreamNet(in_ch, subset, d=d, temporal_layers=t_layers).to(device)
    print('  params %.2fM' % (sum(p.numel() for p in m.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=8e-4, weight_decay=3e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 8e-4, epochs=epochs, steps_per_epoch=len(tr), pct_start=0.1)
    ema = EMA(m, 0.99); best = -1; best_state = None
    t0 = time.time()
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
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d} val {v1:.2f} (best {best:.2f}) [{(time.time()-t0)/60:.1f}min]", flush=True)
    m.load_state_dict(best_state)
    t1, t5, pc = evaluate(m, te, device, subset)
    print(f">>> dual WLASL{subset}[{tag}] top1={t1:.2f}% top5={t5:.2f}% P-C={pc:.2f}% (val {best:.2f}) n={len(yte)}", flush=True)
    rd = ROOT / 'benchmarks' / 'dualstream27'; rd.mkdir(parents=True, exist_ok=True)
    res = {'subset': subset, 'streams': streams, 'tag': tag, 'test_top1': round(t1, 2),
           'test_top5': round(t5, 2), 'test_perclass': round(pc, 2), 'best_val': round(best, 2), 'test_n': len(yte)}
    (rd / f'wlasl{subset}_{tag}.json').write_text(json.dumps(res, indent=2))
    if save_logits:
        m.eval(); L = []
        with torch.no_grad():
            for Xb, yb in te: L.append(m(Xb.to(device))['logits'].cpu())
        np.save(rd / f'logits_wlasl{subset}_{tag}.npy', torch.cat(L).numpy())
        np.save(rd / f'ytest_wlasl{subset}.npy', np.array(yte))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=100); ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--frames', type=int, default=120); ap.add_argument('--streams', type=str, default='joint,bone,motion')
    ap.add_argument('--tag', type=str, default='dual'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--d', type=int, default=192); ap.add_argument('--t_layers', type=int, default=2)
    ap.add_argument('--logits', action='store_true'); ap.add_argument('--batch_size', type=int, default=64)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    run(a.subset, a.epochs, a.frames, a.streams, dev, a.tag, a.seed, a.d, a.t_layers, a.logits, a.batch_size)
    print(f"[done {(time.time()-t0)/60:.1f} min]", flush=True)
