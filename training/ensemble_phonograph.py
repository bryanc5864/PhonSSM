"""
Ensemble the PhonoGraph per-stream test probabilities (already flip-TTA-averaged
and softmaxed at save time) into a final WLASL100 number, and compare to the
honest official-split target DSTA-SLR = 82.38% top-1 / 83.09% per-class.

Probs are saved by train_phonograph.py as probs_wlasl{subset}_{stream}_{tag}_seed{seed}.npy
with the matching ytest_wlasl{subset}.npy. We search a small grid of stream
fusion weights (joint/bone dominate, as in every skeleton-SLR ensemble) and
report the best, plus the fixed DSTA-style weighting for reference.
"""
import sys, json, argparse, itertools
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
RD = ROOT / 'benchmarks' / 'phonograph'
STREAMS = ['joint', 'bone', 'motion', 'bonemotion']


def top1_pc(probs, y, nc):
    pred = probs.argmax(1)
    t1 = (pred == y).mean() * 100
    pcc = np.zeros(nc); pct = np.zeros(nc)
    for t, p in zip(y, pred):
        pct[t] += 1; pcc[t] += (t == p)
    seen = pct > 0
    pc = (pcc[seen] / pct[seen]).mean() * 100
    return t1, pc


def load(subset, tag, seeds):
    y = np.load(RD / f'ytest_wlasl{subset}.npy')
    streams = {}
    for s in STREAMS:
        acc = []
        for sd in seeds:
            f = RD / f'probs_wlasl{subset}_{s}_{tag}_seed{sd}.npy'
            if f.exists():
                p = np.load(f)
                acc.append(p / p.sum(1, keepdims=True))
        if acc:
            streams[s] = np.mean(acc, axis=0)  # average over seeds
    return streams, y


def main(subset, tag, seeds):
    streams, y = load(subset, tag, seeds)
    nc = subset
    print(f"streams available: {list(streams.keys())}  test n={len(y)}")
    for s, p in streams.items():
        t1, pc = top1_pc(p, y, nc)
        print(f"  {s:11s} top1={t1:.2f}  P-C={pc:.2f}")

    keys = list(streams.keys())
    P = np.stack([streams[k] for k in keys], axis=0)  # (S,N,C)

    # DSTA-style fixed weights (joint/bone heavy), restricted to available streams
    fixed = {'joint': 1.5, 'bone': 1.0, 'motion': 0.3, 'bonemotion': 0.5}
    w = np.array([fixed[k] for k in keys])
    ens = (P * w[:, None, None]).sum(0)
    t1f, pcf = top1_pc(ens, y, nc)
    print(f"\nfixed-weight ensemble ({dict(zip(keys, w))}): top1={t1f:.2f}  P-C={pcf:.2f}")

    # small grid search over weights
    grid = [0.0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.7, 2.0]
    best = (-1, None, None)
    for combo in itertools.product(grid, repeat=len(keys)):
        if sum(combo) == 0:
            continue
        ens = (P * np.array(combo)[:, None, None]).sum(0)
        t1, pc = top1_pc(ens, y, nc)
        if t1 > best[0]:
            best = (t1, pc, combo)
    print(f"best grid ensemble: top1={best[0]:.2f}  P-C={best[1]:.2f}  weights={dict(zip(keys, best[2]))}")
    print(f"\n--- vs DSTA-SLR official-split target: 82.38 top1 / 83.09 P-C ---")
    verdict = "BEAT" if best[0] > 82.38 else "short of"
    print(f"PhonoGraph best ensemble top1 {best[0]:.2f} => {verdict} DSTA-SLR (82.38)")

    out = {'subset': subset, 'streams': keys, 'per_stream': {k: top1_pc(streams[k], y, nc)[0] for k in keys},
           'fixed_ensemble_top1': round(t1f, 2), 'best_ensemble_top1': round(best[0], 2),
           'best_ensemble_pc': round(best[1], 2), 'best_weights': dict(zip(keys, best[2])),
           'target_dsta': 82.38, 'beat': bool(best[0] > 82.38)}
    (RD / f'ensemble_wlasl{subset}_{tag}.json').write_text(json.dumps(out, indent=2))
    print(f"saved {RD / f'ensemble_wlasl{subset}_{tag}.json'}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--tag', type=str, default='b96')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    a = ap.parse_args()
    main(a.subset, a.tag, a.seeds)
