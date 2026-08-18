"""Softmax-average logits from several stream-models (DSTA-style multi-stream
ensemble) and report official-split top-1/top-5 for a WLASL subset."""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
RD = ROOT / 'benchmarks' / 'strong27'


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z); return e / e.sum(1, keepdims=True)


def ensemble(subset, tags, weights=None):
    y = np.load(RD / f'ytest_wlasl{subset}.npy')
    probs = None
    w = weights or [1.0] * len(tags)
    used = []
    for tag, wi in zip(tags, w):
        f = RD / f'logits_wlasl{subset}_{tag}.npy'
        if not f.exists():
            print(f"  (skip missing {tag})"); continue
        p = softmax(np.load(f)) * wi
        probs = p if probs is None else probs + p
        used.append(tag)
    pred = probs.argmax(1)
    top5 = np.argsort(-probs, 1)[:, :5]
    t1 = float((pred == y).mean() * 100)
    t5 = float(np.mean([yi in row for yi, row in zip(y, top5)]) * 100)
    print(f"WLASL{subset} ENSEMBLE {used}: top1={t1:.2f}% top5={t5:.2f}% (n={len(y)})")
    res = {'subset': subset, 'streams': used, 'ensemble_top1': round(t1, 2), 'ensemble_top5': round(t5, 2), 'n': len(y)}
    (RD / f'ensemble_wlasl{subset}.json').write_text(json.dumps(res, indent=2))
    return res


if __name__ == '__main__':
    subset = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    tags = sys.argv[2].split(',') if len(sys.argv) > 2 else ['joint', 'bone', 'motion', 'jbm']
    ensemble(subset, tags)
