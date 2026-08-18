"""
Faithful reproduction of the AS-PUBLISHED (leaky) WLASL protocol, from scratch,
on this machine's committed processed pool (X_wlasl_pose_hands.npy).

Purpose: confirm the reported WLASL100/300/1000/2000 numbers are reproducible
from the committed data+code, and observe live whether the impossible
monotonicity (acc(2000) >= acc(1000)) reappears -- the direct symptom of the
random-per-row split scattering the 3 copies (V1/V2/V3) of each video across
train/test.

Split logic mirrors origin/main:training/benchmark_external.py exactly:
  - subset = top-K glosses in WLASL_v0.3.json order
  - stratified random train_test_split(random_state=42) at the official RATIOS
Training mirrors the run configs (batch 128, AdamW lr 3e-4 wd 1e-2, 100 epochs,
warmup 5 + ReduceLROnPlateau, early stop patience 15, label smoothing 0.1,
adaptive temperature).
"""
import os, sys, json, time, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.phonssm import PhonSSM, PhonSSMConfig

WLASL_JSON = ROOT / 'data' / 'wlasl-processed' / 'WLASL_v0.3.json'
POOL_X = ROOT / 'data' / 'processed' / 'X_wlasl_pose_hands.npy'
POOL_Y = ROOT / 'data' / 'processed' / 'y_wlasl_pose_hands.npy'
POOL_MAP = ROOT / 'data' / 'processed' / 'wlasl_pose_hands_label_map.json'

PAPER = {100: 88.37, 300: 74.41, 1000: 62.90, 2000: 72.08}


def temp_for(n):
    return 1.0 if n <= 100 else 0.5 if n <= 500 else 0.3 if n <= 1000 else 0.1


def leaky_split(subset_size):
    wlasl = json.load(open(WLASL_JSON))
    subset = wlasl[:subset_size]
    subset_glosses = [e['gloss'].lower() for e in subset]
    gloss_to_idx = {g: i for i, g in enumerate(subset_glosses)}
    # official ratios
    c = {'train': 0, 'val': 0, 'test': 0}
    for e in subset:
        for inst in e['instances']:
            c[inst.get('split', 'train')] += 1
    tot = sum(c.values())
    train_ratio = c['train'] / tot
    val_ratio = c['val'] / tot

    X_all = np.load(POOL_X)
    y_all = np.load(POOL_Y)
    full_map = json.load(open(POOL_MAP))
    idx_to_gloss = {v: k for k, v in full_map.items()}
    mask = np.array([idx_to_gloss.get(int(l), '') in gloss_to_idx for l in y_all])
    X = X_all[mask]
    y = np.array([gloss_to_idx[idx_to_gloss[int(l)]] for l in y_all[mask]])

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=1 - train_ratio, stratify=y, random_state=42)
    vt = val_ratio / (1 - train_ratio)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=1 - vt, stratify=y_tmp, random_state=42)
    return (X_tr, y_tr, X_va, y_va, X_te, y_te,
            {'official_ratios': c, 'n_total': int(len(y)),
             'expected_videos': tot // 3})


def evaluate(model, dl, device):
    model.eval()
    tot = 0; c1 = 0; c5 = 0
    with torch.no_grad():
        for Xb, yb in dl:
            logits = model(Xb.to(device))['logits']
            yb = yb.to(device)
            _, top5 = logits.topk(5, dim=1)
            c1 += (top5[:, 0] == yb).sum().item()
            c5 += (top5 == yb[:, None]).any(1).sum().item()
            tot += yb.size(0)
    return c1 / tot * 100, c5 / tot * 100


def run(subset, epochs, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    X_tr, y_tr, X_va, y_va, X_te, y_te, meta = leaky_split(subset)
    print(f"[WLASL{subset}] pool rows {meta['n_total']} (~{meta['expected_videos']} videos x3) | "
          f"train {len(y_tr)} val {len(y_va)} test {len(y_te)}", flush=True)

    def dl(X, y, bs, sh):
        return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                        torch.tensor(y, dtype=torch.long)),
                          batch_size=bs, shuffle=sh)
    tr = dl(X_tr, y_tr, 128, True); va = dl(X_va, y_va, 128, False); te = dl(X_te, y_te, 64, False)

    cfg = PhonSSMConfig(num_signs=subset, temperature=temp_for(subset),
                        input_mode='pose_hands', num_landmarks=75)
    model = PhonSSM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    warmup = 5
    wsched = LambdaLR(opt, lambda ep: (ep + 1) / warmup)
    msched = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=7)

    best_val = -1; best_state = None; patience = 0; PAT = 15
    for ep in range(epochs):
        model.train()
        for Xb, yb in tr:
            opt.zero_grad()
            loss = F.cross_entropy(model(Xb.to(device))['logits'], yb.to(device), label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v1, _ = evaluate(model, va, device)
        if ep < warmup:
            wsched.step()
        else:
            msched.step(v1)
        if v1 > best_val:
            best_val = v1; best_state = {k: t.clone() for k, t in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d} val {v1:.2f} (best {best_val:.2f}) lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        if patience >= PAT:
            print(f"  early stop @ep{ep}", flush=True); break
    model.load_state_dict(best_state)
    t1, t5 = evaluate(model, te, device)
    out = {'subset': subset, 'protocol': 'leaky_random_per_row_seed42',
           'test_top1': round(t1, 2), 'test_top5': round(t5, 2),
           'best_val_top1': round(best_val, 2), 'test_n': int(len(y_te)),
           'paper_top1': PAPER.get(subset), 'epochs_run': ep + 1}
    print(f">>> WLASL{subset} LEAKY test top1={t1:.2f}% top5={t5:.2f}% (paper {PAPER.get(subset)}) n={len(y_te)}", flush=True)
    (ROOT / 'benchmarks' / f'repro_leaky_wlasl{subset}.json').write_text(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[device {dev} {torch.cuda.get_device_name(0) if dev=='cuda' else ''}]", flush=True)
    t0 = time.time()
    run(a.subset, a.epochs, a.seed, dev)
    print(f"[done {(time.time()-t0)/60:.1f} min]", flush=True)
