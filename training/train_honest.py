"""
Train PhonSSM on the LEAKAGE-FREE honest split (built by build_honest_split.py).
Identical model/optimizer/schedule to repro_leaky.py so the ONLY difference vs
the reported numbers is the split (video-grouped, no triplet straddles
train/test) -- isolating the inflation.
"""
import os, sys, json, time, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.phonssm import PhonSSM, PhonSSMConfig
from training.repro_leaky import evaluate, temp_for, PAPER

SPLIT = ROOT / 'data' / 'processed' / 'wlasl_honest'


def load(d, name):
    return (np.load(d / f'X_{name}.npy'), np.load(d / f'y_{name}.npy'))


def run(subset, epochs, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    d = SPLIT / f'wlasl{subset}'
    meta = json.load(open(d / 'meta.json'))
    Xtr, ytr = load(d, 'train'); Xva, yva = load(d, 'val'); Xte, yte = load(d, 'test')
    print(f"[WLASL{subset} HONEST] train {len(ytr)} val {len(yva)} test {len(yte)} "
          f"| test near-dup-in-train {meta['test_rows_near_train']}/{meta['test_rows']}", flush=True)

    def dl(X, y, bs, sh):
        return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                        torch.tensor(y, dtype=torch.long)), batch_size=bs, shuffle=sh)
    tr = dl(Xtr, ytr, 128, True); va = dl(Xva, yva, 128, False); te = dl(Xte, yte, 64, False)

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
        if ep < warmup: wsched.step()
        else: msched.step(v1)
        if v1 > best_val:
            best_val = v1; best_state = {k: t.clone() for k, t in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d} val {v1:.2f} (best {best_val:.2f})", flush=True)
        if patience >= PAT:
            print(f"  early stop @ep{ep}", flush=True); break
    model.load_state_dict(best_state)
    t1, t5 = evaluate(model, te, device)
    out = {'subset': subset, 'protocol': 'honest_video_grouped_official_ratios',
           'test_top1': round(t1, 2), 'test_top5': round(t5, 2), 'best_val_top1': round(best_val, 2),
           'test_n': int(len(yte)), 'paper_top1': PAPER.get(subset),
           'leaky_repro': None, 'epochs_run': ep + 1,
           'test_near_dup_in_train': meta['test_rows_near_train']}
    lk = ROOT / 'benchmarks' / f'repro_leaky_wlasl{subset}.json'
    if lk.exists():
        out['leaky_repro'] = json.load(open(lk))['test_top1']
    print(f">>> WLASL{subset} HONEST test top1={t1:.2f}% top5={t5:.2f}% "
          f"(leaky-repro {out['leaky_repro']}, paper {PAPER.get(subset)}) n={len(yte)}", flush=True)
    (ROOT / 'benchmarks' / f'honest_wlasl{subset}.json').write_text(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[device {dev}]", flush=True)
    t0 = time.time()
    run(a.subset, a.epochs, a.seed, dev)
    print(f"[done {(time.time()-t0)/60:.1f} min]", flush=True)
