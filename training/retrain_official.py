"""
Honest WLASL re-run: train a fresh PhonSSM on the OFFICIAL train split and
evaluate on the OFFICIAL test split (built by build_wlasl_official.py, with
source-video dedup + leakage guard). Mirrors benchmark_external's optimizer/loss.
"""
import os, sys, json, argparse, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.phonssm import PhonSSM, PhonSSMConfig
from training.benchmark_external import evaluate, calculate_accuracy

def temp_for(n):
    return 1.0 if n <= 100 else 0.5 if n <= 500 else 0.3 if n <= 1000 else 0.1

def load(split_dir, name):
    d = Path(split_dir)
    X = torch.tensor(np.load(d / f'X_{name}.npy'))
    y = torch.tensor(np.load(d / f'y_{name}.npy')).long()
    return X, y

def main(subset, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[device: {device}{' — ' + torch.cuda.get_device_name(0) if device=='cuda' else ''}]")
    sd = ROOT / 'data' / 'processed' / 'wlasl_official' / f'wlasl{subset}'
    meta = json.load(open(sd / 'meta.json'))
    Xtr, ytr = load(sd, 'train'); Xva, yva = load(sd, 'val'); Xte, yte = load(sd, 'test')
    print(f"WLASL{subset} OFFICIAL: train {len(ytr)} | val {len(yva)} | test {len(yte)} "
          f"(test videos {meta['test_videos']}, classes {meta['classes_in_test']}/{subset})")

    cfg = PhonSSMConfig(num_signs=subset, temperature=temp_for(subset),
                        input_mode='pose_hands', num_landmarks=75)
    model = PhonSSM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True)
    va = DataLoader(TensorDataset(Xva, yva), batch_size=128) if len(yva) else None
    te = DataLoader(TensorDataset(Xte, yte), batch_size=128)

    best_val, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for Xb, yb in tr:
            opt.zero_grad()
            logits = model(Xb.to(device))['logits']
            loss = F.cross_entropy(logits, yb.to(device), label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if va is not None:
            v = evaluate(model, va, device)['top1'] * 100
            if v > best_val:
                best_val = v; best_state = {k: t.clone() for k, t in model.state_dict().items()}
            if ep % 10 == 0 or ep == epochs - 1:
                print(f"  ep{ep:3d}  val {v:.2f}%  (best {best_val:.2f}%)")
    if best_state is not None:
        model.load_state_dict(best_state)
    res = evaluate(model, te, device)
    out = {'subset': subset, 'epochs': epochs, 'seed': seed,
           'official_test_top1': round(res['top1'] * 100, 2),
           'official_test_top5': round(res['top5'] * 100, 2),
           'best_val_top1': round(best_val, 2),
           'test_n': int(len(yte)), 'test_videos': meta['test_videos']}
    print(f"\n>>> WLASL{subset} OFFICIAL-SPLIT test top1 = {out['official_test_top1']}%  "
          f"(top5 {out['official_test_top5']}%, n={out['test_n']})")
    res_path = ROOT / 'benchmarks' / f'official_wlasl{subset}.json'
    res_path.write_text(json.dumps(out, indent=2))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    main(a.subset, a.epochs, a.seed)
    print(f"[done in {(time.time()-t0)/60:.1f} min]")
