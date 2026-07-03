"""
Independent re-run to verify paper WLASL numbers from saved checkpoints.
Reuses the repo's OWN load_wlasl_splits + evaluate for faithful reproduction.
Reports val AND test top-1 for the reproduced (random_state=42) split.
"""
import os, sys, glob, json, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from models.phonssm import PhonSSM, PhonSSMConfig
from training.benchmark_external import load_wlasl_splits, evaluate

PAPER = {100: 88.37, 300: 74.41, 1000: 62.90, 2000: 72.08}
CKPT = {
    100:  'benchmarks/external/wlasl100/20260118_073336/best_model.pt',
    300:  'benchmarks/external/wlasl300/20260118_104801/best_model.pt',
    1000: 'benchmarks/external/wlasl1000/20260120_060624/best_model.pt',
    2000: 'benchmarks/external/wlasl2000/20260119_020829/best_model.pt',
}

def temp_for(n):
    return 1.0 if n <= 100 else 0.5 if n <= 500 else 0.3 if n <= 1000 else 0.1

def run(split):
    device = 'cpu'
    data = load_wlasl_splits(subset_size=split, use_pose_hands=True)
    cfg = PhonSSMConfig(num_signs=data['num_classes'], temperature=temp_for(split),
                        input_mode=data['input_mode'], num_landmarks=data['num_features'] // 3)
    model = PhonSSM(cfg).to(device)
    ck = torch.load(ROOT / CKPT[split], map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck['model_state_dict'], strict=False)
    if missing or unexpected:
        print(f"  [load] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    out = {}
    for name in ('val', 'test'):
        X = torch.tensor(np.asarray(data[f'X_{name}'], dtype=np.float32))
        y = torch.tensor(np.asarray(data[f'y_{name}'], dtype=np.int64))
        dl = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=False)
        res = evaluate(model, dl, device)
        out[name] = (res['top1'] * 100, res['top5'] * 100, len(y))
    stored = ck.get('val_acc')
    print(f"\n=== WLASL{split} ===")
    print(f"  checkpoint: {CKPT[split]}")
    print(f"  stored val_acc : {stored*100:.2f}%" if stored else "  stored val_acc : n/a")
    print(f"  RE-RUN val     : top1 {out['val'][0]:.2f}%  top5 {out['val'][1]:.2f}%  (n={out['val'][2]})")
    print(f"  RE-RUN test    : top1 {out['test'][0]:.2f}%  top5 {out['test'][1]:.2f}%  (n={out['test'][2]})")
    print(f"  paper (PhonSSM): {PAPER[split]:.2f}%")
    return {'split': split, 'stored': round(stored*100,2) if stored else None,
            'rerun_val': round(out['val'][0],2), 'rerun_test': round(out['test'][0],2),
            'paper': PAPER[split]}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='100')
    args = ap.parse_args()
    splits = [int(s) for s in args.splits.split(',')]
    torch.manual_seed(0)
    summary = [run(s) for s in splits]
    print("\n==== SUMMARY (top-1 %) ====")
    print(f"{'split':>10} {'stored_val':>11} {'rerun_val':>10} {'rerun_test':>11} {'paper':>7}")
    for r in summary:
        print(f"WLASL{r['split']:<5} {str(r['stored']):>11} {r['rerun_val']:>10} {r['rerun_test']:>11} {r['paper']:>7}")
    (ROOT / 'benchmarks' / 'rerun_verify_results.json').write_text(json.dumps(summary, indent=2))
