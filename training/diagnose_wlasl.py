"""
Comprehensive diagnostic for WLASL100 training issues.
Checks data format, normalization, model behavior, and gradient flow.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.phonssm import PhonSSM, PhonSSMConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("\n" + "="*70)
print("DIAGNOSTIC 1: Compare WLASL data vs Training data format")
print("="*70)

# Load WLASL data
X_wlasl = np.load(PROJECT_ROOT / 'data/processed/X_wlasl.npy')
y_wlasl = np.load(PROJECT_ROOT / 'data/processed/y_wlasl.npy')

# Load our merged training data
X_merged = np.load(PROJECT_ROOT / 'data/processed/merged/X_train.npy')
y_merged = np.load(PROJECT_ROOT / 'data/processed/merged/y_train.npy')

print(f"\nWLASL data:")
print(f"  Shape: {X_wlasl.shape}")
print(f"  Dtype: {X_wlasl.dtype}")
print(f"  Range: [{X_wlasl.min():.4f}, {X_wlasl.max():.4f}]")
print(f"  Mean: {X_wlasl.mean():.4f}, Std: {X_wlasl.std():.4f}")
print(f"  Labels shape: {y_wlasl.shape}, unique: {len(np.unique(y_wlasl))}")

print(f"\nMerged training data:")
print(f"  Shape: {X_merged.shape}")
print(f"  Dtype: {X_merged.dtype}")
print(f"  Range: [{X_merged.min():.4f}, {X_merged.max():.4f}]")
print(f"  Mean: {X_merged.mean():.4f}, Std: {X_merged.std():.4f}")
print(f"  Labels shape: {y_merged.shape}, unique: {len(np.unique(y_merged))}")

# Check if shapes match expected format
print(f"\n  Expected input shape: (samples, 30 frames, 63 features)")
print(f"  WLASL matches: {X_wlasl.shape[1:] == (30, 63)}")
print(f"  Merged matches: {X_merged.shape[1:] == (30, 63)}")

print("\n" + "="*70)
print("DIAGNOSTIC 2: Sample data inspection")
print("="*70)

# Look at first few samples
print("\nWLASL sample [0, 0, :10] (first frame, first 10 features):")
print(f"  {X_wlasl[0, 0, :10]}")

print("\nMerged sample [0, 0, :10]:")
print(f"  {X_merged[0, 0, :10]}")

# Check for zeros/NaN
wlasl_zeros = (X_wlasl == 0).sum() / X_wlasl.size * 100
merged_zeros = (X_merged == 0).sum() / X_merged.size * 100
print(f"\nZero percentage - WLASL: {wlasl_zeros:.2f}%, Merged: {merged_zeros:.2f}%")

wlasl_nan = np.isnan(X_wlasl).sum()
merged_nan = np.isnan(X_merged).sum()
print(f"NaN count - WLASL: {wlasl_nan}, Merged: {merged_nan}")

# Check variance per feature
wlasl_var = X_wlasl.var(axis=(0,1))
merged_var = X_merged.var(axis=(0,1))
print(f"\nFeature variance - WLASL: min={wlasl_var.min():.4f}, max={wlasl_var.max():.4f}")
print(f"Feature variance - Merged: min={merged_var.min():.4f}, max={merged_var.max():.4f}")

# Check if data looks normalized
print(f"\nData normalization check:")
print(f"  WLASL appears normalized (mean≈0, std≈1): {abs(X_wlasl.mean()) < 0.5 and 0.5 < X_wlasl.std() < 2}")
print(f"  Merged appears normalized: {abs(X_merged.mean()) < 0.5 and 0.5 < X_merged.std() < 2}")

print("\n" + "="*70)
print("DIAGNOSTIC 3: Label distribution for WLASL100")
print("="*70)

# Get WLASL100 subset
with open(PROJECT_ROOT / 'data/raw/wlasl/start_kit/WLASL_v0.3.json') as f:
    wlasl_json = json.load(f)

with open(PROJECT_ROOT / 'data/processed/wlasl_label_map.json') as f:
    label_map = json.load(f)

top100_glosses = [e['gloss'] for e in wlasl_json[:100]]
idx_to_gloss = {v: k for k, v in label_map.items()}

# Filter to top 100
mask = np.array([idx_to_gloss.get(int(l), '') in top100_glosses for l in y_wlasl])
X_sub = X_wlasl[mask]
y_sub = y_wlasl[mask]

# Remap labels
gloss_to_new = {g: i for i, g in enumerate(top100_glosses)}
y_remapped = np.array([gloss_to_new[idx_to_gloss[int(l)]] for l in y_sub])

print(f"\nWLASL100 subset:")
print(f"  Samples: {len(X_sub)}")
print(f"  Classes: {len(np.unique(y_remapped))}")
print(f"  Label range: [{y_remapped.min()}, {y_remapped.max()}]")

from collections import Counter
counts = Counter(y_remapped)
print(f"  Samples per class: min={min(counts.values())}, max={max(counts.values())}, mean={np.mean(list(counts.values())):.1f}")

print("\n" + "="*70)
print("DIAGNOSTIC 4: Model forward pass check")
print("="*70)

# Create model for 100 classes
config = PhonSSMConfig(num_signs=100, temperature=0.5)
model = PhonSSM(config).to(device)

print(f"\nModel config:")
print(f"  num_signs: {config.num_signs}")
print(f"  temperature: {config.temperature}")
print(f"  d_model: {config.d_model}")

# Test with WLASL data
X_batch = torch.FloatTensor(X_sub[:32]).to(device)
y_batch = torch.LongTensor(y_remapped[:32]).to(device)

print(f"\nInput batch:")
print(f"  Shape: {X_batch.shape}")
print(f"  Range: [{X_batch.min().item():.4f}, {X_batch.max().item():.4f}]")

model.eval()
with torch.no_grad():
    outputs = model(X_batch)
    logits = outputs['logits']

print(f"\nOutput logits:")
print(f"  Shape: {logits.shape}")
print(f"  Range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
print(f"  Mean: {logits.mean().item():.4f}, Std: {logits.std().item():.4f}")

# Check softmax
probs = F.softmax(logits, dim=-1)
print(f"\nSoftmax probs:")
print(f"  Max prob: {probs.max().item():.4f}")
print(f"  Min prob: {probs.min().item():.6f}")
print(f"  Entropy (should be ~4.6 for random): {-(probs * probs.log()).sum(dim=-1).mean().item():.4f}")

# Check predictions
preds = logits.argmax(dim=-1)
acc = (preds == y_batch).float().mean().item()
print(f"  Initial accuracy: {acc*100:.1f}% (expected ~1% random)")

print("\n" + "="*70)
print("DIAGNOSTIC 5: Gradient flow check")
print("="*70)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Forward pass
outputs = model(X_batch)
logits = outputs['logits']
loss = F.cross_entropy(logits, y_batch)
print(f"\nInitial loss: {loss.item():.4f} (expected ~4.6 for random)")

# Backward pass
loss.backward()

# Check gradients
grad_stats = {}
zero_grad_params = []
large_grad_params = []

for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_stats[name] = grad_norm
        if grad_norm < 1e-8:
            zero_grad_params.append(name)
        if grad_norm > 10:
            large_grad_params.append((name, grad_norm))

print(f"\nGradient statistics:")
print(f"  Parameters with gradients: {len(grad_stats)}")
print(f"  Gradient norm range: [{min(grad_stats.values()):.6f}, {max(grad_stats.values()):.4f}]")
print(f"  Zero gradients (<1e-8): {len(zero_grad_params)}")
print(f"  Large gradients (>10): {len(large_grad_params)}")

if zero_grad_params:
    print(f"\n  Zero gradient params: {zero_grad_params[:5]}...")

if large_grad_params:
    print(f"\n  Large gradient params:")
    for name, norm in large_grad_params[:5]:
        print(f"    {name}: {norm:.4f}")

print("\n" + "="*70)
print("DIAGNOSTIC 6: Training dynamics (10 steps)")
print("="*70)

# Reset model
model = PhonSSM(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

losses = []
accs = []
logit_stats = []

for i in range(10):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_batch)
    logits = outputs['logits']
    loss = F.cross_entropy(logits, y_batch, label_smoothing=0.1)
    loss.backward()

    # Check gradient norm before clipping
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        outputs = model(X_batch)
        logits = outputs['logits']
        preds = logits.argmax(dim=-1)
        acc = (preds == y_batch).float().mean().item()

    losses.append(loss.item())
    accs.append(acc)
    logit_stats.append((logits.mean().item(), logits.std().item(), logits.min().item(), logits.max().item()))

print(f"\nLosses: {[f'{l:.3f}' for l in losses]}")
print(f"Accuracies: {[f'{a*100:.1f}%' for a in accs]}")
print(f"Loss trend: {'DECREASING' if losses[-1] < losses[0] else 'NOT DECREASING'}")
print(f"Acc trend: {'INCREASING' if accs[-1] > accs[0] else 'NOT INCREASING'}")

print(f"\nLogit statistics over training:")
for i, (mean, std, min_v, max_v) in enumerate(logit_stats):
    print(f"  Step {i}: mean={mean:.3f}, std={std:.3f}, range=[{min_v:.3f}, {max_v:.3f}]")

print("\n" + "="*70)
print("DIAGNOSTIC 7: HPC component analysis")
print("="*70)

model.eval()
with torch.no_grad():
    outputs = model(X_batch)

print("\nComponent similarities (should show variation):")
for name, sim in outputs['component_similarities'].items():
    print(f"  {name}: shape={sim.shape}, range=[{sim.min().item():.3f}, {sim.max().item():.3f}], std={sim.std().item():.3f}")

print("\nSign embedding:")
sign_emb = outputs['sign_embedding']
print(f"  Shape: {sign_emb.shape}")
print(f"  Norm: {sign_emb.norm(dim=-1).mean().item():.3f}")
print(f"  Range: [{sign_emb.min().item():.3f}, {sign_emb.max().item():.3f}]")

# Check sign prototypes
sign_protos = model.hpc.sign_prototypes
print(f"\nSign prototypes:")
print(f"  Shape: {sign_protos.shape}")
print(f"  Norm: {sign_protos.norm(dim=-1).mean().item():.3f}")

# Similarity matrix between sign prototypes (should be diverse)
proto_norm = F.normalize(sign_protos, dim=-1)
sim_matrix = torch.matmul(proto_norm, proto_norm.T)
off_diag = sim_matrix[~torch.eye(100, dtype=bool, device=device)]
print(f"  Prototype similarity (off-diagonal): mean={off_diag.mean().item():.3f}, std={off_diag.std().item():.3f}")

print("\n" + "="*70)
print("DIAGNOSTIC 8: Compare with simple baseline")
print("="*70)

# Simple MLP baseline
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=30*63, hidden_dim=256, num_classes=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

mlp = SimpleMLP().to(device)
mlp_opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3)

mlp_losses = []
mlp_accs = []

for i in range(10):
    mlp.train()
    mlp_opt.zero_grad()
    logits = mlp(X_batch)
    loss = F.cross_entropy(logits, y_batch, label_smoothing=0.1)
    loss.backward()
    mlp_opt.step()

    mlp.eval()
    with torch.no_grad():
        logits = mlp(X_batch)
        preds = logits.argmax(dim=-1)
        acc = (preds == y_batch).float().mean().item()

    mlp_losses.append(loss.item())
    mlp_accs.append(acc)

print(f"\nSimple MLP baseline (same data, 10 steps):")
print(f"  Losses: {[f'{l:.3f}' for l in mlp_losses]}")
print(f"  Accuracies: {[f'{a*100:.1f}%' for a in mlp_accs]}")
print(f"  Loss trend: {'DECREASING' if mlp_losses[-1] < mlp_losses[0] else 'NOT DECREASING'}")

print("\n" + "="*70)
print("DIAGNOSTIC 9: Extended training test (100 steps)")
print("="*70)

# Train PhonSSM for 100 steps on this batch
model = PhonSSM(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for i in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = F.cross_entropy(outputs['logits'], y_batch, label_smoothing=0.1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if i % 20 == 0 or i == 99:
        model.eval()
        with torch.no_grad():
            outputs = model(X_batch)
            preds = outputs['logits'].argmax(dim=-1)
            acc = (preds == y_batch).float().mean().item()
        print(f"  Step {i:3d}: loss={loss.item():.3f}, acc={acc*100:.1f}%")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nKey findings will appear above. Look for:")
print("1. Data format mismatches between WLASL and training data")
print("2. Normalization differences")
print("3. Gradient flow issues")
print("4. Whether loss decreases on a single batch (overfitting test)")
print("5. Whether simple MLP can learn (data quality check)")
