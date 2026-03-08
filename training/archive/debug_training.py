"""Debug PhonSSM training on WLASL100."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.phonssm import PhonSSM, PhonSSMConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load small subset of WLASL data
X = np.load(PROJECT_ROOT / 'data/processed/X_wlasl.npy')[:1000]
y = np.load(PROJECT_ROOT / 'data/processed/y_wlasl.npy')[:1000]

# Remap to 0-99 for simplicity
unique_labels = np.unique(y)[:100]
label_map = {l: i for i, l in enumerate(unique_labels)}
mask = np.isin(y, unique_labels)
X = X[mask][:500]
y = np.array([label_map[l] for l in y[mask][:500]])

print(f"Data: X={X.shape}, y={y.shape}, classes={len(np.unique(y))}")

# Create model
config = PhonSSMConfig(num_signs=100)
model = PhonSSM(config).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Single batch test
X_batch = torch.FloatTensor(X[:32]).to(device)
y_batch = torch.LongTensor(y[:32]).to(device)

print("\n=== Forward Pass Test ===")
model.train()
outputs = model(X_batch)
logits = outputs['logits']
print(f"Logits shape: {logits.shape}")
print(f"Logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
print(f"Logits min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")

# Check softmax
probs = F.softmax(logits, dim=-1)
print(f"Probs sum: {probs.sum(dim=-1).mean().item():.4f}")
print(f"Max prob: {probs.max().item():.4f}")

# Compute loss
loss = F.cross_entropy(logits, y_batch)
print(f"\nLoss: {loss.item():.4f} (expected for random: {np.log(100):.4f})")

print("\n=== Gradient Test ===")
loss.backward()

# Check gradients
grad_norms = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norms[name] = param.grad.norm().item()

print(f"Params with gradients: {len(grad_norms)}")
print(f"Params without gradients: {sum(1 for p in model.parameters() if p.grad is None)}")

# Show gradient statistics
grad_values = list(grad_norms.values())
print(f"Gradient norm - min: {min(grad_values):.6f}, max: {max(grad_values):.6f}, mean: {np.mean(grad_values):.6f}")

# Check for vanishing/exploding gradients
zero_grads = sum(1 for g in grad_values if g < 1e-8)
large_grads = sum(1 for g in grad_values if g > 100)
print(f"Zero gradients (<1e-8): {zero_grads}")
print(f"Large gradients (>100): {large_grads}")

# Show some layer gradients
print("\nSample layer gradients:")
for name, norm in list(grad_norms.items())[:10]:
    print(f"  {name}: {norm:.6f}")

print("\n=== Training Step Test ===")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

losses = []
for i in range(10):
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = F.cross_entropy(outputs['logits'], y_batch, label_smoothing=0.1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())

print(f"Losses over 10 steps: {[f'{l:.3f}' for l in losses]}")
print(f"Loss decreased: {losses[-1] < losses[0]}")

# Check predictions after training
model.eval()
with torch.no_grad():
    outputs = model(X_batch)
    preds = outputs['logits'].argmax(dim=-1)
    acc = (preds == y_batch).float().mean().item()
print(f"Accuracy after 10 steps: {acc*100:.1f}%")
