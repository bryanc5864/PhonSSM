"""Quick test to verify PhonSSM model works correctly."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\n" + "=" * 50)
print("TESTING PHONSSM MODEL")
print("=" * 50)

# Test imports
print("\n1. Testing imports...")
try:
    from models.phonssm import (
        PhonSSMConfig, PhonSSM, create_phonssm,
        AnatomicalGraphAttention, PhonologicalDisentanglement,
        BiSSM, HierarchicalPrototypicalClassifier
    )
    print("   [OK] All imports successful")
except ImportError as e:
    print(f"   [FAIL] Import error: {e}")
    sys.exit(1)

# Test config
print("\n2. Testing config...")
config = PhonSSMConfig(num_signs=100)  # Smaller for testing
print(f"   [OK] Config created: {config.num_signs} signs, {config.d_model} dims")

# Test individual components
print("\n3. Testing AGAN...")
agan = AnatomicalGraphAttention(
    in_dim=3, hidden_dim=64, out_dim=128, num_heads=4, num_nodes=21
)
x_agan = torch.randn(2, 30, 21, 3)  # (B, T, N, C)
out_agan = agan(x_agan)
print(f"   Input: {x_agan.shape} -> Output: {out_agan.shape}")
print(f"   [OK] AGAN works")

print("\n4. Testing PDM...")
pdm = PhonologicalDisentanglement(in_dim=128, component_dim=32)
x_pdm = torch.randn(2, 30, 128)
out_pdm = pdm(x_pdm)
print(f"   Input: {x_pdm.shape}")
print(f"   Output fused: {out_pdm['fused'].shape}")
print(f"   Components: handshape={out_pdm['handshape'].shape}, movement={out_pdm['movement'].shape}")
ortho_loss = pdm.orthogonality_loss(out_pdm)
print(f"   Orthogonality loss: {ortho_loss.item():.4f}")
print(f"   [OK] PDM works")

print("\n5. Testing BiSSM...")
bissm = BiSSM(d_model=128, d_state=16, num_layers=2)
x_bissm = torch.randn(2, 30, 128)
out_bissm = bissm(x_bissm)
print(f"   Input: {x_bissm.shape} -> Output: {out_bissm.shape}")
print(f"   [OK] BiSSM works")

print("\n6. Testing HPC...")
hpc = HierarchicalPrototypicalClassifier(
    d_model=128, component_dim=32, num_signs=100
)
components = {
    'handshape': torch.randn(2, 30, 32),
    'location': torch.randn(2, 30, 32),
    'movement': torch.randn(2, 30, 32),
    'orientation': torch.randn(2, 30, 32)
}
temporal = torch.randn(2, 30, 128)
out_hpc = hpc(temporal, components)
print(f"   Logits: {out_hpc['logits'].shape}")
print(f"   Sign embedding: {out_hpc['sign_embedding'].shape}")
print(f"   [OK] HPC works")

# Test full model
print("\n7. Testing full PhonSSM model...")
model = create_phonssm(num_signs=100)
x_full = torch.randn(2, 30, 63)  # Flattened input (B, T, 21*3)
outputs = model(x_full)
print(f"   Input: {x_full.shape}")
print(f"   Logits: {outputs['logits'].shape}")
print(f"   Sign embedding: {outputs['sign_embedding'].shape}")
print(f"   [OK] Full model forward pass works")

# Test loss computation
print("\n8. Testing loss computation...")
targets = torch.randint(0, 100, (2,))
losses = model.compute_loss(outputs, targets)
print(f"   Classification loss: {losses['classification'].item():.4f}")
print(f"   Orthogonality loss: {losses['orthogonality'].item():.4f}")
print(f"   Total loss: {losses['total'].item():.4f}")
print(f"   [OK] Loss computation works")

# Test backward pass
print("\n9. Testing backward pass...")
losses['total'].backward()
print(f"   [OK] Backward pass works")

# Test predictions
print("\n10. Testing predictions...")
model.eval()
with torch.no_grad():
    outputs = model(x_full)
    preds = model.get_predictions(outputs, top_k=5)
print(f"   Top-5 predictions: {preds['top_k_indices'].shape}")
print(f"   Predicted classes: {preds['predicted_class'].tolist()}")
print(f"   [OK] Predictions work")

# Parameter count
print("\n11. Parameter counts:")
counts = model.count_parameters()
for name, count in counts.items():
    print(f"   {name}: {count:,}")

# Test with GPU if available
if torch.cuda.is_available():
    print("\n12. Testing on GPU...")
    model = model.cuda()
    x_gpu = x_full.cuda()
    targets_gpu = targets.cuda()

    outputs = model(x_gpu)
    losses = model.compute_loss(outputs, targets_gpu)
    losses['total'].backward()
    print(f"   [OK] GPU forward/backward works")

    # Memory usage
    print(f"   GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
print("\nPhonSSM model is ready for training.")
print("Run: python training/train_phonssm.py")
