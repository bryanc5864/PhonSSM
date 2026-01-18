"""Check data statistics."""
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Check WLASL data vs merged data
X_wlasl = np.load(PROJECT_ROOT / 'data/processed/X_wlasl.npy', allow_pickle=True)
X_merged = np.load(PROJECT_ROOT / 'data/processed/merged/X_train.npy', allow_pickle=True)

print('=== Data Comparison ===')
print(f'WLASL shape: {X_wlasl.shape}')
print(f'Merged shape: {X_merged.shape}')

print(f'\nWLASL stats:')
print(f'  Mean: {X_wlasl.mean():.4f}')
print(f'  Std:  {X_wlasl.std():.4f}')
print(f'  Min:  {X_wlasl.min():.4f}')
print(f'  Max:  {X_wlasl.max():.4f}')

print(f'\nMerged stats:')
print(f'  Mean: {X_merged.mean():.4f}')
print(f'  Std:  {X_merged.std():.4f}')
print(f'  Min:  {X_merged.min():.4f}')
print(f'  Max:  {X_merged.max():.4f}')

# Check sample values
print(f'\nWLASL sample [0,0,:10]: {X_wlasl[0, 0, :10]}')
print(f'Merged sample [0,0,:10]: {X_merged[0, 0, :10]}')

# Check for NaN/inf
print(f'\nWLASL NaN: {np.isnan(X_wlasl).sum()}, Inf: {np.isinf(X_wlasl).sum()}')
print(f'Merged NaN: {np.isnan(X_merged).sum()}, Inf: {np.isinf(X_merged).sum()}')

# Check zero rows
wlasl_zero_rows = (X_wlasl == 0).all(axis=(1,2)).sum()
merged_zero_rows = (X_merged == 0).all(axis=(1,2)).sum()
print(f'\nWLASL all-zero samples: {wlasl_zero_rows}')
print(f'Merged all-zero samples: {merged_zero_rows}')
