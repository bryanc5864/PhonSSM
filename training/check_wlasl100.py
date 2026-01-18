"""Check WLASL100 class distribution."""
import numpy as np
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent

# Load WLASL data
X = np.load(PROJECT_ROOT / 'data/processed/X_wlasl.npy')
y = np.load(PROJECT_ROOT / 'data/processed/y_wlasl.npy')
with open(PROJECT_ROOT / 'data/processed/wlasl_label_map.json') as f:
    label_map = json.load(f)

# Load WLASL JSON to get top 100 glosses
with open(PROJECT_ROOT / 'data/raw/wlasl/start_kit/WLASL_v0.3.json') as f:
    wlasl = json.load(f)

top100 = [e['gloss'] for e in wlasl[:100]]
idx_to_gloss = {v:k for k,v in label_map.items()}

# Filter to top 100
mask = [idx_to_gloss.get(int(l),'') in top100 for l in y]
y_sub = y[mask]

# Remap
gloss_to_new = {g:i for i,g in enumerate(top100)}
y_new = [gloss_to_new[idx_to_gloss[int(l)]] for l in y_sub]

counts = Counter(y_new)
print(f'WLASL100 samples: {len(y_new)}')
print(f'Classes: {len(counts)}')
print(f'Samples per class: min={min(counts.values())}, max={max(counts.values())}, mean={np.mean(list(counts.values())):.1f}')
print(f'Classes with <10 samples: {sum(1 for c in counts.values() if c < 10)}')
print(f'Classes with <20 samples: {sum(1 for c in counts.values() if c < 20)}')
print(f'Classes with <30 samples: {sum(1 for c in counts.values() if c < 30)}')
