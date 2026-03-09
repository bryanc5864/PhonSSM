# PhonSSM: Phonological State Space Model for Sign Language Recognition

[![ICLR 2026 Workshop](https://img.shields.io/badge/ICLR%202026-Workshop-blue)](https://iclr.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A novel architecture for skeleton-based sign language recognition that achieves state-of-the-art performance using only pose landmarks (no RGB video). PhonSSM incorporates linguistic priors from sign language phonology to achieve superior accuracy with dramatically fewer parameters.

## Key Results

| Dataset | Classes | Top-1 Accuracy | Previous SOTA | Improvement |
|---------|---------|----------------|---------------|-------------|
| WLASL100 | 100 | **88.37%** | 63.18% (DSTA-SLR) | +25.19 pts |
| WLASL300 | 300 | **74.41%** | 58.42% (DSTA-SLR) | +15.99 pts |
| WLASL1000 | 1,000 | **62.90%** | 47.14% (DSTA-SLR) | +15.76 pts |
| WLASL2000 | 2,000 | **72.08%** | 53.70% (DSTA-SLR) | +18.38 pts |

**Key advantages:**
- **3.2M parameters** vs 25M+ for RGB-based methods
- **260 samples/sec** inference on CPU
- **+225% improvement** on few-shot classes (1-5 training samples)
- Skeleton-only input enables real-time mobile deployment

## Architecture

PhonSSM consists of four key components designed around sign language phonology:

```
Input (Pose + Hands Landmarks)
         │
         ▼
┌─────────────────────────────────────┐
│  AGAN: Anatomical Graph Attention   │
│  - Hand topology-aware adjacency    │
│  - Multi-head graph attention       │
│  - Preserves skeletal structure     │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  PDM: Phonological Disentanglement  │
│  - 4 orthogonal subspaces:          │
│    • Handshape (finger config)      │
│    • Location (signing space)       │
│    • Movement (trajectory)          │
│    • Orientation (palm facing)      │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  BiSSM: Bidirectional State Space   │
│  - O(n) temporal modeling           │
│  - Forward + backward context       │
│  - Selective state spaces (Mamba)   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  HPC: Hierarchical Prototypes       │
│  - Learnable class prototypes       │
│  - Temperature-scaled similarity    │
│  - Few-shot friendly classification │
└─────────────────────────────────────┘
         │
         ▼
      Output (Sign Class)
```

### Component Details

1. **AGAN (Anatomical Graph Attention Network)**
   - Processes skeleton as a graph with anatomically-motivated adjacency
   - Hand landmarks connected following finger bone structure
   - Multi-head attention learns joint relationships beyond physical connections
   - Output: Spatially-aware joint embeddings

2. **PDM (Phonological Disentanglement Module)**
   - Based on Stokoe's sign language phonology (1960)
   - Projects features into 4 orthogonal subspaces via learned linear projections
   - Orthogonality loss: `L_orth = Σ||W_i^T W_j||_F` for i≠j
   - Enables interpretable feature analysis and component-specific feedback

3. **BiSSM (Bidirectional State Space Model)**
   - Efficient O(n) sequence modeling vs O(n²) for transformers
   - Based on selective state space models (Mamba architecture)
   - Bidirectional processing captures both anticipatory and perseveratory coarticulation
   - Discretized state equation: `h_t = Āh_{t-1} + B̄x_t`

4. **HPC (Hierarchical Prototypical Classifier)**
   - Learnable prototype vectors for each sign class
   - Classification via temperature-scaled cosine similarity
   - Particularly effective for long-tail distributions common in sign datasets
   - Few-shot friendly: +225% accuracy improvement on classes with ≤5 samples

## Installation

```bash
git clone https://github.com/bryanc5864/PhonSSM.git
cd PhonSSM
pip install -e .
```

**Requirements:** Python 3.8+, PyTorch 2.0+, CUDA recommended

## Quick Start

### Inference

```python
from models.phonssm import PhonSSM, PhonSSMConfig
import torch

# Load model
config = PhonSSMConfig.for_wlasl(100)
model = PhonSSM(config)
checkpoint = torch.load('path/to/checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Input: (batch, frames=30, landmarks, coords=3)
x = torch.randn(1, 30, 75, 3)  # 75 landmarks for pose+hands
outputs = model(x)

predictions = outputs['logits'].argmax(dim=-1)
phonological = outputs['phonological_components']  # Interpretable features
```

### Training

```bash
# Train on WLASL
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100

# Train on custom data
python training/train_phonssm.py --epochs 100 --batch-size 32 --device cuda
```

## Project Structure

```
phonssm/
├── models/phonssm/       # Core architecture (AGAN, PDM, BiSSM, HPC)
├── training/             # Training and benchmark scripts
└── benchmarks/           # Benchmark results
```

## Input Format

PhonSSM accepts skeleton landmarks extracted via MediaPipe:

- **Pose landmarks**: 33 body keypoints (x, y, z)
- **Left hand**: 21 hand keypoints (x, y, z)
- **Right hand**: 21 hand keypoints (x, y, z)
- **Total**: 75 landmarks × 3 coordinates = 225 features per frame
- **Sequence length**: 30 frames (uniformly sampled)

Input shape: `(batch_size, 30, 75, 3)` or `(batch_size, 30, 225)`

## Model Specifications

| Component | Parameters | Description |
|-----------|------------|-------------|
| AGAN | ~800K | Graph attention on 75 landmarks |
| PDM | ~130K | 4 orthogonal subspaces (32-dim each) |
| BiSSM | ~1.5M | Bidirectional selective state space |
| HPC | ~800K | Prototype classifier (for 2000 classes) |
| **Total** | **~3.2M** | Full PhonSSM model |

### Inference Performance
- **Throughput**: 260 samples/sec on CPU
- **Latency**: 3.85ms per sample
- **Memory**: <500MB GPU memory

## Detailed Results

### WLASL Benchmarks

| Dataset | Classes | Test Samples | Top-1 | Top-5 | Top-10 |
|---------|---------|--------------|-------|-------|--------|
| WLASL100 | 100 | 774 | 88.37% | 94.06% | 96.77% |
| WLASL300 | 300 | 2,005 | 74.41% | 88.93% | 92.22% |
| WLASL1000 | 1,000 | 5,628 | 62.90% | 82.60% | 86.35% |
| WLASL2000 | 2,000 | 8,634 | 72.08% | 86.26% | 88.56% |

### Comparison with Prior Work

| Method | Input Type | Params | WLASL100 | WLASL2000 |
|--------|------------|--------|----------|-----------|
| I3D | RGB Video | 25M | 65.89% | 32.48% |
| Pose-TGCN | Skeleton | 3.1M | 55.43% | - |
| SignBERT | RGB Video | 85M | 79.36% | - |
| DSTA-SLR | Skeleton | 4.2M | 63.18% | 53.70% |
| NLA-SLR | RGB+Skeleton | 42M | 67.54% | - |
| **PhonSSM (Ours)** | **Skeleton** | **3.2M** | **88.37%** | **72.08%** |

### Few-Shot Performance

| Training Samples | Bi-LSTM Baseline | PhonSSM | Improvement |
|------------------|------------------|---------|-------------|
| 1-5 samples | 12.3% | 39.8% | +225% |
| 6-10 samples | 34.7% | 58.2% | +68% |
| 11-20 samples | 52.1% | 71.4% | +37% |
| 20+ samples | 68.9% | 82.6% | +20% |

## Ablation Studies

| Configuration | WLASL100 | WLASL2000 |
|---------------|----------|-----------|
| Full PhonSSM | **88.37%** | **72.08%** |
| w/o PDM (no disentanglement) | 82.14% | 65.23% |
| w/o BiSSM (LSTM instead) | 79.56% | 61.87% |
| w/o HPC (linear classifier) | 85.21% | 68.45% |
| w/o AGAN (MLP instead) | 76.89% | 58.92% |

## Citation

```bibtex
@inproceedings{phonssm2026,
  title={State Space Models are Effective Sign Language Learners: Exploiting Phonological Compositionality for Vocabulary-Scale Recognition},
  author={Zhang, Jasper and Cheng, Bryan and Jin, Austin},
  booktitle={ICLR 2026 Workshop},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [WLASL Dataset](https://dxli94.github.io/WLASL/) for benchmark data
- [MediaPipe](https://mediapipe.dev/) for pose estimation
- [Mamba](https://github.com/state-spaces/mamba) for state space model inspiration
