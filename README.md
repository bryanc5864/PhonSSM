# PhonSSM: Phonological State Space Model for Sign Language Recognition

A novel architecture for skeleton-based sign language recognition that achieves state-of-the-art performance using only pose landmarks (no RGB video). PhonSSM incorporates linguistic priors from sign language phonology to achieve superior accuracy with dramatically fewer parameters.

## Key Results

| Dataset | Classes | Top-1 Accuracy | Previous SOTA | Improvement |
|---------|---------|----------------|---------------|-------------|
| WLASL100 | 100 | **88.37%** | 63.18% (DSTA-SLR) | +25.19 pts |
| WLASL300 | 300 | **74.41%** | 58.42% (DSTA-SLR) | +15.99 pts |
| WLASL1000 | 1,000 | **62.90%** | 47.14% (DSTA-SLR) | +15.76 pts |
| WLASL2000 | 2,000 | **72.08%** | 53.70% (DSTA-SLR) | +18.38 pts |
| Merged-5565 | 5,565 | **53.34%** | - | New benchmark |

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
# Clone repository
git clone https://github.com/anonymoussubmitter-167/phon-ssm.git
cd phon-ssm

# Install dependencies
pip install torch>=2.0.0 numpy scikit-learn matplotlib seaborn

# For training diagnostics models (optional)
pip install tensorflow>=2.15.0
```

## Quick Start

### Training on WLASL

```bash
# Train on WLASL100
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100

# Train on WLASL2000
python training/benchmark_external.py --dataset wlasl --subset 2000 --epochs 100

# Resume from checkpoint
python training/benchmark_external.py --dataset wlasl --subset 2000 \
    --resume benchmarks/external/wlasl2000/TIMESTAMP
```

### Running Analysis

```bash
# Confusion matrix analysis
python analysis/confusion_matrix.py --subset 100

# t-SNE visualization of phonological subspaces
python analysis/tsne_phonology.py --subset 100

# Attention heatmap visualization
python analysis/attention_heatmap.py --subset 100
```

## Project Structure

```
phon-ssm/
├── models/
│   └── phonssm/
│       ├── __init__.py          # PhonSSM model definition
│       ├── agan.py              # Anatomical Graph Attention
│       ├── pdm.py               # Phonological Disentanglement
│       ├── bissm.py             # Bidirectional State Space
│       └── hpc.py               # Hierarchical Prototypes
├── training/
│   ├── benchmark_external.py    # Main training script
│   ├── train_diagnosis.py       # Error diagnosis model
│   └── train_movement.py        # Movement analyzer model
├── analysis/
│   ├── confusion_matrix.py      # Confusion matrix analysis
│   ├── tsne_phonology.py        # t-SNE visualization
│   └── attention_heatmap.py     # Attention visualization
├── benchmarks/
│   └── external/
│       ├── wlasl100/            # WLASL100 results
│       ├── wlasl300/            # WLASL300 results
│       ├── wlasl1000/           # WLASL1000 results
│       └── wlasl2000/           # WLASL2000 results
├── web/
│   ├── server.py                # FastAPI web server
│   └── static/index.html        # Web interface
├── RESULTS.md                   # Detailed benchmark results
└── README.md                    # This file
```

## Input Format

PhonSSM accepts skeleton landmarks extracted via MediaPipe:

- **Pose landmarks**: 33 body keypoints (x, y, z)
- **Left hand**: 21 hand keypoints (x, y, z)
- **Right hand**: 21 hand keypoints (x, y, z)
- **Total**: 75 landmarks × 3 coordinates = 225 features per frame
- **Sequence length**: 30 frames (uniformly sampled)

Input shape: `(batch_size, 30, 225)`

### Preprocessing
1. Uniform temporal sampling to 30 frames
2. Spatial normalization: center at wrist, scale by max landmark distance
3. Missing landmarks filled via linear interpolation

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

### Large-Scale Benchmark: Merged-5565

Combined dataset from WLASL, ASL Citizen, and SignBank:
- **5,565 unique signs**
- **260,000+ samples**
- **Top-1 Accuracy: 53.34%**

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

PhonSSM excels on classes with limited training data:

| Training Samples | Bi-LSTM Baseline | PhonSSM | Improvement |
|------------------|------------------|---------|-------------|
| 1-5 samples | 12.3% | 39.8% | +225% |
| 6-10 samples | 34.7% | 58.2% | +68% |
| 11-20 samples | 52.1% | 71.4% | +37% |
| 20+ samples | 68.9% | 82.6% | +20% |

### Key Findings

1. **Skeleton-only superiority**: PhonSSM outperforms RGB video methods while using 8-25x fewer parameters
2. **Linguistically meaningful errors**: Most confusion occurs between phonologically similar signs (e.g., minimal pairs differing in one component)
3. **Phonological disentanglement**: t-SNE visualization shows clear clustering by handshape/location in respective subspaces
4. **Efficient inference**: Real-time capable at 260 samples/sec on CPU

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
@article{phonssm2026,
  title={PhonSSM: Phonological State Space Model for Sign Language Recognition},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}
```

## License

This project is released under the MIT License.
