# PhonSSM Benchmark Plan

## Overview

This document outlines the complete benchmarking strategy for PhonSSM, organized in priority order.

---

## Part 1: Datasets for Direct SOTA Comparison

### Priority Order

| Priority | Dataset | Classes | Why It Matters |
|----------|---------|---------|----------------|
| **CRITICAL** | WLASL | 100/300/2000 | Primary SLR benchmark, most cited |
| **HIGH** | AUTSL | 226 | ChaLearn challenge dataset, skeleton provided |
| **HIGH** | MSASL | 100/200/500/1000 | Microsoft benchmark, large-scale |
| **MEDIUM** | ASL Citizen | 2,731 | NeurIPS 2023, similar vocab size to ours |
| **MEDIUM** | LSA64 | 64 | Small but standard, good for quick validation |

---

## 1. WLASL (Word-Level American Sign Language) - CRITICAL

**The most important benchmark for publication.**

### Dataset Info
| Subset | Classes | Train | Val | Test | Total |
|--------|---------|-------|-----|------|-------|
| WLASL100 | 100 | 1,359 | 340 | 339 | 2,038 |
| WLASL300 | 300 | 3,034 | 749 | 765 | 4,548 |
| WLASL1000 | 1,000 | 8,562 | 2,119 | 2,134 | 12,815 |
| WLASL2000 | 2,000 | 14,289 | 3,916 | 3,775 | 21,980 |

### How to Get It

**Option A: Official Repository (Recommended)**
```bash
git clone https://github.com/dxli94/WLASL.git
cd WLASL
pip install yt-dlp
python start_kit/video_downloader.py
```

**Option B: Request Pre-processed Videos**
- Form: https://docs.google.com/forms/d/e/1FAIpQLSc3yHyAranhpkC9ur_Z-Gu5gS5M0WnKtHV07Vo6eL6nZHzruw/viewform

**Option C: Kaggle**
- https://www.kaggle.com/datasets/sttaseen/wlasl2000-resized
- https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed

**Option D: HuggingFace**
```python
import fiftyone.utils.huggingface as fouh
dataset = fouh.load_from_hub("Voxel51/WLASL")
```

### SOTA to Beat (Skeleton-Only)
| Method | WLASL100 | WLASL300 | WLASL2000 |
|--------|----------|----------|-----------|
| TMS-Net (2024) | ~85% | ~70% | **56.4%** |
| Siformer (2024) | **86.50%** | - | - |
| Pose-TGCN (2020) | ~50% | ~40% | ~30% |

### Status
- [x] Dataset downloaded (`data/raw/wlasl/`)
- [x] Preprocessed (`data/processed/X_wlasl.npy`, `y_wlasl.npy`)
- [x] Label map created (`data/processed/wlasl_label_map.json`)
- [ ] WLASL100 benchmark
- [ ] WLASL300 benchmark
- [ ] WLASL2000 benchmark

---

## 2. AUTSL (Turkish Sign Language) - HIGH PRIORITY

**Includes skeleton data natively (Kinect). Good for skeleton-based evaluation.**

### Dataset Info
| Split | Samples | Notes |
|-------|---------|-------|
| Train | 28,142 | Signer-independent |
| Val | 4,418 | |
| Test | 3,742 | Labels released after challenge |
| **Total** | **36,302** | **226 classes, 43 signers** |

### How to Get It
- Website: https://cvml.ankara.edu.tr/datasets/
- Pre-processed: https://github.com/jackyjsy/CVPR21Chal-SLR

### SOTA to Beat (Skeleton-Only)
| Method | Top-1 Accuracy |
|--------|----------------|
| TMS-Net (2024) | **96.62%** |
| SAM-SLR (2021) | 95.95% |
| Baseline BiLSTM | 62.02% |

### Status
- [ ] Dataset downloaded
- [ ] Preprocessed
- [ ] Benchmark completed

---

## 3. MSASL (Microsoft ASL) - HIGH PRIORITY

**Large-scale, challenging (signer-independent), from Microsoft.**

### Dataset Info
| Subset | Classes | Train | Val | Test |
|--------|---------|-------|-----|------|
| MSASL100 | 100 | ~1,600 | ~520 | ~420 |
| MSASL200 | 200 | ~3,200 | ~1,040 | ~840 |
| MSASL500 | 500 | ~8,000 | ~2,600 | ~2,100 |
| MSASL1000 | 1,000 | 16,054 | 5,287 | 4,172 |

### How to Get It
- Download: https://www.microsoft.com/en-us/download/details.aspx?id=100121
- Video downloader: https://github.com/iamgarcia/msasl-video-downloader

### SOTA to Beat (Skeleton-Only)
| Method | MSASL1000 Top-1 |
|--------|-----------------|
| TMS-Net (2024) | **65.13%** |

### Status
- [ ] Dataset downloaded
- [ ] Preprocessed
- [ ] Benchmark completed

---

## 4. ASL Citizen - MEDIUM PRIORITY

**NeurIPS 2023 benchmark. Similar vocabulary size to our dataset!**

### Dataset Info
| Metric | Value |
|--------|-------|
| Classes | 2,731 signs |
| Videos | 83,399 |
| Signers | 52 (all D/deaf or HoH) |

### How to Get It
- Download: https://www.microsoft.com/en-us/download/details.aspx?id=105253
- Kaggle: https://www.kaggle.com/datasets/abd0kamel/asl-citizen

### SOTA Results
| Method | Top-1 | Top-10 |
|--------|-------|--------|
| I3D + Classification | 63% | 91% |
| Ensemble Swin (2025) | **86.53%** | - |

### Status
- [x] Dataset downloaded (`data/raw/ASL_Citizen/`)
- [x] Preprocessed (`data/processed/X_asl_citizen.npy`, `y_asl_citizen.npy`)
- [ ] Benchmark completed

---

## 5. LSA64 (Argentinian SL) - MEDIUM PRIORITY

**Small but standard. Good for sanity check.**

### Dataset Info
| Metric | Value |
|--------|-------|
| Classes | 64 signs |
| Videos | 3,200 |
| Signers | 10 |

### How to Get It
- Website: https://facundoq.github.io/datasets/lsa64/

### SOTA to Beat
| Method | Top-1 Accuracy |
|--------|----------------|
| DSLNet (2025) | **99.79%** |

### Status
- [ ] Dataset downloaded
- [ ] Preprocessed
- [ ] Benchmark completed

---

## Part 2: Ablation Studies (CRITICAL)

Quantify each module's contribution:

| Configuration | Top-1 | Delta | Notes |
|---------------|-------|-------|-------|
| Full PhonSSM | 53.24% | - | Baseline |
| w/o AGAN (replace w/ MLP) | ? | ? | Test graph attention value |
| w/o PDM (no phonology) | ? | ? | Test phonological decomposition |
| w/o BiSSM (use LSTM) | ? | ? | Test state space vs LSTM |
| w/o HPC (flat softmax) | ? | ? | Test hierarchical prototypes |
| w/o Contrastive Loss | ? | ? | Test contrastive learning |
| w/o Orthogonality Loss | ? | ? | Test regularization |

### Implementation
```bash
# Each ablation requires modifying the model config
python training/train_phonssm.py --ablation no_agan
python training/train_phonssm.py --ablation no_pdm
python training/train_phonssm.py --ablation no_bissm
python training/train_phonssm.py --ablation no_hpc
```

---

## Part 3: Additional Experiments

### Computational Efficiency
| Metric | Bi-LSTM | PhonSSM | Transformer |
|--------|---------|---------|-------------|
| Inference Time (ms) | 0.41 | 3.85 | ~50 |
| FLOPs (M) | ? | ? | ? |
| Memory (MB) | ? | ? | ? |
| Parameters | 748K | 3.2M | ? |

### Few-Shot Performance
| Samples/Class | Bi-LSTM | PhonSSM |
|---------------|---------|---------|
| 1-5 | 4.08% | 13.27% |
| 6-10 | 3.70% | 9.26% |
| 11-20 | 5.29% | 12.71% |
| 21-50 | 10.13% | 24.03% |
| 51-100 | 1.52% | 26.03% |
| 101+ | 52.66% | 92.82% |

### Cross-Dataset Generalization
Train on SignSense, test zero-shot on:
- [ ] WLASL (overlapping vocabulary)
- [ ] MSASL (different signers)
- [ ] ASL Citizen

### Temporal Sequence Length Sensitivity
| Frame Count | Accuracy | Notes |
|-------------|----------|-------|
| 15 | ? | Short signs |
| 30 | 53.24% | Current default |
| 45 | ? | Long signs |
| 60 | ? | Compound signs |

### Confusion Analysis
- Which handshapes are most confused?
- Are minimal pairs (differ by 1 parameter) hardest?
- Correlation between PDM embeddings and linguistic categories

### Embedding Visualization
- t-SNE/UMAP of PDM component embeddings
- Show handshape clusters align with linguistic categories
- Demonstrate minimal pair proximity

---

## Part 4: Experiment Timeline

### Week 1-2: SOTA Comparison
| Experiment | Dataset | Priority | Status |
|------------|---------|----------|--------|
| WLASL100 | WLASL | CRITICAL | [ ] |
| WLASL300 | WLASL | CRITICAL | [ ] |
| WLASL2000 | WLASL | CRITICAL | [ ] |
| AUTSL | AUTSL | HIGH | [ ] |
| MSASL1000 | MSASL | HIGH | [ ] |

### Week 3-4: Ablations & Analysis
| Experiment | Description | Status |
|------------|-------------|--------|
| w/o AGAN | Replace graph attention with MLP | [ ] |
| w/o PDM | Remove phonological disentanglement | [ ] |
| w/o BiSSM | Replace Mamba with LSTM | [ ] |
| w/o HPC | Use flat softmax | [ ] |
| Confusion Analysis | Phonological parameter confusion | [ ] |

### Week 5-6: Additional Studies
| Experiment | Description | Status |
|------------|-------------|--------|
| Cross-dataset | Zero-shot transfer | [ ] |
| Efficiency | FLOPs, memory profiling | [ ] |
| Visualization | t-SNE embeddings | [ ] |

---

## Part 5: Expected Results Table (Paper)

```
Table X: Comparison with State-of-the-Art on WLASL (Skeleton-Only)

| Method            | Modality | WLASL100 | WLASL300 | WLASL2000 |
|-------------------|----------|----------|----------|-----------|
| Pose-TGCN (2020)  | Skeleton | ~50%     | ~40%     | ~30%      |
| ST-GCN (2018)     | Skeleton | 51.67%   | -        | -         |
| TMS-Net (2024)    | Skeleton | ~85%     | ~70%     | 56.4%     |
| Siformer (2024)   | Skeleton | 86.50%   | -        | -         |
|-------------------|----------|----------|----------|-----------|
| PhonSSM (Ours)    | Skeleton | ??%      | ??%      | ??%       |
```

---

## Benchmark Commands

```bash
# WLASL Benchmarks
conda run -n tf-gpu python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100
conda run -n tf-gpu python training/benchmark_external.py --dataset wlasl --subset 300 --epochs 100
conda run -n tf-gpu python training/benchmark_external.py --dataset wlasl --subset 2000 --epochs 100

# ASL Citizen Benchmark
conda run -n tf-gpu python training/benchmark_external.py --dataset asl_citizen --epochs 100
```

---

## Citations

```bibtex
@inproceedings{li2020word,
  title={Word-level Deep Sign Language Recognition from Video},
  author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
  booktitle={WACV},
  year={2020}
}

@article{sincan2020autsl,
  title={AUTSL: A Large Scale Multi-modal Turkish Sign Language Dataset},
  author={Sincan, Ozge Mercanoglu and Keles, Hacer Yalim},
  journal={IEEE Access},
  year={2020}
}

@inproceedings{joze2019msasl,
  title={MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language},
  author={Vaezi Joze, Hamid Reza and Koller, Oscar},
  booktitle={BMVC},
  year={2019}
}

@inproceedings{desai2023asl,
  title={ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition},
  author={Desai, Aashaka and others},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2023}
}

@article{ronchetti2016lsa64,
  title={LSA64: A Dataset of Argentinian Sign Language},
  author={Ronchetti, Franco and others},
  journal={CACIC},
  year={2016}
}
```
