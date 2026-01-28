# PhonSSM Experiments TODO

Remaining experiments for paper submission. Run on GPU server for faster training.

---

## Tier 1: CRITICAL (Must Have)

### 1.1 Comprehensive Ablation Study
**Time: 2-3 days | Priority: HIGHEST**

Run ablations on WLASL100 and WLASL2000:

```bash
# Full model (baseline)
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100

# Ablation: Remove AGAN (replace with MLP)
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100 --ablate agan

# Ablation: Remove PDM (no disentanglement)
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100 --ablate pdm

# Ablation: Remove BiSSM (replace with LSTM)
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100 --ablate bissm

# Ablation: Remove HPC (replace with linear)
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100 --ablate hpc

# Ablation: Remove orthogonality loss
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100 --no-ortho-loss

# Ablation: Remove anatomical adjacency
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100 --no-anatomical
```

**Note:** Need to implement `--ablate` flags in benchmark_external.py first.

Expected output table:
| Configuration | WLASL100 | WLASL2000 | Δ |
|---------------|----------|-----------|---|
| Full PhonSSM | 88.37% | 72.08% | — |
| − AGAN | ? | ? | ? |
| − PDM | ? | ? | ? |
| − BiSSM | ? | ? | ? |
| − HPC | ? | ? | ? |
| − Orthogonality | ? | ? | ? |
| − Anatomical adj | ? | ? | ? |

---

### 1.2 Statistical Significance (3+ Seeds)
**Time: 1-2 days | Priority: HIGH**

Re-run best models with 3 different random seeds:

```bash
# WLASL100 - 3 seeds
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100 --seed 42
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100 --seed 123
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100 --seed 456

# WLASL2000 - 3 seeds
python training/benchmark_external.py --dataset wlasl --subset 2000 --epochs 100 --seed 42
python training/benchmark_external.py --dataset wlasl --subset 2000 --epochs 100 --seed 123
python training/benchmark_external.py --dataset wlasl --subset 2000 --epochs 100 --seed 456
```

Report: Mean ± Std for each metric.

---

### 1.3 Phonological Subspace Validation
**Time: 4-6 hours | Priority: HIGH**

Validate that PDM subspaces learn correct phonological features:

```bash
python analysis/validate_subspaces.py --subset 100
```

**Need to implement:** `analysis/validate_subspaces.py`
- Extract PDM embeddings for all test samples
- Project onto each learned subspace (handshape, location, movement, orientation)
- Correlate with ground-truth phonological labels from ASL-LEX
- Report classification accuracy per phonological parameter

Expected output:
| Subspace | Handshape Acc | Location Acc | Movement Acc | Orientation Acc |
|----------|---------------|--------------|--------------|-----------------|
| Subspace 1 | 89% | 23% | 31% | 19% |
| Subspace 2 | 18% | 85% | 27% | 22% |
| Subspace 3 | 21% | 19% | 82% | 25% |
| Subspace 4 | 24% | 21% | 29% | 78% |

---

## Tier 2: Strongly Expected

### 2.1 Minimal Pair Analysis
**Time: 4-6 hours | Priority: MEDIUM**

Test on signs differing by exactly one phonological parameter:

```bash
python analysis/minimal_pairs.py --subset 100
```

**Need to implement:** `analysis/minimal_pairs.py`
- Identify minimal pairs from ASL-LEX (e.g., MOTHER vs FATHER differ only in location)
- Measure pairwise accuracy on these confusable pairs
- Compare to Bi-LSTM baseline

---

### 2.2 SSM vs Transformer Comparison
**Time: 1-2 days | Priority: MEDIUM**

```bash
# BiSSM (ours)
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100

# Transformer replacement
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100 --temporal transformer

# LSTM replacement
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100 --temporal lstm
```

Compare: Accuracy, inference speed, memory, sequence length scaling.

---

## Tier 3: Would Strengthen Significantly

### 3.1 Compositional Generalization Test
**Time: 2-3 days | Priority: MEDIUM**

Test if PhonSSM can recognize novel combinations of known components:

```bash
python analysis/compositional_test.py
```

**Need to implement:** `analysis/compositional_test.py`
- Hold out all signs with a specific (handshape, location) combination
- Train on remaining signs
- Test on held-out combination
- Compare to baseline that can't compose

---

### 3.2 Scaling Laws
**Time: 3-5 days | Priority: MEDIUM**

#### Vocabulary Scaling (partially done!)
Already have: WLASL 100, 300, 1000, 2000
Need: Plot accuracy vs vocab size

```bash
python analysis/plot_scaling.py --type vocab
```

#### Data Scaling
```bash
python training/benchmark_external.py --dataset wlasl --subset 100 --data-fraction 0.1
python training/benchmark_external.py --dataset wlasl --subset 100 --data-fraction 0.25
python training/benchmark_external.py --dataset wlasl --subset 100 --data-fraction 0.5
python training/benchmark_external.py --dataset wlasl --subset 100 --data-fraction 1.0
```

#### Model Scaling
```bash
python training/benchmark_external.py --dataset wlasl --subset 100 --model-scale 0.25  # ~1M params
python training/benchmark_external.py --dataset wlasl --subset 100 --model-scale 1.0   # ~4M params
python training/benchmark_external.py --dataset wlasl --subset 100 --model-scale 2.5   # ~10M params
python training/benchmark_external.py --dataset wlasl --subset 100 --model-scale 7.5   # ~30M params
```

---

### 3.3 Sample Complexity Theorem
**Time: 1 day | Priority: LOW (theoretical)**

Formalize PAC-learning bound for compositional representation:
- N = 5,565 signs
- K = 4 phonological parameters
- |C| ≈ 135 total components
- Claim: O(K × |C|) vs O(N) sample complexity

**Need:** Write proof sketch, compute empirical validation.

---

## Quick Reference: Commands Already Working

```bash
# Train on WLASL
python training/benchmark_external.py --dataset wlasl --subset 100 --epochs 100
python training/benchmark_external.py --dataset wlasl --subset 2000 --epochs 100

# Train on ASL Citizen
python training/benchmark_external.py --dataset asl_citizen --epochs 100

# Resume training
python training/benchmark_external.py --dataset wlasl --subset 2000 --resume benchmarks/external/wlasl2000/TIMESTAMP

# Run analysis
python analysis/confusion_matrix.py --subset 100
python analysis/tsne_phonology.py --subset 100
python analysis/attention_heatmap.py --subset 100
```

---

## Completed Experiments ✅

- [x] Confusion matrix analysis (WLASL100, WLASL2000)
- [x] t-SNE visualization of phonological subspaces
- [x] Attention heatmap visualization
- [x] WLASL benchmarks (100, 300, 1000, 2000)

---

*Last updated: 2026-01-28*
