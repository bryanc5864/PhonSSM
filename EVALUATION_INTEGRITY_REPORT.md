# PhonSSM — Evaluation Integrity Report

**Status: the reported benchmark numbers are invalid as computed.** They are
inflated by train/test data leakage and are not comparable to the published
baselines they are shown against. The model architecture is real and trains, and
the metric math is correct — the problem is entirely in how the data was split
and evaluated.

This report was produced by (1) an independent re-run of the saved checkpoints,
(2) a multi-agent code review with adversarial verification of every high-severity
finding, and (3) writing corrected, leakage-free evaluation code.

---

## TL;DR

| Claim (as published) | Reality |
|---|---|
| WLASL100 **88.4%** (SOTA, +25 pts over DSTA-SLR) | Leaky split. Honest official-split ≈ **high-40s (~48%)**. Real DSTA-SLR baseline is **83.56%**, not 63.2%. |
| WLASL2000 **72.1%** (+18.4 pts) | Leaky split; **72.1% > WLASL1000 62.9% is impossible** under a clean protocol — a direct symptom of leakage. |
| Merged-5565 **53.3%** | Inflated (augment-before-split leakage + signer leakage) and **non-reproducible** (unseeded split RNG). |
| Zero-shot ASL Citizen **64.1% > RGB 63.2%** | **Not zero-shot**: ~80% of ASL Citizen was in training and its classes were in-vocabulary. |
| "Anatomical masked graph attention" (AGAN) | **No-op mask** — the code runs fully-connected dense attention. |

---

## The core leakage (WLASL)

Two things combined:

1. **Every WLASL video was triplicated.** Preprocessing extracted landmarks
   three times per recording (`landmarks_V1/V2/V3.npz`), so each physical video
   appears as **3 near-identical rows** (63,249 rows = 3 × 21,083 videos).
2. **The split was random per-row, not per-video.**
   `training/benchmark_external.py` loaded the official WLASL split *only to
   compute train/val ratios*, then discarded it and used a random
   `train_test_split(random_state=42)` on the pooled, triplicated array.

**Consequence:** the three copies of one video get scattered — e.g. V1→train,
V2→test, V3→train. At test time the model classifies V2 of a clip it already
trained on (V1, V3): same signer, same sign, same motion. That measures
*memorization of near-duplicates*, not generalization.

The official WLASL protocol (used by DSTA-SLR, I3D, Pose-TGCN) assigns each
**video** to a split, so a test video's copies are never in training — which is
why the leaky number is also not comparable to those baselines.

### Evidence: same model, same seed, only the split differs

| WLASL100 val acc | Old (leaky random split) | New (honest official split) |
|---|---|---|
| epoch 10 | 38.69% | 29.35% |
| epoch 20 | **68.87%** | **39.63%** |
| epoch 30 | 81.13% | ~46% |
| epoch 50 | 85.57% | ~48% |
| epoch 99 | **88.02%** | **~48% (plateaued)** |

At epoch 20 the leaky run is already ~29 points higher on validation, purely
from the split. (The honest run was stopped before full convergence; val had
plateaued at ~48%. A completed run is still needed for the exact final figure.)

### The smoking gun: impossible monotonicity

Reported: WLASL100 88.4 · WLASL300 74.4 · WLASL1000 62.9 · **WLASL2000 72.1**.
Accuracy cannot *increase* from 1000→2000 classes under a clean protocol
(cf. I3D 65.9 → 56.1 → 47.3 → 32.5). The 62.9 → 72.1 inversion is a direct
symptom of leakage scaling with pool size, reproduced on the held-out test set.

---

## Other confirmed issues

- **Merged-5565 (53.3%) — augment-before-split leakage:** `preprocess.py`
  augmented the full array 5× and *then* split, so ~4 near-identical siblings of
  every val/test sample sat in train (and val/test were themselves augmented).
  These leaky splits propagate into the merge.
- **ASL Citizen signer leakage + non-reproducibility:** the official
  signer-independent split is dropped at preprocessing; `merge_all_datasets.py`
  re-splits with an **unseeded** random 80/10/10, mixing the same signers across
  train/test and making the exact 76/12/12 partition unregenerable.
- **"Zero-shot" is not zero-shot:** `evaluate_zeroshot.py` scores a seed-42
  re-slice of `X_asl_citizen.npy`, but ~80% of ASL Citizen entered merged
  training and every ASL Citizen gloss is in the 5,565-way vocabulary. The
  "2,731 unseen classes" framing is false; the beat over supervised RGB is invalid.
- **Fabricated DSTA-SLR baseline:** the README/site "63.18/58.42/47.14" figure
  has **no code provenance** and contradicts the paper's own cited Hu et al. 2024
  value (**83.56**). Against the real baseline, WLASL100 is ~+4.8 pts and PhonSSM
  *loses* on WLASL300/1000 (the precision–generalization tradeoff).
- **AGAN anatomical mask is a no-op:** `A = A_anat + sigmoid(A_learn)*0.5` is
  strictly positive everywhere, so the GATLayer's `(adj==0)` mask never fires —
  it is dense attention, not the anatomical masked attention the paper describes.
- **Aux-model accuracies are circular:** Movement Analyzer (100%) / Feedback
  Ranker (99%) targets are deterministic functions of their own inputs.

The metric computation itself (`calculate_accuracy`) is correct — no
axis/top-k/label-remap bug. The inflation is entirely splits + augmentation +
provenance.

---

## Fixes applied in this commit

- **`training/build_wlasl_official.py`** (new): builds WLASL subsets honoring the
  **official per-video split** (partition by `instance['split']`), keeps V1/V2/V3
  together on one side, dedups the **test set to one video per row**, and asserts
  **no video_id overlap** between train and test.
- **`training/retrain_official.py`** (new): trains a fresh model on official-train
  and evaluates on official-test (GPU-aware). Needed because the saved checkpoints
  were trained on the leaky split (they already saw the test videos).
- **`training/rerun_verify.py`** (new): reproduces the reported (leaky) numbers
  from the saved checkpoints — confirms provenance and the monotonicity inversion.
- **`training/benchmark_external.py`**: `load_wlasl_splits` now loads the
  leakage-free official split and **hard-fails** on the old random-split path.
- **`training/preprocess.py`**: split **before** augment; augment **train only**.
- **`training/merge_all_datasets.py`**: **seeded** the split; documented the
  ASL-Citizen signer-leakage limitation and the proper fix.
- **`models/phonssm/agan.py`**: documented the no-op mask and the exact fix
  (left unchanged so the honest re-eval measures the model *as published*).

The original leaky-split training history (`benchmarks/external/wlasl*/…/` with
`best_model.pt`, `history.json`, `test_results.json`) is preserved unchanged for
the record and comparison.

---

## What still needs to be done

1. **Finish the honest WLASL re-runs** (100/300/1000/2000) on official splits to
   lock the corrected numbers (WLASL100 preliminary ~48%).
2. **Rebuild Merged-5565** with seeded, group/official splits (augment train only;
   carry ASL Citizen's official signer-independent split) and retrain.
3. **Real zero-shot:** exclude ASL Citizen from training and evaluate on its
   official held-out test split.
4. **Decide on AGAN:** either implement the anatomical mask and retrain, or
   correct the paper's wording to "adaptive dense attention with an anatomical prior."
5. **Correct README / TECHNICAL_BRIEF / website / paper** to the honest numbers
   and use the real DSTA-SLR baselines.

Until 1–5 are done, none of the comparative claims (SOTA, +25 pts, beats RGB,
zero-shot) should be presented as evidence.
