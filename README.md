# SignSense / PhonSSM

Skeleton-based American Sign Language recognition. No RGB video at inference time — the
models take pose and hand landmarks only, which keeps them small enough to run on a laptop
CPU and avoids storing anyone's face.

The repo holds three things: the model code (`models/`), the training and evaluation
scripts (`training/`, `analysis/`), and a small FastAPI demo plus a Next.js marketing site
(`web/`, `demo-website/`).

## A note on the numbers

An earlier version of this project reported 88.37% on WLASL100 and claimed state of the
art. That number was wrong. Landmark extraction had written three copies of every video
(V1/V2/V3) into the pool (63,249 rows = 3 x 21,083 videos), and the split was random per
row rather than per video, so near-duplicates of test clips sat in training. Retraining
the same architecture and config on a leakage-free per-video split collapses the numbers:

| Subset | As published | Leaky protocol, reproduced | Leakage-free per-video split |
|---|---|---|---|
| WLASL100 | 88.37 | 88.63 | 50.00 |
| WLASL300 | 74.41 | 74.06 | 31.96 |
| WLASL1000 | 62.90 | 61.09 | 17.48 |
| WLASL2000 | 72.08 | 72.76 | 19.67 |

The published 62.90 -> 72.08 rise from 1000 to 2000 classes is impossible under a clean
protocol and was the tell. The right-hand column is the `wlasl_honest` split
(`benchmarks/honest_wlasl*.json`, `benchmarks/inflation_summary.json`): copies of a video
are kept together and the split is taken at the official train/val/test *ratios*, with
residual near-duplicate leakage measured and repaired (276 / 679 / 1899 / 3187 test
videos, measured test leakage 0). It is not the official partition itself, which is used
for the tables below. `training/repro_leaky.py` reproduces the inflated column and
`training/build_honest_split.py` + `training/train_honest.py` rebuild the clean one; both
run from `data/processed/X_wlasl_pose_hands.npy`, which is in the repo.

The full leakage post-mortem (how the bug was found, the per-epoch old-vs-new validation
divergence, and the review of the merged-5565 and "zero-shot" evaluations) was written up
in `EVALUATION_INTEGRITY_REPORT.md`. That file was removed when the repo was cleaned; it
survives in git history at `git show 08a9e30:EVALUATION_INTEGRITY_REPORT.md`, as do
`RESULTS.md`, `TECHNICAL_BRIEF.md`, `APP_DESCRIPTION.md`, `docs/BENCHMARK_PLAN.md` and
`docs/MODELS.md` (`git show f90c1d8^:<path>`).

Everything reported below is on the official WLASL person-independent split
(258 / 668 / 1876 / 2878 test videos). The test split was used to score more than one
configuration -- `benchmarks/official/` holds four scored WLASL100 variants (baseline,
v2_archonly, v2_msaganaug, v3_wide) and the DSTA-Net-port and BiGRU rows below are the
best stream combination out of several -- so treat the gaps between rows as indicative
rather than as a single held-out measurement. Only the PhonoGraph ensemble weights were
genuinely fixed before scoring.

The retracted numbers have NOT been corrected everywhere. The `main` branch of this repo
(its default branch) still carries the pre-retraction README and `paper/` with 88.37 /
74.41 / 62.90 / 72.08 / 53.34 presented as state of the art, and `demo-website/` still
displays "88.4% on WLASL100 -- best skeleton-only result ever published" against a 63.2%
DSTA-SLR baseline that has no source. None of those should be cited. The paper's ablation
table (-AGAN 79.84, -PDM 76.49, -BiSSM 82.17, -HPC 84.11, -Ortho 85.92), its seed
standard deviations, and its 73.2% causal-intervention figure have no backing results file
anywhere in this repo or its history.

## Results

Chasing the leak turned out to be more informative than the original result. Two findings:
the features mattered far more than the architecture, and multi-seed averaging closed most
of the remaining gap to published baselines.

Swapping MediaPipe's 75 landmarks for the 27-joint HRNet whole-body skeletons that
SAM-SLR-v2 and DSTA-SLR use roughly doubled accuracy with a *simpler* model — a plain BiGRU
on HRNet features beat the full PhonSSM on MediaPipe features by 30 points.

WLASL100 top-1, official split, pose-only, no pretraining:

| Model | Top-1 |
|---|---|
| PhonSSM on MediaPipe landmarks | 42.6 |
| BiGRU on HRNet-27, 3-stream ensemble | 73.6 |
| DSTA-Net port (our reimplementation) | 74.4 |
| PhonoGraph, 4-seed ensemble | 83.3 |
| DSTA-SLR (published) | 82.4 |

PhonoGraph is the one architecture here that beat the published baseline honestly. It is a
part-decoupled GCN: motion energy decides which hand is dominant, three feature-isolated
encoders handle body / dominant hand / non-dominant hand, cross-part attention fuses them,
and a prototype-memory head with an orthogonality penalty does the classification. 1.23M
parameters against DSTA-Net's 7.2M. Fusion weights were fixed a priori (joint 1.5, bone
1.0, motion 0.3, bone-motion 0.5) rather than tuned on test; re-running
`training/ensemble_phonograph.py` on the saved per-stream probabilities reproduces 83.33
exactly (a test-tuned grid gives 83.72, which is not the number quoted).

The "beat the published baseline" claim rests on which DSTA-SLR figure is correct, and
this repo is not internally consistent about that. `ensemble_phonograph.py` hard-codes the
target as 82.38, which is what the 82.4 above refers to; the earlier evaluation-integrity
write-up instead cites Hu et al. 2024 at 83.56 for WLASL100. Against 83.56, PhonoGraph's
83.3 does not beat it. Nothing in this repo pins down which value is the right one, and
the DSTA-SLR column here is transcribed from the literature, not re-measured.

Pretraining the DSTA-Net port on the full WLASL2000 train set and fine-tuning down reaches
87.6 on WLASL100, but the benefit shrinks as the target subset grows and vanishes at 2000
classes where the pretraining task *is* the target task:

| Subset | Pretrain + finetune | DSTA-SLR |
|---|---|---|
| WLASL100 | 87.6 | 82.4 |
| WLASL300 | 77.5 | 80.0 |
| WLASL1000 | 62.2 | 67.8 |
| WLASL2000 | 48.8 | 53.7 |

That table is the weakest thing on this page. No results file records those four numbers;
what survives is the per-stream joint and bone runs
(`benchmarks/dstanet_pretrain/ft_wlasl*_{joint,bone}_ft1.json` -- 84.5 / 76.4 / 61.7 /
47.6 for joint) plus their saved test logits. Fusing joint and bone from those logits
lands near the quoted values but does not reproduce them under any single weighting: a
probability-space 1.5:1 fuse gives 87.6 / 78.9 / 62.6 / 48.9, and no weight in a 0.1-3.0
sweep matches all four cells at once. Treat the row as approximate until it is re-run.

The auxiliary models in `models/` (error diagnosis, movement analyzer, feedback ranker)
report very high accuracy, but their labels are deterministic functions of their own
inputs, so those figures are circular and should not be read as evidence of anything.

## Install

```bash
pip install -r requirements.txt
```

PyTorch for the recognition models, TensorFlow for the older Keras auxiliary models, and
MediaPipe only if you need to extract landmarks from new video.

## Running things

### What is and is not in the repo

Only one large input is version-controlled: the triplicated MediaPipe pool
`data/processed/X_wlasl_pose_hands.npy` (+ labels and label map), via Git LFS. Everything
the HRNet-27 results are built on is present on a working checkout but is excluded by
`.gitignore` and therefore is *not* in a fresh clone:

- `data/sota_skeleton/wlasl27/` (978 MB) -- `{train,val}_data_joint.npy` +
  `{train,val}_label.pkl`, the 27-joint HRNet whole-body skeletons in SAM-SLR-v2 format,
  shape (N, C=3, T=150, V=27, M=1). Every number in both tables above comes from these.
  They are the released SAM-SLR-v2 / DSTA-SLR WLASL features
  (https://github.com/jackyjsy/SAM-SLR-v2 -- see its "Data" section for the download
  link); this repo does not re-derive them.
- `data/processed/wlasl_official/wlasl{100,300,1000,2000}/` and
  `data/processed/wlasl_honest/wlasl{...}/` -- the per-subset split tensors.
  `wlasl_official` was built by `training/build_wlasl_official.py` from the raw landmark
  dumps, which have been deleted (see below); `wlasl_honest` is rebuildable from the
  committed pool with `training/build_honest_split.py`.
- `benchmarks/**/*.npy` -- the saved per-stream test logits and probabilities. These are
  what make the ensemble rows re-derivable without retraining (`ensemble_phonograph.py`,
  `ensemble27.py` read them directly), so keep them if you copy this tree.

Deleted and not recoverable from this repo:

- `data/raw/` (~14 GB of per-video MediaPipe landmark `.npz`) and
  `data/wlasl-processed/videos/` (5.1 GB of source clips). `data/wlasl-processed/` still
  holds the metadata that goes with them: `WLASL_v0.3.json`, `nslt_{100,300,1000,2000}.json`,
  `wlasl_class_list.txt`, `missing.txt`.
- `benchmarks/external/wlasl*/*/best_model.pt` -- the original leaky-split checkpoints.
  Their `config.json` / `history.json` / `test_results.json` survive. Because the weights
  are gone, `training/rerun_verify.py` and the three scripts in `analysis/` (which all
  load those checkpoints) cannot be run as-is; `training/repro_leaky.py` retrains the
  leaky protocol from scratch instead. The plots already in `analysis/results/` were
  produced from those leaky-split checkpoints and should be read with that in mind.

### Getting the raw data

WLASL is distributed as a list of source URLs, not as video, and the pre-extracted video
is access-controlled, so budget time for this:

- Official repo and downloader: `git clone https://github.com/dxli94/WLASL.git`, then
  `pip install yt-dlp && python start_kit/video_downloader.py`. Expect missing clips;
  `data/wlasl-processed/missing.txt` lists the ones that were already dead here.
- Pre-processed video by request form (the route used for this project):
  https://docs.google.com/forms/d/e/1FAIpQLSc3yHyAranhpkC9ur_Z-Gu5gS5M0WnKtHV07Vo6eL6nZHzruw/viewform
- Kaggle mirrors: https://www.kaggle.com/datasets/sttaseen/wlasl2000-resized and
  https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed
- HuggingFace: `fiftyone.utils.huggingface.load_from_hub("Voxel51/WLASL")`

Other datasets referenced by the auxiliary/merged pipelines:

- ASL Citizen: https://www.microsoft.com/en-us/download/details.aspx?id=105253 (mirror:
  https://www.kaggle.com/datasets/abd0kamel/asl-citizen)
- MS-ASL: https://www.microsoft.com/en-us/download/details.aspx?id=100121
- ChicagoFSWild: https://home.ttic.edu/~klivescu/ChicagoFSWild.htm, scripted in
  `training/download_chicagofswild.py`
- LSA64: https://facundoq.github.io/datasets/lsa64/

One step of the chain is missing, and getting the video back does not restore it. Both
`training/preprocess_wlasl_full.py` and `training/build_wlasl_official.py` start from
per-video `landmarks_V*.npz` files in `data/raw/wlasl-landmarks/`, and no script in this
repo -- or anywhere in its git history -- produces those from video. The MediaPipe
extraction that wrote them (and that wrote each video out three times, which is what
caused the leak) was never committed. So `data/processed/wlasl_official/` is effectively
a terminal artifact: it can be used, but not rebuilt. If you need to start from video,
`training/extract_pose133.py` (RTMW-x, COCO-WholeBody-133, one `.npy` per clip) is the
surviving extraction path, and it is a different feature set from the one behind
`X_wlasl_pose_hands.npy`.

Train PhonoGraph on one stream and one seed, then ensemble:

```bash
python training/train_phonograph.py --subset 100 --streams joint --base 96 --seed 0 --probs
python training/train_phonograph.py --subset 100 --streams bone --base 96 --seed 0 --probs
python training/ensemble_phonograph.py --subset 100 --tag b96 --seeds 0 1 2 3
```

The DSTA-Net port and its pretrain/finetune variant:

```bash
python training/train_dstanet.py --subset 100 --streams joint --logits
python training/pretrain_finetune_dsta.py pretrain --streams joint
python training/pretrain_finetune_dsta.py finetune --subset 100 --streams joint --logits
```

Rebuild the leakage-free split and retrain the original PhonSSM on it:

```bash
python training/build_honest_split.py --subset 100
python training/train_honest.py --subset 100
```

Analysis plots (confusion matrix, t-SNE over the phonological subspaces, attention maps)
are in `analysis/` and write to `analysis/results/`. As noted above they load the deleted
leaky-split checkpoints, so they need to be repointed at a checkpoint that still exists
before they will run.

The demo server is `python web/server.py`; the Next.js site is `npm run dev` inside
`demo-website/`.

## Known problems

AGAN's anatomical mask used to be a no-op: `A = A_anat + sigmoid(A_learn) * 0.5` is
strictly positive everywhere, so the `adj == 0` mask in the attention layer never fired and
the layer ran as plain dense attention rather than the anatomically masked attention it was
described as. The learnable term is now restricted to real skeletal edges, but every
PhonSSM checkpoint in the repo was trained before that fix, so the old ablation numbers
measure MLP-versus-dense-attention, not masking.

The merged 5,565-class dataset and the "zero-shot" ASL Citizen evaluation both still have
leakage (augment-before-split, and ~80% of ASL Citizen was in training). Those numbers are
not trustworthy and are not quoted above.

## License

MIT.
