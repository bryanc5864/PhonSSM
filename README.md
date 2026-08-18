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
(V1/V2/V3) into the pool, and the split was random per row rather than per video, so
near-duplicates of test clips sat in training. Re-running the same checkpoints on a
leakage-free official per-video split gives 50.0% on WLASL100 and 17-20% on the larger
subsets. `training/repro_leaky.py` reproduces the inflated numbers and
`training/build_honest_split.py` builds the clean ones, if you want to see the gap
yourself.

Everything reported below is on the official WLASL person-independent split
(258 / 668 / 1876 / 2878 test videos), and the test set was scored once.

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
1.0, motion 0.3, bone-motion 0.5) rather than tuned on test.

Pretraining the DSTA-Net port on the full WLASL2000 train set and fine-tuning down reaches
87.6 on WLASL100, but the benefit shrinks as the target subset grows and vanishes at 2000
classes where the pretraining task *is* the target task:

| Subset | Pretrain + finetune | DSTA-SLR |
|---|---|---|
| WLASL100 | 87.6 | 82.4 |
| WLASL300 | 77.5 | 80.0 |
| WLASL1000 | 62.2 | 67.8 |
| WLASL2000 | 48.8 | 53.7 |

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

The 27-joint HRNet skeletons live in `data/sota_skeleton/wlasl27/` and the derived
per-subset splits in `data/processed/`. Raw video and raw landmark dumps are not in the
repo; regenerate them with the preprocessing scripts if you need them.

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
are in `analysis/` and write to `analysis/results/`.

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
