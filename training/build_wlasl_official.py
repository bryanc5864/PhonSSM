"""
Build WLASL subset data honoring the OFFICIAL train/val/test split, with
source-video association (video_id) so we can (a) split by official protocol and
(b) dedup the test set to one copy per source video.

Fixes the leakage in benchmark_external.py (random split of triplicated pool).
Reuses the repo's own landmark extraction/normalization for faithfulness.
"""
import os, sys, json, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from training.preprocess_wlasl_full import (extract_pose_and_hands, normalize_pose_hands,
                                            pad_or_truncate, temporal_impute)

LM_DIR = ROOT / 'data' / 'raw' / 'wlasl-landmarks'
OUT = ROOT / 'data' / 'processed' / 'wlasl_official'
OUT.mkdir(parents=True, exist_ok=True)


def video_id_of(entry):
    return Path(entry['video_path']).stem


def build(subset_size, max_frames=30, dedup_test=True):
    parsed = json.load(open(LM_DIR / 'WLASL_parsed_data.json'))
    # top-N glosses by first appearance order = WLASL_v0.3 top-N convention.
    # match benchmark_external: it used WLASL_v0.3.json[:subset] gloss order.
    v03 = json.load(open(ROOT / 'data' / 'wlasl-processed' / 'WLASL_v0.3.json'))
    subset_glosses = [e['gloss'].lower() for e in v03[:subset_size]]
    gloss_to_idx = {g: i for i, g in enumerate(subset_glosses)}

    rows = {'train': defaultdict(list), 'val': defaultdict(list), 'test': defaultdict(list)}
    # rows[split][video_id] = list of feature arrays (V1/V2/V3 copies)
    counts = Counter()
    npz_files = sorted(LM_DIR.glob('landmarks_V*.npz'))
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        for key in tqdm(data.files, desc=npz_path.name, leave=False):
            idx = int(key)
            if idx >= len(parsed):
                continue
            e = parsed[idx]
            g = e['gloss'].lower()
            if g not in gloss_to_idx:
                continue
            split = e.get('split', 'train')
            lm = np.asarray(data[key])
            if len(lm) == 0:
                continue
            # landmark-ordering guard: extract_pose_and_hands assumes the 180-pt
            # MediaPipe-Holistic layout (pose 0-32, L-hand 33-53, R-hand 54-74,
            # face 75-179). Other formats (e.g. the 553-pt V3 file) would silently
            # slice the wrong joints, so skip anything that is not 180-wide.
            if lm.ndim != 3 or lm.shape[1] < 75:
                continue
            raw = temporal_impute(np.nan_to_num(extract_pose_and_hands(lm), nan=0.0))
            ph = pad_or_truncate(normalize_pose_hands(raw), max_frames)
            rows[split][video_id_of(e)].append((gloss_to_idx[g], ph.reshape(max_frames, -1).astype(np.float32)))
            counts[split] += 1

    def pack(split, dedup):
        X, y, vids = [], [], []
        for vid, lst in rows[split].items():
            use = lst[:1] if dedup else lst
            for label, feat in use:
                X.append(feat); y.append(label); vids.append(vid)
        if not X:
            return np.zeros((0, max_frames, 225), np.float32), np.zeros((0,), np.int64), []
        return np.stack(X), np.array(y, np.int64), vids

    Xtr, ytr, vtr = pack('train', dedup=False)   # keep V1/V2/V3 in train (augmentation)
    Xva, yva, vva = pack('val', dedup=True)
    Xte, yte, vte = pack('test', dedup=dedup_test)

    # leakage guard: no video_id shared across train/test
    inter = set(vtr) & set(vte)
    assert not inter, f"LEAK: {len(inter)} video_ids in both train and test"

    out_root = OUT if max_frames == 30 else OUT.parent / f'wlasl_official_f{max_frames}'
    out_root.mkdir(parents=True, exist_ok=True)
    d = out_root / f'wlasl{subset_size}'
    d.mkdir(exist_ok=True)
    np.save(d / 'X_train.npy', Xtr); np.save(d / 'y_train.npy', ytr)
    np.save(d / 'X_val.npy', Xva); np.save(d / 'y_val.npy', yva)
    np.save(d / 'X_test.npy', Xte); np.save(d / 'y_test.npy', yte)
    json.dump({'subset': subset_size, 'raw_counts': dict(counts),
               'train_rows': int(len(ytr)), 'val_rows': int(len(yva)), 'test_rows': int(len(yte)),
               'train_videos': len(set(vtr)), 'test_videos': len(set(vte)),
               'classes_in_test': int(len(set(yte.tolist()))), 'num_classes': subset_size},
              open(d / 'meta.json', 'w'), indent=2)
    print(f"WLASL{subset_size}: train {len(ytr)} rows / {len(set(vtr))} vids | "
          f"val {len(yva)} | test {len(yte)} rows / {len(set(vte))} vids | "
          f"test classes {len(set(yte.tolist()))}/{subset_size}")
    return d


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=100)
    ap.add_argument('--frames', type=int, default=30)
    a = ap.parse_args()
    build(a.subset, max_frames=a.frames)
