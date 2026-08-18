"""
Preprocess WLASL data with full pose + both hands.
This enables fair comparison with SOTA methods that use full skeleton.

Output: (N, 30, 225) where 225 = 75 landmarks * 3 coords
Layout: [Pose(33) + Left Hand(21) + Right Hand(21)] * 3 coords
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent


def extract_pose_and_hands(landmarks):
    """(frames, 180, 3) Holistic -> (frames, 75, 3) pose + both hands.

    WLASL layout: 0-32 pose, 33-53 left hand, 54-74 right hand, 75-179 face (dropped).
    """
    pose = landmarks[:, 0:33, :]       # 33 landmarks
    left_hand = landmarks[:, 33:54, :]  # 21 landmarks
    right_hand = landmarks[:, 54:75, :] # 21 landmarks

    return np.concatenate([pose, left_hand, right_hand], axis=1)


def normalize_pose_hands(landmarks, clip=5.0):
    """
    Robust, position/scale-invariant normalization computed ONCE PER SEQUENCE.

    The previous implementation divided EVERY frame by that frame's own raw
    shoulder width (floored only at 1e-6). When MediaPipe collapses the shoulders
    (subject turned, or a bad detection) the width is tiny and coordinates are
    amplified to +/-10^4; and because the scale is recomputed per frame it also
    injects frame-to-frame scale flicker that destroys the motion signal. Missing
    (all-zero) joints were mapped to a spurious -center/scale location instead of
    staying "absent".

    This version:
      - uses ONE center (median shoulder-midpoint over frames where both
        shoulders are present) and ONE scale (median shoulder width over those
        frames) for the whole clip;
      - is scale-agnostic: if the shoulder width is degenerate it falls back to
        the median radius of present joints, so it works whether coords are in
        normalized [0,1] image space or pixel space;
      - leaves genuinely missing joints at 0 AFTER scaling (not -center/scale);
      - clips to +/-`clip` so a residual bad frame cannot blow up the features.

    Args:
        landmarks: (frames, 75, 3) array
    Returns:
        (frames, 75, 3) float32, values in [-clip, clip], missing joints = 0
    """
    lm = np.nan_to_num(np.asarray(landmarks, dtype=np.float32), nan=0.0)
    if lm.ndim != 3:
        return lm
    F, J, C = lm.shape
    present = ~np.all(lm == 0.0, axis=2)                      # (F, J) joint visible?

    ls, rs = lm[:, 11, :], lm[:, 12, :]                       # shoulders (pose 11,12)
    sh_ok = present[:, 11] & present[:, 12]                   # frames with both shoulders

    if sh_ok.any():
        center = np.median((ls[sh_ok] + rs[sh_ok]) / 2.0, axis=0)
        scale = float(np.median(np.linalg.norm(ls[sh_ok] - rs[sh_ok], axis=1)))
    else:
        pts = lm[present]
        center = np.median(pts, axis=0) if len(pts) else np.zeros(3, np.float32)
        scale = 0.0

    # robust fallback / floor: median radius of present joints about the center.
    pts = lm[present]
    if len(pts):
        radius = float(np.median(np.linalg.norm(pts - center, axis=1)))
    else:
        radius = 1.0
    floor = max(0.2 * radius, 1e-3)
    if not np.isfinite(scale) or scale < floor:
        scale = floor if (np.isfinite(radius) and radius > 0) else 1.0

    out = (lm - center) / scale
    out[~present] = 0.0
    out = np.clip(out, -clip, clip)
    return np.nan_to_num(out, nan=0.0, posinf=clip, neginf=-clip).astype(np.float32)


def temporal_impute(landmarks):
    """fill all-zero joints by linear interpolation over time.

    MediaPipe drops hands in ~30% of frames and leaving them at 0 injects huge
    fake jumps. ends are held at the nearest value; joints never seen stay 0.
    """
    lm = np.asarray(landmarks, dtype=np.float32).copy()
    Fr, J, C = lm.shape
    present = ~np.all(lm == 0.0, axis=2)          # (Fr, J)
    t = np.arange(Fr)
    for j in range(J):
        p = present[:, j]
        if p.all() or not p.any():
            continue
        for c in range(C):
            lm[:, j, c] = np.interp(t, t[p], lm[p, j, c])
    return lm


def pad_or_truncate(sequence, target_length=30):
    """Resample a sequence to a fixed length by linear interpolation over time.

    The previous version integer-index-subsampled (aliasing / dropping motion)
    and padded short clips by repeating the last frame. Linear interpolation
    resamples smoothly and preserves the movement trajectory, which is the main
    discriminative cue for many signs.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    current_length = len(sequence)
    if current_length == target_length:
        return sequence
    if current_length == 1:
        return np.repeat(sequence, target_length, axis=0)
    src = np.linspace(0.0, current_length - 1, target_length)
    lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, current_length - 1)
    frac = (src - lo).reshape(-1, *([1] * (sequence.ndim - 1)))
    return (sequence[lo] * (1 - frac) + sequence[hi] * frac).astype(np.float32)


def preprocess_wlasl_full(data_dir, output_dir, max_frames=30):
    """preprocess every WLASL clip into pose + both hands, max_frames long."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load parsed data for gloss mapping
    parsed_path = data_dir / 'WLASL_parsed_data.json'
    with open(parsed_path, 'r') as f:
        parsed_data = json.load(f)

    print(f"Total entries in parsed data: {len(parsed_data)}")

    # get all available glosses
    available_glosses = sorted(set(entry['gloss'].lower() for entry in parsed_data))
    print(f"Total unique signs: {len(available_glosses)}")

    # create label mapping for ALL signs
    sign_to_idx = {s: i for i, s in enumerate(available_glosses)}
    idx_to_sign = {i: s for s, i in sign_to_idx.items()}

    # load all npz files
    npz_files = list(data_dir.glob('landmarks_V*.npz'))
    print(f"Found {len(npz_files)} landmark files")

    X_data = []
    y_data = []
    skipped = 0
    processed = 0

    for npz_path in npz_files:
        print(f"\nProcessing {npz_path.name}...")
        data = np.load(npz_path, allow_pickle=True)

        for key in tqdm(data.files, desc=f"Loading {npz_path.name}"):
            try:
                idx = int(key)
                if idx >= len(parsed_data):
                    skipped += 1
                    continue

                gloss = parsed_data[idx]['gloss'].lower()

                if gloss not in sign_to_idx:
                    skipped += 1
                    continue

                # load landmarks (frames, 180, 3)
                landmarks = data[key]

                if len(landmarks) == 0:
                    skipped += 1
                    continue

                # extract pose + both hands (frames, 75, 3)
                pose_hands = extract_pose_and_hands(landmarks)

                # handle NaN
                pose_hands = np.nan_to_num(pose_hands, nan=0.0)

                pose_hands = normalize_pose_hands(pose_hands)

                # adjust to fixed length
                pose_hands = pad_or_truncate(pose_hands, max_frames)

                # flatten for model input: (30, 225)
                X_data.append(pose_hands.reshape(max_frames, -1))
                y_data.append(sign_to_idx[gloss])
                processed += 1

            except Exception as e:
                skipped += 1
                continue

    print(f"\nProcessed: {processed}, Skipped: {skipped}")

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)

    print(f"\nFinal data shape: X={X.shape}, y={y.shape}")
    print(f"Features per frame: {X.shape[-1]} (75 landmarks * 3 coords)")

    np.save(output_dir / 'X_wlasl_pose_hands.npy', X)
    np.save(output_dir / 'y_wlasl_pose_hands.npy', y)

    with open(output_dir / 'wlasl_pose_hands_label_map.json', 'w') as f:
        json.dump(sign_to_idx, f, indent=2)

    print(f"\nSaved to {output_dir}:")
    print(f"  X_wlasl_pose_hands.npy: {X.shape}")
    print(f"  y_wlasl_pose_hands.npy: {y.shape}")
    print(f"  wlasl_pose_hands_label_map.json: {len(sign_to_idx)} signs")

    return X, y, sign_to_idx


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess WLASL with pose + hands')
    parser.add_argument('--data-dir', type=str,
                        default='data/raw/wlasl-landmarks',
                        help='Path to wlasl-landmarks directory')
    parser.add_argument('--output-dir', type=str,
                        default='data/processed',
                        help='Output directory')

    args = parser.parse_args()

    preprocess_wlasl_full(args.data_dir, args.output_dir)
