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
    """
    Extract pose + both hands from MediaPipe Holistic landmarks.

    WLASL landmarks structure (180 total):
    - Pose: indices 0-32 (33 landmarks)
    - Left hand: indices 33-53 (21 landmarks)
    - Right hand: indices 54-74 (21 landmarks)
    - Face: indices 75-179 (105 landmarks) - skip for now

    Args:
        landmarks: (frames, 180, 3) array

    Returns:
        (frames, 75, 3) array with pose + both hands
    """
    pose = landmarks[:, 0:33, :]       # 33 landmarks
    left_hand = landmarks[:, 33:54, :]  # 21 landmarks
    right_hand = landmarks[:, 54:75, :] # 21 landmarks

    return np.concatenate([pose, left_hand, right_hand], axis=1)


def normalize_pose_hands(landmarks):
    """
    Normalize pose+hands landmarks to be position/scale invariant.

    Uses the midpoint between shoulders as the origin and
    shoulder width as the scale reference.

    Args:
        landmarks: (frames, 75, 3) array

    Returns:
        Normalized landmarks
    """
    normalized = []

    for frame in landmarks:
        # Skip empty frames
        if np.all(frame == 0) or np.all(np.isnan(frame)):
            normalized.append(np.zeros_like(frame))
            continue

        # Get shoulder positions (indices 11 and 12 in pose)
        left_shoulder = frame[11]
        right_shoulder = frame[12]

        # Center at midpoint between shoulders
        center = (left_shoulder + right_shoulder) / 2
        centered = frame - center

        # Scale by shoulder width (or max distance if shoulders not visible)
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        if shoulder_width < 1e-6:
            # Fallback: use max distance from center
            distances = np.linalg.norm(centered, axis=1)
            scale = distances.max() if distances.max() > 1e-6 else 1.0
        else:
            scale = shoulder_width

        centered = centered / scale
        normalized.append(centered)

    return np.array(normalized)


def pad_or_truncate(sequence, target_length=30):
    """Adjust sequence to fixed length."""
    current_length = len(sequence)

    if current_length == target_length:
        return sequence
    elif current_length < target_length:
        pad_length = target_length - current_length
        padding = np.repeat(sequence[-1:], pad_length, axis=0)
        return np.concatenate([sequence, padding], axis=0)
    else:
        indices = np.linspace(0, current_length - 1, target_length).astype(int)
        return sequence[indices]


def preprocess_wlasl_full(data_dir, output_dir, max_frames=30):
    """
    Preprocess all WLASL data with pose + both hands.

    Args:
        data_dir: Path to wlasl-landmarks directory
        output_dir: Output directory
        max_frames: Fixed sequence length
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load parsed data for gloss mapping
    parsed_path = data_dir / 'WLASL_parsed_data.json'
    with open(parsed_path, 'r') as f:
        parsed_data = json.load(f)

    print(f"Total entries in parsed data: {len(parsed_data)}")

    # Get all available glosses
    available_glosses = sorted(set(entry['gloss'].lower() for entry in parsed_data))
    print(f"Total unique signs: {len(available_glosses)}")

    # Create label mapping for ALL signs
    sign_to_idx = {s: i for i, s in enumerate(available_glosses)}
    idx_to_sign = {i: s for s, i in sign_to_idx.items()}

    # Load all npz files
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

                # Load landmarks (frames, 180, 3)
                landmarks = data[key]

                if len(landmarks) == 0:
                    skipped += 1
                    continue

                # Extract pose + both hands (frames, 75, 3)
                pose_hands = extract_pose_and_hands(landmarks)

                # Handle NaN
                pose_hands = np.nan_to_num(pose_hands, nan=0.0)

                # Normalize
                pose_hands = normalize_pose_hands(pose_hands)

                # Adjust to fixed length
                pose_hands = pad_or_truncate(pose_hands, max_frames)

                # Flatten for model input: (30, 225)
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

    # Save
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
