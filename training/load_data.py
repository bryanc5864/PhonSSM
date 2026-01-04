"""
SignSense Data Loading Module
Loads and preprocesses Kaggle ASL-Signs dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json


# 50 Target Signs - from Kaggle ASL-Signs dataset
# Selected for practical daily use and learning progression
TARGET_SIGNS = [
    # Greetings & Basics (6)
    'hello', 'bye', 'please', 'thankyou', 'yes', 'no',
    # Family (10)
    'mom', 'dad', 'grandma', 'grandpa', 'brother', 'aunt', 'uncle',
    'boy', 'girl', 'child',
    # Feelings (7)
    'happy', 'sad', 'mad', 'sick', 'hungry', 'thirsty', 'sleepy',
    # Actions (12)
    'go', 'wait', 'finish', 'give', 'like', 'have', 'look', 'listen',
    'read', 'drink', 'sleep', 'wake',
    # Objects & Places (6)
    'home', 'bed', 'book', 'water', 'food', 'milk',
    # Time (5)
    'morning', 'night', 'now', 'tomorrow', 'yesterday',
    # Questions (3)
    'where', 'who', 'why',
    # Other useful (1)
    'all'
]


def get_available_signs(data_dir):
    """Get list of all signs available in the dataset."""
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    return sorted(train_df['sign'].unique().tolist())


def filter_target_signs(available_signs, target_signs=TARGET_SIGNS):
    """Filter target signs to only those available in dataset."""
    available_set = set(available_signs)
    valid_targets = [s for s in target_signs if s in available_set]
    missing = [s for s in target_signs if s not in available_set]

    if missing:
        print(f"Warning: {len(missing)} target signs not in dataset: {missing[:10]}...")

    return valid_targets


def normalize_landmarks(landmarks):
    """
    Normalize landmarks to be position/scale invariant.

    Args:
        landmarks: (frames, 21, 3) array of hand landmarks

    Returns:
        Normalized landmarks centered at wrist and scaled
    """
    normalized = []

    for frame in landmarks:
        # Skip empty frames
        if np.all(frame == 0) or np.all(np.isnan(frame)):
            normalized.append(np.zeros_like(frame))
            continue

        # Center at wrist (landmark 0)
        wrist = frame[0]
        centered = frame - wrist

        # Scale by max distance from wrist
        distances = np.linalg.norm(centered, axis=1)
        scale = distances.max()

        if scale > 1e-6:
            centered = centered / scale

        normalized.append(centered)

    return np.array(normalized)


def pad_or_truncate(sequence, target_length=30):
    """
    Adjust sequence to fixed length.

    Args:
        sequence: (frames, ...) array
        target_length: desired number of frames

    Returns:
        Sequence with exactly target_length frames
    """
    current_length = len(sequence)

    if current_length == target_length:
        return sequence
    elif current_length < target_length:
        # Pad with edge replication
        pad_length = target_length - current_length
        padding = np.repeat(sequence[-1:], pad_length, axis=0)
        return np.concatenate([sequence, padding], axis=0)
    else:
        # Subsample to target length
        indices = np.linspace(0, current_length - 1, target_length).astype(int)
        return sequence[indices]


def get_dominant_hand(landmarks):
    """
    Determine and return dominant hand based on motion magnitude.

    Args:
        landmarks: (frames, 42, 3) array with both hands

    Returns:
        (frames, 21, 3) array of dominant hand
    """
    left_hand = landmarks[:, :21, :]
    right_hand = landmarks[:, 21:, :]

    # Calculate motion magnitude (std of positions over time)
    left_motion = np.nanstd(left_hand)
    right_motion = np.nanstd(right_hand)

    # Also check for presence (non-zero values)
    left_present = np.sum(~np.isnan(left_hand) & (left_hand != 0))
    right_present = np.sum(~np.isnan(right_hand) & (right_hand != 0))

    # Prefer hand with more presence and motion
    left_score = left_motion * left_present
    right_score = right_motion * right_present

    return left_hand if left_score > right_score else right_hand


def load_parquet_file(parquet_path):
    """
    Load and extract hand landmarks from a parquet file.

    The Kaggle ASL-Signs data is in long format with columns:
    frame, row_id, type, landmark_index, x, y, z

    Args:
        parquet_path: Path to parquet file

    Returns:
        (frames, 42, 3) array of hand landmarks (both hands)
    """
    df = pd.read_parquet(parquet_path)

    # Filter to hand landmarks only
    hand_df = df[df['type'].isin(['left_hand', 'right_hand'])].copy()

    if len(hand_df) == 0:
        raise ValueError(f"No hand landmarks found in {parquet_path}")

    # Get unique frames
    frames = sorted(hand_df['frame'].unique())
    n_frames = len(frames)

    # Initialize output array: (n_frames, 42 landmarks, 3 coords)
    # 42 = 21 left + 21 right
    landmarks = np.zeros((n_frames, 42, 3), dtype=np.float32)

    # Fill in landmarks
    for i, frame_num in enumerate(frames):
        frame_data = hand_df[hand_df['frame'] == frame_num]

        for _, row in frame_data.iterrows():
            lm_idx = int(row['landmark_index'])

            # Left hand: indices 0-20, Right hand: indices 21-41
            if row['type'] == 'left_hand':
                idx = lm_idx
            else:  # right_hand
                idx = 21 + lm_idx

            landmarks[i, idx, 0] = row['x']
            landmarks[i, idx, 1] = row['y']
            landmarks[i, idx, 2] = row['z']

    # Handle NaN
    landmarks = np.nan_to_num(landmarks, nan=0.0)

    return landmarks


def load_kaggle_data(data_dir, target_signs=None, max_frames=30, max_samples=None):
    """
    Load and preprocess Kaggle ASL-Signs dataset.

    Args:
        data_dir: Path to asl-signs directory
        target_signs: List of signs to include (None = use TARGET_SIGNS)
        max_frames: Fixed sequence length
        max_samples: Maximum samples to load (None = all)

    Returns:
        X: (N, max_frames, 63) array of normalized landmarks
        y: (N,) array of sign indices
        sign_to_idx: Dictionary mapping sign names to indices
        idx_to_sign: Dictionary mapping indices to sign names
    """
    data_dir = Path(data_dir)

    # Load metadata
    train_df = pd.read_csv(data_dir / 'train.csv')
    print(f"Total samples in dataset: {len(train_df)}")

    # Filter to only participants whose folders exist
    landmark_dir = data_dir / 'train_landmark_files'
    existing_participants = [int(p.name) for p in landmark_dir.iterdir() if p.is_dir()]
    train_df = train_df[train_df['participant_id'].isin(existing_participants)]
    print(f"Samples from existing participants: {len(train_df)}")

    # Get available signs
    available_signs = get_available_signs(data_dir)
    print(f"Available signs: {len(available_signs)}")

    # Filter to target signs
    if target_signs is None:
        target_signs = filter_target_signs(available_signs)
    else:
        target_signs = filter_target_signs(available_signs, target_signs)

    print(f"Using {len(target_signs)} target signs")

    # Filter dataframe
    train_df = train_df[train_df['sign'].isin(target_signs)]
    print(f"Filtered samples: {len(train_df)}")

    if max_samples:
        train_df = train_df.head(max_samples)
        print(f"Limited to {len(train_df)} samples")

    # Create label mapping
    signs = sorted(train_df['sign'].unique())
    sign_to_idx = {s: i for i, s in enumerate(signs)}
    idx_to_sign = {i: s for s, i in sign_to_idx.items()}

    X_data = []
    y_data = []
    failed = 0

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Loading data"):
        parquet_path = data_dir / row['path']

        try:
            # Load landmarks
            landmarks = load_parquet_file(parquet_path)

            # Get dominant hand
            hand = get_dominant_hand(landmarks)

            # Normalize
            hand = normalize_landmarks(hand)

            # Window to fixed length
            hand = pad_or_truncate(hand, max_frames)

            # Flatten for model input: (30, 63)
            X_data.append(hand.reshape(max_frames, -1))
            y_data.append(sign_to_idx[row['sign']])

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"Error loading {parquet_path}: {e}")
            continue

    if failed > 0:
        print(f"Failed to load {failed} files")

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)

    print(f"\nLoaded {len(X)} samples")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Signs: {len(sign_to_idx)}")

    return X, y, sign_to_idx, idx_to_sign


def save_label_map(sign_to_idx, output_path):
    """Save label mapping to JSON."""
    with open(output_path, 'w') as f:
        json.dump(sign_to_idx, f, indent=2)
    print(f"Saved label map to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Load Kaggle ASL-Signs data')
    parser.add_argument('--data-dir', type=str,
                        default='data/raw/asl-signs',
                        help='Path to asl-signs directory')
    parser.add_argument('--output-dir', type=str,
                        default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to load (for testing)')

    args = parser.parse_args()

    # Load data
    X, y, sign_to_idx, idx_to_sign = load_kaggle_data(
        args.data_dir,
        max_samples=args.max_samples
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'X_raw.npy', X)
    np.save(output_dir / 'y_raw.npy', y)
    save_label_map(sign_to_idx, output_dir / 'label_map.json')

    print(f"\nSaved to {output_dir}")
