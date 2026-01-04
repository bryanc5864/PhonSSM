"""
SignSense WLASL Data Loading Module
Loads and preprocesses WLASL landmarks dataset.
"""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


# Same target signs as MVP - for consistency
TARGET_SIGNS = [
    'hello', 'bye', 'please', 'thankyou', 'yes', 'no',
    'mom', 'dad', 'grandma', 'grandpa', 'brother', 'aunt', 'uncle',
    'boy', 'girl', 'child',
    'happy', 'sad', 'mad', 'sick', 'hungry', 'thirsty', 'sleepy',
    'go', 'wait', 'finish', 'give', 'like', 'have', 'look', 'listen',
    'read', 'drink', 'sleep', 'wake',
    'home', 'bed', 'book', 'water', 'food', 'milk',
    'morning', 'night', 'now', 'tomorrow', 'yesterday',
    'where', 'who', 'why',
    'all'
]


def extract_hands_from_holistic(landmarks):
    """
    Extract hand landmarks from MediaPipe Holistic landmarks.

    WLASL landmarks structure (180 total):
    - Pose: indices 0-32 (33 landmarks)
    - Left hand: indices 33-53 (21 landmarks)
    - Right hand: indices 54-74 (21 landmarks)
    - Face: indices 75-179 (105 landmarks)

    Args:
        landmarks: (frames, 180, 3) array

    Returns:
        (frames, 42, 3) array with both hands
    """
    left_hand = landmarks[:, 33:54, :]   # 21 landmarks
    right_hand = landmarks[:, 54:75, :]  # 21 landmarks

    # Combine: left_hand first, then right_hand
    return np.concatenate([left_hand, right_hand], axis=1)


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


def load_wlasl_data(data_dir, target_signs=None, max_frames=30, load_all=False):
    """
    Load and preprocess WLASL landmarks dataset.

    Args:
        data_dir: Path to wlasl-landmarks directory
        target_signs: List of signs to include (None = use TARGET_SIGNS)
        max_frames: Fixed sequence length
        load_all: If True, load ALL 2000 WLASL signs (ignores target_signs)

    Returns:
        X: (N, max_frames, 63) array of normalized landmarks
        y: (N,) array of sign indices
        sign_to_idx: Dictionary mapping sign names to indices
        idx_to_sign: Dictionary mapping indices to sign names
    """
    data_dir = Path(data_dir)

    # Load parsed data for gloss mapping
    # NPZ keys are indices into the parsed data array
    parsed_path = data_dir / 'WLASL_parsed_data.json'
    with open(parsed_path, 'r') as f:
        parsed_data = json.load(f)

    print(f"Total entries in parsed data: {len(parsed_data)}")

    # Get all available glosses from parsed data
    available_glosses = sorted(set(entry['gloss'].lower() for entry in parsed_data))
    print(f"Total unique signs in WLASL: {len(available_glosses)}")

    if load_all:
        # Load ALL WLASL signs
        valid_targets = available_glosses
        print(f"Loading ALL {len(valid_targets)} WLASL signs")
    else:
        # Filter to target signs only
        if target_signs is None:
            target_signs = TARGET_SIGNS

        valid_targets = [s.lower() for s in target_signs if s.lower() in available_glosses]
        missing = [s for s in target_signs if s.lower() not in available_glosses]

        if missing:
            print(f"Target signs not in WLASL: {missing}")
        print(f"Using {len(valid_targets)} target signs")

    # Create label mapping
    # For extended dataset, create new comprehensive label map
    signs = sorted(valid_targets)
    sign_to_idx = {s: i for i, s in enumerate(signs)}
    idx_to_sign = {i: s for s, i in sign_to_idx.items()}
    print(f"Created label map with {len(sign_to_idx)} signs")

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
            # NPZ key is an index into parsed_data
            try:
                idx = int(key)
                if idx >= len(parsed_data):
                    skipped += 1
                    continue

                # Get gloss for this index
                gloss = parsed_data[idx]['gloss'].lower()

                # Check if this is a valid sign
                if gloss not in sign_to_idx:
                    skipped += 1
                    continue

                # Load landmarks (frames, 180, 3)
                landmarks = data[key]

                if len(landmarks) == 0:
                    skipped += 1
                    continue

                # Extract hands from holistic
                hands = extract_hands_from_holistic(landmarks)

                # Get dominant hand
                hand = get_dominant_hand(hands)

                # Handle NaN
                hand = np.nan_to_num(hand, nan=0.0)

                # Normalize
                hand = normalize_landmarks(hand)

                # Window to fixed length
                hand = pad_or_truncate(hand, max_frames)

                # Flatten for model input: (30, 63)
                X_data.append(hand.reshape(max_frames, -1))
                y_data.append(sign_to_idx[gloss])
                processed += 1

            except Exception as e:
                skipped += 1
                continue

    print(f"\nProcessed: {processed}, Skipped: {skipped}")

    if len(X_data) == 0:
        raise ValueError("No valid samples found in WLASL data")

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)

    print(f"\nLoaded {len(X)} samples from WLASL")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Signs: {len(set(y))} unique")

    return X, y, sign_to_idx, idx_to_sign


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Load WLASL landmarks data')
    parser.add_argument('--data-dir', type=str,
                        default='data/raw/wlasl-landmarks',
                        help='Path to wlasl-landmarks directory')
    parser.add_argument('--output-dir', type=str,
                        default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--load-all', action='store_true',
                        help='Load all 2000 WLASL signs instead of just target signs')

    args = parser.parse_args()

    # Load data
    X, y, sign_to_idx, idx_to_sign = load_wlasl_data(
        args.data_dir,
        load_all=args.load_all
    )

    # Save WLASL-specific files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'X_wlasl.npy', X)
    np.save(output_dir / 'y_wlasl.npy', y)

    # Save WLASL label map
    with open(output_dir / 'wlasl_label_map.json', 'w') as f:
        json.dump(sign_to_idx, f, indent=2)

    print(f"\nSaved WLASL data to {output_dir}")
    print(f"  X_wlasl.npy: {X.shape}")
    print(f"  y_wlasl.npy: {y.shape}")
    print(f"  wlasl_label_map.json: {len(sign_to_idx)} signs")
