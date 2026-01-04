"""
SignSense Data Preprocessing Module
Augmentation and train/val/test splitting.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import json


def augment_landmarks(landmarks, seed=None):
    """
    Apply random augmentations to landmark sequence.

    Args:
        landmarks: (30, 63) flattened landmarks
        seed: Random seed for reproducibility

    Returns:
        Augmented landmarks (30, 63)
    """
    if seed is not None:
        np.random.seed(seed)

    # Reshape to (30, 21, 3)
    augmented = landmarks.copy().reshape(-1, 21, 3)

    # Random rotation around Z-axis (±15 degrees)
    angle = np.random.uniform(-15, 15) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    augmented = np.dot(augmented, rotation.T)

    # Random scaling (0.9-1.1x)
    scale = np.random.uniform(0.9, 1.1)
    augmented = augmented * scale

    # Random translation (small shift)
    shift = np.random.uniform(-0.05, 0.05, size=3)
    augmented = augmented + shift

    # Gaussian noise
    noise = np.random.normal(0, 0.01, augmented.shape)
    augmented = augmented + noise

    # Temporal jitter (small frame shifts)
    if np.random.random() > 0.5:
        shift_frames = np.random.randint(-2, 3)
        augmented = np.roll(augmented, shift_frames, axis=0)

    return augmented.reshape(-1, 63).astype(np.float32)


def augment_dataset(X, y, augmentation_factor=5, verbose=True):
    """
    Augment entire dataset.

    Args:
        X: (N, 30, 63) landmark sequences
        y: (N,) labels
        augmentation_factor: How many total copies (including original)
        verbose: Print progress

    Returns:
        X_aug: (N * factor, 30, 63)
        y_aug: (N * factor,)
    """
    if verbose:
        print(f"Augmenting {len(X)} samples with factor {augmentation_factor}...")

    X_aug = [X]
    y_aug = [y]

    for i in range(augmentation_factor - 1):
        if verbose:
            print(f"  Augmentation round {i + 2}/{augmentation_factor}")

        X_new = np.array([
            augment_landmarks(x, seed=i * 100000 + j)
            for j, x in enumerate(X)
        ])
        X_aug.append(X_new)
        y_aug.append(y)

    X_result = np.vstack(X_aug)
    y_result = np.hstack(y_aug)

    if verbose:
        print(f"Augmented: {len(X)} -> {len(X_result)} samples")

    return X_result, y_result


def create_splits(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Create stratified train/val/test splits.

    Args:
        X: Feature array
        y: Label array
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed

    Returns:
        Dictionary with train/val/test arrays
    """
    # Check if stratification is possible
    from collections import Counter
    class_counts = Counter(y)
    min_count = min(class_counts.values())
    use_stratify = min_count >= 3  # Need at least 3 samples per class for stratified split

    if not use_stratify:
        print(f"Warning: Some classes have <3 samples. Using non-stratified split.")

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y if use_stratify else None,
        random_state=random_state
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)

    # Re-check stratification for second split
    class_counts_trainval = Counter(y_trainval)
    min_count_trainval = min(class_counts_trainval.values())
    use_stratify_trainval = min_count_trainval >= 2

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        stratify=y_trainval if use_stratify_trainval else None,
        random_state=random_state
    )

    print(f"Split sizes:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


def save_splits(splits, output_dir):
    """Save splits to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in splits.items():
        path = output_dir / f'{name}.npy'
        np.save(path, data)
        print(f"Saved {path}: {data.shape}")


def compute_class_weights(y):
    """Compute class weights for imbalanced data."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {c: total / (len(classes) * count) for c, count in zip(classes, counts)}
    return weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess and augment data')
    parser.add_argument('--input-dir', type=str,
                        default='data/processed',
                        help='Directory with X_raw.npy and y_raw.npy')
    parser.add_argument('--output-dir', type=str,
                        default='data/processed',
                        help='Output directory')
    parser.add_argument('--augment-factor', type=int, default=5,
                        help='Augmentation factor')
    parser.add_argument('--no-augment', action='store_true',
                        help='Skip augmentation')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Load raw data
    print("Loading raw data...")
    X = np.load(input_dir / 'X_raw.npy')
    y = np.load(input_dir / 'y_raw.npy')
    print(f"Loaded: X={X.shape}, y={y.shape}")

    # Augment if requested
    if not args.no_augment:
        X, y = augment_dataset(X, y, augmentation_factor=args.augment_factor)

    # Create splits
    print("\nCreating train/val/test splits...")
    splits = create_splits(X, y)

    # Save
    print("\nSaving splits...")
    save_splits(splits, output_dir)

    # Compute and save class weights
    weights = compute_class_weights(splits['y_train'])
    with open(output_dir / 'class_weights.json', 'w') as f:
        json.dump({int(k): v for k, v in weights.items()}, f, indent=2)
    print(f"Saved class weights to {output_dir / 'class_weights.json'}")

    print("\nDone!")
