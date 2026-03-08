"""
Process Fingerspelling Datasets and Merge into Main Dataset
============================================================

This script:
1. Processes ASL Alphabet (static images, 87K samples, A-Z)
2. Processes ASL MNIST (28x28 grayscale, 35K samples, A-Z)
3. Downloads and processes ChicagoFSWild (video sequences, 7K samples)
4. Merges all into the main dataset
5. Reshuffles with 80/10/10 train/val/test splits

Usage:
    python training/process_fingerspelling.py
    python training/process_fingerspelling.py --skip-download  # Skip ChicagoFSWild download
    python training/process_fingerspelling.py --alphabet-only  # Only process alphabet
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import pandas as pd
import json
import subprocess
import tarfile
from pathlib import Path
from tqdm import tqdm
import urllib.request
import argparse

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "mediapipe"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# ChicagoFSWild info
CHICAGOFSWILD_URL = "https://dl.ttic.edu/ChicagoFSWild.tgz"
CHICAGOFSWILD_DIR = DATA_RAW / "chicagofswild"


def download_model():
    """Download MediaPipe hand landmarker model if not present."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not HAND_MODEL_PATH.exists():
        print(f"Downloading hand landmarker model...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    return str(HAND_MODEL_PATH)


def create_hand_landmarker():
    """Create MediaPipe hand landmarker."""
    model_path = download_model()
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks_from_image(image, landmarker):
    """Extract hand landmarks from a single image."""
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
    return landmarks


def normalize_landmarks(landmarks):
    """Normalize landmarks to be position/scale invariant."""
    if landmarks is None or np.all(landmarks == 0):
        return None

    wrist = landmarks[0]
    centered = landmarks - wrist

    distances = np.linalg.norm(centered, axis=1)
    scale = distances.max()

    if scale > 1e-6:
        centered = centered / scale

    return centered


def process_asl_alphabet(landmarker, max_frames=30):
    """Process ASL Alphabet dataset."""
    print("\n" + "=" * 60)
    print("PROCESSING ASL ALPHABET")
    print("=" * 60)

    train_dir = DATA_RAW / "asl-alphabet" / "asl_alphabet_train" / "asl_alphabet_train"
    if not train_dir.exists():
        train_dir = DATA_RAW / "asl-alphabet" / "asl_alphabet_train"

    if not train_dir.exists():
        print(f"ERROR: ASL Alphabet not found at {train_dir}")
        return None, None, None

    letters = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"Found {len(letters)} classes")

    # Filter to A-Z only (exclude del, nothing, space for consistency)
    alphabet_letters = [l for l in letters if len(l) == 1 and l.isalpha()]
    label_map = {letter.lower(): ord(letter.lower()) - ord('a') for letter in alphabet_letters}

    X_data, y_data = [], []
    failed = 0

    for letter in tqdm(alphabet_letters, desc="Processing letters"):
        letter_dir = train_dir / letter
        images = list(letter_dir.glob("*.jpg")) + list(letter_dir.glob("*.png"))

        for img_path in images:
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    failed += 1
                    continue

                landmarks = extract_landmarks_from_image(image, landmarker)
                if landmarks is None:
                    failed += 1
                    continue

                landmarks = normalize_landmarks(landmarks)
                if landmarks is None:
                    failed += 1
                    continue

                features = landmarks.flatten()
                sequence = np.tile(features, (max_frames, 1))

                X_data.append(sequence)
                y_data.append(label_map[letter.lower()])

            except Exception:
                failed += 1

    if len(X_data) == 0:
        return None, None, None

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)

    print(f"Processed: {len(X)}, Failed: {failed}")
    print(f"Shape: {X.shape}")

    return X, y, label_map


def process_asl_mnist(landmarker, max_frames=30):
    """Process ASL MNIST dataset."""
    print("\n" + "=" * 60)
    print("PROCESSING ASL MNIST")
    print("=" * 60)

    train_csv = DATA_RAW / 'asl-mnist' / 'sign_mnist_train.csv'
    test_csv = DATA_RAW / 'asl-mnist' / 'sign_mnist_test.csv'

    if not train_csv.exists():
        print(f"ERROR: ASL MNIST not found at {train_csv}")
        return None, None, None

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    df = pd.concat([train_df, test_df], ignore_index=True)

    print(f"Total samples: {len(df)}")

    # Labels 0-25 for A-Z (J=9 and Z=25 excluded in original)
    label_map = {chr(65 + i).lower(): i for i in range(26)}

    X_data, y_data = [], []
    failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        try:
            pixels = row.drop('label').values.astype(np.uint8)
            image = pixels.reshape(28, 28)
            image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_CUBIC)

            landmarks = extract_landmarks_from_image(image, landmarker)
            if landmarks is None:
                failed += 1
                continue

            landmarks = normalize_landmarks(landmarks)
            if landmarks is None:
                failed += 1
                continue

            features = landmarks.flatten()
            sequence = np.tile(features, (max_frames, 1))

            X_data.append(sequence)
            y_data.append(row['label'])

        except Exception:
            failed += 1

    if len(X_data) == 0:
        print("WARNING: No ASL MNIST samples processed (images too small for MediaPipe)")
        return None, None, None

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)

    print(f"Processed: {len(X)}, Failed: {failed}")
    print(f"Shape: {X.shape}")

    return X, y, label_map


def download_chicagofswild():
    """Download ChicagoFSWild dataset."""
    print("\n" + "=" * 60)
    print("DOWNLOADING CHICAGOFSWILD (14 GB)")
    print("=" * 60)

    archive_path = DATA_RAW / "chicagofswild.tgz"

    if archive_path.exists():
        print(f"Archive already exists: {archive_path}")
    else:
        print(f"Downloading from {CHICAGOFSWILD_URL}...")
        try:
            # Use curl for better progress
            subprocess.run(
                ['curl', '-L', '-C', '-', '-o', str(archive_path), CHICAGOFSWILD_URL],
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to Python download
            urllib.request.urlretrieve(CHICAGOFSWILD_URL, archive_path)

    # Extract
    if not CHICAGOFSWILD_DIR.exists():
        print(f"Extracting to {CHICAGOFSWILD_DIR}...")
        CHICAGOFSWILD_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(CHICAGOFSWILD_DIR)
    else:
        print(f"Already extracted: {CHICAGOFSWILD_DIR}")

    return True


def process_single_chicago_sequence(args):
    """Process a single ChicagoFSWild sequence (for multiprocessing)."""
    seq_dir, label_idx, max_frames = args

    try:
        frame_files = sorted(Path(seq_dir).glob("*.jpg"))
        if not frame_files:
            return None

        # Create landmarker for this process
        model_path = download_model()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5
        )

        with vision.HandLandmarker.create_from_options(options) as landmarker:
            all_landmarks = []
            for frame_path in frame_files[:max_frames * 2]:
                image = cv2.imread(str(frame_path))
                if image is None:
                    continue

                landmarks = extract_landmarks_from_image(image, landmarker)
                if landmarks is not None:
                    landmarks = normalize_landmarks(landmarks)
                    if landmarks is not None:
                        all_landmarks.append(landmarks.flatten())

            if len(all_landmarks) < 5:
                return None

            all_landmarks = np.array(all_landmarks)
            if len(all_landmarks) < max_frames:
                padding = np.tile(all_landmarks[-1:], (max_frames - len(all_landmarks), 1))
                all_landmarks = np.concatenate([all_landmarks, padding])
            else:
                indices = np.linspace(0, len(all_landmarks) - 1, max_frames).astype(int)
                all_landmarks = all_landmarks[indices]

            return (all_landmarks, label_idx)

    except Exception:
        return None


def process_chicagofswild(landmarker, max_frames=30, max_samples=None, num_workers=None):
    """Process ChicagoFSWild dataset with multiprocessing."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    print("\n" + "=" * 60)
    print("PROCESSING CHICAGOFSWILD")
    print(f"Using {num_workers} CPU cores")
    print("=" * 60)

    # Find the actual data directory (may be nested)
    data_dir = CHICAGOFSWILD_DIR
    if (data_dir / "ChicagoFSWild").exists():
        data_dir = data_dir / "ChicagoFSWild"

    # Find CSV
    csv_files = list(data_dir.rglob("*.csv"))
    main_csv = None
    for csv_file in csv_files:
        if 'Hand' not in csv_file.name:
            main_csv = csv_file
            break

    if main_csv is None:
        print(f"ERROR: No CSV found in {data_dir}")
        return None, None, None

    print(f"Using CSV: {main_csv}")

    # Parse CSV
    df = pd.read_csv(main_csv)
    print(f"Found {len(df)} sequences")

    if max_samples:
        df = df.head(max_samples)

    # Build directory index ONCE (much faster than rglob per sequence)
    print("Building directory index...")
    seq_to_dir = {}
    for subdir in data_dir.iterdir():
        if subdir.is_dir() and subdir.name not in ['BBox']:
            for seq_dir in subdir.iterdir():
                if seq_dir.is_dir():
                    seq_to_dir[seq_dir.name] = seq_dir
    print(f"Indexed {len(seq_to_dir)} sequence directories")

    # Create label map - check various column names
    label_col = None
    for col in ['label', 'label_proc', 'Label', 'processed_label', 'text']:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        print(f"ERROR: No label column found. Columns: {df.columns.tolist()}")
        return None, None, None

    print(f"Using label column: {label_col}")
    all_labels = sorted(df[label_col].dropna().unique())
    label_map = {str(label).lower(): i for i, label in enumerate(all_labels)}
    print(f"Found {len(label_map)} unique labels")

    # Prepare work items
    work_items = []
    not_found = 0
    for idx, row in df.iterrows():
        filename = row.get('filename', row.get('Filename', ''))
        if not filename:
            continue

        # Handle paths like "aslized/elsie_stecker_0001"
        seq_id = Path(filename).name  # Get last part of path
        if seq_id not in seq_to_dir:
            not_found += 1
            continue

        label = str(row.get(label_col, '')).lower()
        if label and label in label_map:
            work_items.append((str(seq_to_dir[seq_id]), label_map[label], max_frames))

    print(f"Sequences not found in index: {not_found}")

    print(f"Processing {len(work_items)} sequences with {num_workers} workers...")

    X_data, y_data = [], []
    failed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_chicago_sequence, item): item for item in work_items}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result is not None:
                X_data.append(result[0])
                y_data.append(result[1])
            else:
                failed += 1

    if len(X_data) == 0:
        return None, None, None

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)

    print(f"Processed: {len(X)}, Failed: {failed}")
    print(f"Shape: {X.shape}")

    return X, y, label_map


def merge_and_reshuffle(datasets, output_dir):
    """
    Merge all datasets and reshuffle with 80/10/10 splits.

    Args:
        datasets: List of (X, y, label_map, name) tuples
        output_dir: Output directory
    """
    print("\n" + "=" * 60)
    print("MERGING AND RESHUFFLING")
    print("=" * 60)

    output_dir = Path(output_dir)

    # Load existing merged dataset
    merged_dir = output_dir / "merged"

    X_existing, y_existing = None, None
    existing_label_map = {}

    if (merged_dir / "X_train.npy").exists():
        print("Loading existing merged dataset...")
        X_train = np.load(merged_dir / "X_train.npy")
        X_val = np.load(merged_dir / "X_val.npy")
        X_test = np.load(merged_dir / "X_test.npy")
        y_train = np.load(merged_dir / "y_train.npy")
        y_val = np.load(merged_dir / "y_val.npy")
        y_test = np.load(merged_dir / "y_test.npy")

        X_existing = np.concatenate([X_train, X_val, X_test])
        y_existing = np.concatenate([y_train, y_val, y_test])

        with open(merged_dir / "label_map.json") as f:
            existing_label_map = json.load(f)

        print(f"Existing: {len(X_existing)} samples, {len(existing_label_map)} signs")

    # Combine all new datasets
    all_X, all_y = [], []
    all_labels = set(existing_label_map.keys())

    # Add existing data
    if X_existing is not None:
        all_X.append(X_existing)
        all_y.append(y_existing)

    # Add new datasets with label remapping
    for X, y, label_map, name in datasets:
        if X is None:
            continue

        print(f"Adding {name}: {len(X)} samples")

        # Create new labels for fingerspelling (prefix with 'fs_')
        new_label_map = {}
        for label, idx in label_map.items():
            new_label = f"fs_{label}"
            if new_label not in all_labels:
                new_idx = len(all_labels) + len(existing_label_map)
                new_label_map[new_label] = new_idx
                all_labels.add(new_label)
            else:
                # Find existing index
                for k, v in existing_label_map.items():
                    if k == new_label:
                        new_label_map[new_label] = v
                        break

        # Remap y values
        y_remapped = np.array([new_label_map.get(f"fs_{list(label_map.keys())[list(label_map.values()).index(yi)]}", -1)
                               for yi in y])

        # Filter out invalid labels
        valid_mask = y_remapped >= 0
        X_valid = X[valid_mask]
        y_valid = y_remapped[valid_mask]

        if len(X_valid) > 0:
            all_X.append(X_valid)
            all_y.append(y_valid)

    if not all_X:
        print("ERROR: No data to merge!")
        return

    # Concatenate all
    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)

    print(f"\nTotal samples: {len(X_all)}")

    # Shuffle
    print("Shuffling...")
    indices = np.random.permutation(len(X_all))
    X_all = X_all[indices]
    y_all = y_all[indices]

    # Split 80/10/10
    n_total = len(X_all)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_val = X_all[n_train:n_train + n_val]
    y_val = y_all[n_train:n_train + n_val]
    X_test = X_all[n_train + n_val:]
    y_test = y_all[n_train + n_val:]

    print(f"Train: {len(X_train)}")
    print(f"Val: {len(X_val)}")
    print(f"Test: {len(X_test)}")

    # Create unified label map
    unified_label_map = dict(existing_label_map)
    for X, y, label_map, name in datasets:
        if label_map is None:
            continue
        for label in label_map.keys():
            new_label = f"fs_{label}"
            if new_label not in unified_label_map:
                unified_label_map[new_label] = len(unified_label_map)

    print(f"Total vocabulary: {len(unified_label_map)} signs")

    # Save
    merged_dir.mkdir(parents=True, exist_ok=True)

    np.save(merged_dir / "X_train.npy", X_train)
    np.save(merged_dir / "y_train.npy", y_train)
    np.save(merged_dir / "X_val.npy", X_val)
    np.save(merged_dir / "y_val.npy", y_val)
    np.save(merged_dir / "X_test.npy", X_test)
    np.save(merged_dir / "y_test.npy", y_test)

    with open(merged_dir / "label_map.json", 'w') as f:
        json.dump(unified_label_map, f, indent=2)

    # Save stats
    stats = {
        'total_samples': int(n_total),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'test_samples': int(len(X_test)),
        'vocabulary_size': len(unified_label_map),
        'shape': list(X_train.shape)
    }
    with open(merged_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to {merged_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples: {n_total:,}")
    print(f"Train: {len(X_train):,} (80%)")
    print(f"Val: {len(X_val):,} (10%)")
    print(f"Test: {len(X_test):,} (10%)")
    print(f"Vocabulary: {len(unified_label_map):,} signs")
    print(f"Shape: {X_train.shape}")
    print(f"Size: {(X_train.nbytes + X_val.nbytes + X_test.nbytes) / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Process fingerspelling datasets and merge')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip ChicagoFSWild download')
    parser.add_argument('--alphabet-only', action='store_true',
                        help='Only process ASL Alphabet')
    parser.add_argument('--mnist-only', action='store_true',
                        help='Only process ASL MNIST')
    parser.add_argument('--chicago-only', action='store_true',
                        help='Only download/process ChicagoFSWild')
    parser.add_argument('--max-chicago-samples', type=int, default=None,
                        help='Max ChicagoFSWild samples to process')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Output directory')

    args = parser.parse_args()

    # Initialize MediaPipe
    print("Initializing MediaPipe hand landmarker...")
    landmarker = create_hand_landmarker()

    datasets = []

    # Process ASL Alphabet
    if not args.mnist_only and not args.chicago_only:
        X, y, label_map = process_asl_alphabet(landmarker)
        if X is not None:
            datasets.append((X, y, label_map, "ASL Alphabet"))
            np.save(DATA_PROCESSED / "X_asl_alphabet.npy", X)
            np.save(DATA_PROCESSED / "y_asl_alphabet.npy", y)
            with open(DATA_PROCESSED / "asl_alphabet_label_map.json", 'w') as f:
                json.dump(label_map, f, indent=2)

    # Process ASL MNIST
    if not args.alphabet_only and not args.chicago_only:
        X, y, label_map = process_asl_mnist(landmarker)
        if X is not None:
            datasets.append((X, y, label_map, "ASL MNIST"))
            np.save(DATA_PROCESSED / "X_asl_mnist.npy", X)
            np.save(DATA_PROCESSED / "y_asl_mnist.npy", y)
            with open(DATA_PROCESSED / "asl_mnist_label_map.json", 'w') as f:
                json.dump(label_map, f, indent=2)

    # Download and process ChicagoFSWild
    if not args.alphabet_only and not args.mnist_only:
        if not args.skip_download:
            download_chicagofswild()

        if CHICAGOFSWILD_DIR.exists():
            X, y, label_map = process_chicagofswild(
                landmarker,
                max_samples=args.max_chicago_samples
            )
            if X is not None:
                datasets.append((X, y, label_map, "ChicagoFSWild"))
                np.save(DATA_PROCESSED / "X_chicagofswild.npy", X)
                np.save(DATA_PROCESSED / "y_chicagofswild.npy", y)
                with open(DATA_PROCESSED / "chicagofswild_label_map.json", 'w') as f:
                    json.dump(label_map, f, indent=2)

    # Merge and reshuffle
    if datasets:
        merge_and_reshuffle(datasets, args.output_dir)

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
