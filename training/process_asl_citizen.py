"""
ASL Citizen Video Processing Script
Extracts MediaPipe hand landmarks from ASL Citizen videos.

Requirements:
    pip install mediapipe opencv-python tqdm

Usage:
    python process_asl_citizen.py --video-dir /path/to/videos --output-dir data/processed
"""

import os
# Suppress TensorFlow/MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import urllib.request

# Suppress absl logging
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# MediaPipe Tasks API (0.10+)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Model paths
MODEL_DIR = Path(__file__).parent.parent / "models" / "mediapipe"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def download_model():
    """Download MediaPipe hand landmarker model if not present."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not HAND_MODEL_PATH.exists():
        print(f"Downloading hand landmarker model...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print(f"Model saved to {HAND_MODEL_PATH}")

    return str(HAND_MODEL_PATH)


def extract_landmarks_from_video(video_path, max_frames=None):
    """
    Extract hand landmarks from a video using MediaPipe Hand Landmarker.

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to process (None = all)

    Returns:
        landmarks: (n_frames, 42, 3) array of hand landmarks
                   42 = 21 left hand + 21 right hand
    """
    model_path = download_model()

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    landmarks_list = []
    frame_count = 0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_count >= max_frames:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect hands
            result = landmarker.detect(mp_image)

            # Extract hand landmarks (42 total: 21 left + 21 right)
            frame_landmarks = np.zeros((42, 3), dtype=np.float32)

            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                # Determine if left or right hand
                handedness = result.handedness[hand_idx][0].category_name

                offset = 0 if handedness == "Left" else 21

                for i, lm in enumerate(hand_landmarks):
                    frame_landmarks[offset + i] = [lm.x, lm.y, lm.z]

            landmarks_list.append(frame_landmarks)
            frame_count += 1

    cap.release()

    if len(landmarks_list) == 0:
        return None

    return np.array(landmarks_list, dtype=np.float32)


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

    left_motion = np.nanstd(left_hand)
    right_motion = np.nanstd(right_hand)

    left_present = np.sum(~np.isnan(left_hand) & (left_hand != 0))
    right_present = np.sum(~np.isnan(right_hand) & (right_hand != 0))

    left_score = left_motion * left_present
    right_score = right_motion * right_present

    return left_hand if left_score > right_score else right_hand


def process_single_video(args):
    """
    Worker function to process a single video (for multiprocessing).

    Args:
        args: Tuple of (video_path, gloss, label_idx, max_frames)

    Returns:
        Tuple of (features, label_idx) or None if failed
    """
    video_path, gloss, label_idx, max_frames = args

    try:
        landmarks = extract_landmarks_from_video(video_path)

        if landmarks is not None and len(landmarks) > 0:
            hand = get_dominant_hand(landmarks)
            hand = np.nan_to_num(hand, nan=0.0)
            hand = normalize_landmarks(hand)
            hand = pad_or_truncate(hand, max_frames)
            features = hand.reshape(max_frames, -1)
            return (features, label_idx)
    except Exception:
        pass

    return None


def find_videos_in_directory(video_dir):
    """
    Find all video files and try to extract gloss from filename/path.

    ASL Citizen naming convention: {gloss}_{signer_id}_{video_id}.mp4
    or organized in folders by gloss
    """
    video_dir = Path(video_dir)
    videos = []

    # Find all video files
    for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']:
        for video_path in video_dir.rglob(ext):
            # Try to extract gloss from path
            # Option 1: Folder structure /gloss/video.mp4
            if video_path.parent.name != video_dir.name:
                gloss = video_path.parent.name.lower()
            # Option 2: Filename {gloss}_{id}.mp4
            else:
                parts = video_path.stem.split('_')
                gloss = parts[0].lower() if parts else video_path.stem.lower()

            videos.append({
                'video_path': str(video_path),
                'video_id': video_path.stem,
                'gloss': gloss
            })

    return videos


def load_metadata(metadata_path):
    """Load metadata from JSON or CSV file."""
    metadata_path = Path(metadata_path)

    if metadata_path.suffix == '.json':
        with open(metadata_path, 'r') as f:
            return json.load(f)
    elif metadata_path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(metadata_path)
        return df.to_dict('records')
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")


def load_asl_citizen_splits(splits_dir):
    """
    Load ASL Citizen official train/val/test splits.

    Expected files: train.csv, val.csv, test.csv
    CSV format: Participant ID, Video file, Gloss, ASL-LEX Code
    """
    import csv
    splits_dir = Path(splits_dir)

    all_videos = []

    for split in ['train', 'val', 'test']:
        csv_path = splits_dir / f'{split}.csv'
        if csv_path.exists():
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_videos.append({
                        'video_file': row['Video file'],
                        'gloss': row['Gloss'].lower().strip(),
                        'participant': row['Participant ID'],
                        'split': split
                    })

    return all_videos


def process_asl_citizen(
    video_dir,
    output_dir,
    metadata_path=None,
    splits_dir=None,
    max_frames=30,
    num_workers=4,
    batch_size=100
):
    """
    Process ASL Citizen videos and extract landmarks.

    Args:
        video_dir: Directory containing videos (ASL_Citizen/videos/)
        output_dir: Output directory for processed data
        metadata_path: Path to metadata file (optional)
        splits_dir: Path to ASL Citizen splits directory (ASL_Citizen/splits/)
        max_frames: Target sequence length
        num_workers: Number of parallel workers for multiprocessing
        batch_size: Save intermediate results every N videos
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ASL CITIZEN VIDEO PROCESSING")
    print("=" * 60)
    print(f"Parallel workers: {num_workers}")

    # Download model first
    print("\n0. Checking MediaPipe model...")
    download_model()

    # Find videos
    print(f"\n1. Scanning for videos in {video_dir}...")

    # Try to auto-detect ASL Citizen structure
    if splits_dir is None:
        # Check if video_dir is ASL_Citizen root or videos subfolder
        if (video_dir.parent / 'splits').exists():
            splits_dir = video_dir.parent / 'splits'
        elif (video_dir / 'splits').exists():
            splits_dir = video_dir / 'splits'
            video_dir = video_dir / 'videos'

    if splits_dir and Path(splits_dir).exists():
        # Use official ASL Citizen splits
        print(f"   Using ASL Citizen splits from {splits_dir}")
        video_metadata = load_asl_citizen_splits(splits_dir)
        videos = []
        for entry in video_metadata:
            video_path = video_dir / entry['video_file']
            if video_path.exists():
                videos.append({
                    'video_path': str(video_path),
                    'video_id': video_path.stem,
                    'gloss': entry['gloss'],
                    'split': entry['split']
                })
    elif metadata_path:
        metadata = load_metadata(metadata_path)
        videos = []
        for entry in metadata:
            video_id = entry.get('video_id', entry.get('filename', ''))
            gloss = entry.get('gloss', entry.get('label', 'unknown'))

            # Find video file
            for ext in ['.mp4', '.avi', '.mov', '.MP4']:
                video_path = video_dir / f"{video_id}{ext}"
                if video_path.exists():
                    videos.append({
                        'video_path': str(video_path),
                        'video_id': video_id,
                        'gloss': gloss.lower()
                    })
                    break
    else:
        videos = find_videos_in_directory(video_dir)

    print(f"   Found {len(videos)} videos")

    if len(videos) == 0:
        print("   No videos found!")
        return

    # Get unique glosses
    glosses = sorted(set(v['gloss'] for v in videos))
    print(f"   Unique signs: {len(glosses)}")

    # Create label map
    sign_to_idx = {g: i for i, g in enumerate(glosses)}

    # Save label map early (needed for resume)
    with open(output_dir / 'asl_citizen_label_map.json', 'w') as f:
        json.dump(sign_to_idx, f, indent=2)

    # Check for checkpoint to resume from
    checkpoint_X = output_dir / 'X_asl_citizen_temp.npy'
    checkpoint_y = output_dir / 'y_asl_citizen_temp.npy'
    checkpoint_processed = output_dir / 'processed_videos.json'

    X_data = []
    y_data = []
    processed_paths = set()

    if checkpoint_X.exists() and checkpoint_y.exists() and checkpoint_processed.exists():
        print(f"\n   Found checkpoint, resuming...")
        X_data = list(np.load(checkpoint_X))
        y_data = list(np.load(checkpoint_y))
        with open(checkpoint_processed, 'r') as f:
            processed_paths = set(json.load(f))
        print(f"   Loaded {len(X_data)} samples from checkpoint")
        print(f"   Skipping {len(processed_paths)} already processed videos")

    # Filter out already processed videos
    remaining_videos = [v for v in videos if v['video_path'] not in processed_paths]

    # Process videos with multiprocessing
    print(f"\n2. Processing videos with {num_workers} workers...")
    print(f"   Remaining: {len(remaining_videos)} / {len(videos)} videos")

    # Prepare work items
    work_items = [
        (v['video_path'], v['gloss'], sign_to_idx[v['gloss']], max_frames)
        for v in remaining_videos
    ]

    processed = len(X_data)
    failed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks and process results as they complete
        futures = {executor.submit(process_single_video, item): item for item in work_items}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", initial=0):
            item = futures[future]
            video_path = item[0]
            result = future.result()

            processed_paths.add(video_path)

            if result is not None:
                features, label_idx = result
                X_data.append(features)
                y_data.append(label_idx)
                processed += 1
            else:
                failed += 1

            # Save intermediate results
            if processed > 0 and processed % batch_size == 0:
                print(f"\n   Checkpoint: {processed} processed, {failed} failed")
                X_temp = np.array(X_data, dtype=np.float32)
                y_temp = np.array(y_data, dtype=np.int32)
                np.save(output_dir / 'X_asl_citizen_temp.npy', X_temp)
                np.save(output_dir / 'y_asl_citizen_temp.npy', y_temp)
                with open(output_dir / 'processed_videos.json', 'w') as f:
                    json.dump(list(processed_paths), f)

    print(f"\n   Processed: {processed}, Failed: {failed}")

    if processed == 0:
        print("   No videos successfully processed!")
        return

    # Convert to arrays
    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)

    print(f"\n3. Saving results...")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Unique labels: {len(set(y))}")

    # Save
    np.save(output_dir / 'X_asl_citizen.npy', X)
    np.save(output_dir / 'y_asl_citizen.npy', y)

    with open(output_dir / 'asl_citizen_label_map.json', 'w') as f:
        json.dump(sign_to_idx, f, indent=2)

    # Remove temp files
    temp_files = [
        output_dir / 'X_asl_citizen_temp.npy',
        output_dir / 'y_asl_citizen_temp.npy',
        output_dir / 'processed_videos.json'
    ]
    for f in temp_files:
        if f.exists():
            f.unlink()

    # Save processing stats
    stats = {
        'total_videos': len(videos),
        'processed': processed,
        'failed': failed,
        'unique_signs': len(glosses),
        'X_shape': list(X.shape),
        'y_shape': list(y.shape)
    }
    with open(output_dir / 'asl_citizen_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n   Saved to {output_dir}")

    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Videos processed: {processed}/{len(videos)}")
    print(f"  Signs: {len(glosses)}")
    print(f"  Output: {output_dir}")

    return X, y, sign_to_idx


if __name__ == "__main__":
    import multiprocessing
    parser = argparse.ArgumentParser(description='Process ASL Citizen videos')
    parser.add_argument('--video-dir', type=str, required=True,
                        help='Directory containing ASL Citizen videos (or ASL_Citizen root)')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--splits-dir', type=str, default=None,
                        help='Path to splits directory (auto-detected if not specified)')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to metadata file (JSON or CSV)')
    parser.add_argument('--max-frames', type=int, default=30,
                        help='Target sequence length')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Checkpoint every N videos')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                        help='Number of parallel workers (default: all CPU cores)')

    args = parser.parse_args()

    process_asl_citizen(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        splits_dir=args.splits_dir,
        metadata_path=args.metadata,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
