"""
SignSense Data Processing Pipeline
Run the complete data processing pipeline.

Usage:
    python training/run_pipeline.py --data-dir data/raw/asl-signs

This will:
1. Load and preprocess Kaggle ASL-Signs data
2. Filter to 50 target signs
3. Normalize landmarks
4. Augment data (5x)
5. Generate synthetic errors (3x)
6. Create train/val/test splits
7. Save all processed data to data/processed/
"""

import argparse
import sys
from pathlib import Path

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent))

from load_data import load_kaggle_data, save_label_map, TARGET_SIGNS
from preprocess import augment_dataset, create_splits, save_splits
from generate_errors import augment_with_errors, save_error_metadata

import numpy as np


def run_pipeline(
    data_dir: str,
    output_dir: str,
    target_signs: list = None,
    max_samples: int = None,
    augment_factor: int = 5,
    errors_per_sample: int = 3,
    skip_augment: bool = False,
    skip_errors: bool = False
):
    """
    Run the complete data processing pipeline.

    Args:
        data_dir: Path to asl-signs dataset
        output_dir: Output directory for processed data
        target_signs: List of signs to include (None = default 50)
        max_samples: Max samples to load (None = all)
        augment_factor: Data augmentation factor
        errors_per_sample: Synthetic errors per sample
        skip_augment: Skip augmentation step
        skip_errors: Skip error generation step
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SIGNSENSE DATA PROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading Kaggle ASL-Signs data...")
    print("-" * 40)

    X, y, sign_to_idx, idx_to_sign = load_kaggle_data(
        data_dir,
        target_signs=target_signs or TARGET_SIGNS,
        max_samples=max_samples
    )

    # Save label map
    save_label_map(sign_to_idx, output_dir / 'label_map.json')

    # Save raw data
    np.save(output_dir / 'X_raw.npy', X)
    np.save(output_dir / 'y_raw.npy', y)
    print(f"Saved raw data: X={X.shape}, y={y.shape}")

    # Step 2: Augment (optional)
    if not skip_augment:
        print("\n[2/5] Augmenting data...")
        print("-" * 40)
        X, y = augment_dataset(X, y, augmentation_factor=augment_factor)
    else:
        print("\n[2/5] Skipping augmentation...")

    # Step 3: Create splits
    print("\n[3/5] Creating train/val/test splits...")
    print("-" * 40)
    splits = create_splits(X, y)
    save_splits(splits, output_dir)

    # Step 4: Generate errors (optional)
    if not skip_errors:
        print("\n[4/5] Generating synthetic errors...")
        print("-" * 40)

        X_train = splits['X_train']
        y_train = splits['y_train']

        X_with_errors, y_with_errors, is_correct, comp_scores, err_labels = augment_with_errors(
            X_train, y_train,
            errors_per_sample=errors_per_sample
        )

        # Save error-augmented training data
        np.save(output_dir / 'X_train_with_errors.npy', X_with_errors)
        np.save(output_dir / 'y_train_with_errors.npy', y_with_errors)
        np.save(output_dir / 'is_correct_train.npy', is_correct)
        np.save(output_dir / 'component_scores_train.npy', comp_scores)
        np.save(output_dir / 'error_labels_train.npy', err_labels)

        save_error_metadata(output_dir)

        print(f"\nError-augmented training set:")
        print(f"  Total: {len(X_with_errors)}")
        print(f"  Correct: {is_correct.sum()}")
        print(f"  Errors: {(~is_correct).sum()}")
    else:
        print("\n[4/5] Skipping error generation...")

    # Step 5: Summary
    print("\n[5/5] Pipeline complete!")
    print("-" * 40)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")

    for f in sorted(output_dir.glob('*.npy')):
        arr = np.load(f)
        print(f"  {f.name}: {arr.shape} ({arr.dtype})")

    for f in sorted(output_dir.glob('*.json')):
        print(f"  {f.name}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run SignSense data processing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python training/run_pipeline.py --data-dir data/raw/asl-signs

  # Quick test with 100 samples
  python training/run_pipeline.py --data-dir data/raw/asl-signs --max-samples 100

  # Skip augmentation for faster testing
  python training/run_pipeline.py --data-dir data/raw/asl-signs --skip-augment
        """
    )

    parser.add_argument('--data-dir', type=str,
                        default='data/raw/asl-signs',
                        help='Path to asl-signs directory')
    parser.add_argument('--output-dir', type=str,
                        default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to load (for testing)')
    parser.add_argument('--augment-factor', type=int, default=5,
                        help='Data augmentation factor (default: 5)')
    parser.add_argument('--errors-per-sample', type=int, default=3,
                        help='Synthetic errors per sample (default: 3)')
    parser.add_argument('--skip-augment', action='store_true',
                        help='Skip data augmentation')
    parser.add_argument('--skip-errors', action='store_true',
                        help='Skip synthetic error generation')

    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        augment_factor=args.augment_factor,
        errors_per_sample=args.errors_per_sample,
        skip_augment=args.skip_augment,
        skip_errors=args.skip_errors
    )
