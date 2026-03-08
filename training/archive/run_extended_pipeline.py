"""
SignSense Extended Pipeline
Processes WLASL data and creates extended dataset combining MVP + WLASL.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from load_wlasl import load_wlasl_data
from create_extended_dataset import create_extended_dataset
from generate_errors import generate_error_augmented_dataset


def run_extended_pipeline(
    wlasl_dir='data/raw/wlasl-landmarks',
    processed_dir='data/processed',
    generate_errors=True
):
    """
    Run the full extended dataset pipeline.

    1. Process WLASL landmarks data
    2. Create extended dataset (MVP + WLASL)
    3. Optionally generate synthetic errors for extended dataset
    """
    wlasl_dir = Path(wlasl_dir)
    processed_dir = Path(processed_dir)
    extended_dir = processed_dir / 'extended'

    print("=" * 60)
    print("SIGNSENSE EXTENDED PIPELINE")
    print("=" * 60)

    # Step 1: Process WLASL data
    print("\n" + "=" * 60)
    print("STEP 1: Processing WLASL Data")
    print("=" * 60)

    try:
        X_wlasl, y_wlasl, sign_to_idx, idx_to_sign = load_wlasl_data(
            wlasl_dir,
            max_frames=30
        )

        # Save WLASL data
        import numpy as np
        np.save(processed_dir / 'X_wlasl.npy', X_wlasl)
        np.save(processed_dir / 'y_wlasl.npy', y_wlasl)
        print(f"\nSaved WLASL data: {X_wlasl.shape}")

    except Exception as e:
        print(f"Error processing WLASL: {e}")
        print("Continuing with MVP data only...")

    # Step 2: Create extended dataset
    print("\n" + "=" * 60)
    print("STEP 2: Creating Extended Dataset")
    print("=" * 60)

    extended_data = create_extended_dataset(
        processed_dir,
        extended_dir
    )

    # Step 3: Generate errors for extended dataset
    if generate_errors:
        print("\n" + "=" * 60)
        print("STEP 3: Generating Synthetic Errors for Extended Dataset")
        print("=" * 60)

        import numpy as np

        X_train = np.load(extended_dir / 'X_train.npy')
        y_train = np.load(extended_dir / 'y_train.npy')

        error_result = generate_error_augmented_dataset(
            X_train, y_train,
            error_rate=0.75,
            augmentations_per_sample=3
        )

        # Save error-augmented extended data
        np.save(extended_dir / 'X_train_with_errors.npy', error_result['X'])
        np.save(extended_dir / 'y_train_with_errors.npy', error_result['y'])
        np.save(extended_dir / 'error_labels_train.npy', error_result['error_labels'])
        np.save(extended_dir / 'is_correct_train.npy', error_result['is_correct'])
        np.save(extended_dir / 'component_scores_train.npy', error_result['component_scores'])

        import json
        with open(extended_dir / 'error_metadata.json', 'w') as f:
            json.dump(error_result['metadata'], f, indent=2)

        print(f"\nSaved error-augmented extended data:")
        print(f"  X_train_with_errors: {error_result['X'].shape}")
        print(f"  error_labels: {error_result['error_labels'].shape}")

    # Final summary
    print("\n" + "=" * 60)
    print("EXTENDED PIPELINE COMPLETE")
    print("=" * 60)

    print(f"\nExtended dataset saved to: {extended_dir}")
    print("\nFiles created:")
    for f in sorted(extended_dir.glob('*.npy')):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run extended dataset pipeline')
    parser.add_argument('--wlasl-dir', type=str,
                        default='data/raw/wlasl-landmarks',
                        help='Path to WLASL landmarks directory')
    parser.add_argument('--processed-dir', type=str,
                        default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--no-errors', action='store_true',
                        help='Skip error generation')

    args = parser.parse_args()

    run_extended_pipeline(
        wlasl_dir=args.wlasl_dir,
        processed_dir=args.processed_dir,
        generate_errors=not args.no_errors
    )
