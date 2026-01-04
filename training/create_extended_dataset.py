"""
SignSense Extended Dataset Creator
Combines MVP (Kaggle) and WLASL datasets into an extended training set.
Supports full WLASL vocabulary (2000 signs) + MVP target signs.
"""

import numpy as np
import json
from pathlib import Path


def remap_labels(y, old_label_map, new_label_map):
    """
    Remap labels from old label map to new unified label map.

    Args:
        y: Array of labels using old indices
        old_label_map: Dict mapping sign names to old indices
        new_label_map: Dict mapping sign names to new indices

    Returns:
        Array of labels using new indices
    """
    # Create old_idx -> sign -> new_idx mapping
    old_idx_to_sign = {v: k for k, v in old_label_map.items()}

    y_new = np.zeros_like(y)
    for i, old_idx in enumerate(y):
        sign = old_idx_to_sign[old_idx]
        y_new[i] = new_label_map[sign]

    return y_new


def create_extended_dataset(processed_dir, output_dir=None):
    """
    Create extended dataset by combining MVP and WLASL data.

    Args:
        processed_dir: Directory containing processed MVP data and WLASL data
        output_dir: Directory for extended dataset (default: processed_dir/extended)
    """
    processed_dir = Path(processed_dir)

    if output_dir is None:
        output_dir = processed_dir / 'extended'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating Extended Dataset")
    print("=" * 60)

    # Load MVP data
    print("\n1. Loading MVP data...")
    mvp_data = {}
    for split in ['train', 'val', 'test']:
        X_path = processed_dir / f'X_{split}.npy'
        y_path = processed_dir / f'y_{split}.npy'
        if X_path.exists() and y_path.exists():
            mvp_data[f'X_{split}'] = np.load(X_path)
            mvp_data[f'y_{split}'] = np.load(y_path)
            print(f"  {split}: X={mvp_data[f'X_{split}'].shape}, y={mvp_data[f'y_{split}'].shape}")

    # Load MVP label map
    mvp_label_path = processed_dir / 'label_map.json'
    with open(mvp_label_path, 'r') as f:
        mvp_label_map = json.load(f)
    print(f"  MVP vocabulary: {len(mvp_label_map)} signs")

    # Load WLASL data
    print("\n2. Loading WLASL data...")
    wlasl_X = None
    wlasl_y = None
    wlasl_label_map = None

    wlasl_X_path = processed_dir / 'X_wlasl.npy'
    wlasl_y_path = processed_dir / 'y_wlasl.npy'
    wlasl_label_path = processed_dir / 'wlasl_label_map.json'

    if wlasl_X_path.exists() and wlasl_y_path.exists():
        wlasl_X = np.load(wlasl_X_path)
        wlasl_y = np.load(wlasl_y_path)
        print(f"  X_wlasl: {wlasl_X.shape}")
        print(f"  y_wlasl: {wlasl_y.shape}")

        if wlasl_label_path.exists():
            with open(wlasl_label_path, 'r') as f:
                wlasl_label_map = json.load(f)
            print(f"  WLASL vocabulary: {len(wlasl_label_map)} signs")
        else:
            print("  ERROR: WLASL label map not found!")
            return None
    else:
        print("  Warning: WLASL data not found, will only copy MVP data")

    # Create merged label map
    print("\n3. Creating merged label map...")

    if wlasl_label_map is not None:
        # Merge vocabularies
        all_signs = set(wlasl_label_map.keys()) | set(mvp_label_map.keys())
        merged_signs = sorted(all_signs)
        merged_label_map = {s: i for i, s in enumerate(merged_signs)}

        wlasl_only = set(wlasl_label_map.keys()) - set(mvp_label_map.keys())
        mvp_only = set(mvp_label_map.keys()) - set(wlasl_label_map.keys())
        overlap = set(mvp_label_map.keys()) & set(wlasl_label_map.keys())

        print(f"  Total vocabulary: {len(merged_label_map)} signs")
        print(f"    - WLASL only: {len(wlasl_only)}")
        print(f"    - MVP only: {len(mvp_only)} (signs: {mvp_only})")
        print(f"    - Overlap: {len(overlap)}")
    else:
        merged_label_map = mvp_label_map
        print(f"  Using MVP vocabulary: {len(merged_label_map)} signs")

    # Remap MVP labels to merged label map
    print("\n4. Remapping MVP labels...")
    for split in ['train', 'val', 'test']:
        if f'y_{split}' in mvp_data:
            mvp_data[f'y_{split}'] = remap_labels(
                mvp_data[f'y_{split}'],
                mvp_label_map,
                merged_label_map
            )
            print(f"  Remapped y_{split}")

    # Process WLASL data
    if wlasl_X is not None and wlasl_label_map is not None:
        print("\n5. Processing WLASL data...")

        # Remap WLASL labels
        wlasl_y = remap_labels(wlasl_y, wlasl_label_map, merged_label_map)
        print(f"  Remapped WLASL labels")

        # Shuffle WLASL data
        indices = np.random.permutation(len(wlasl_X))
        wlasl_X = wlasl_X[indices]
        wlasl_y = wlasl_y[indices]

        # Split: 70% train, 15% val, 15% test
        n_total = len(wlasl_X)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)

        wlasl_splits = {
            'X_train': wlasl_X[:n_train],
            'y_train': wlasl_y[:n_train],
            'X_val': wlasl_X[n_train:n_train + n_val],
            'y_val': wlasl_y[n_train:n_train + n_val],
            'X_test': wlasl_X[n_train + n_val:],
            'y_test': wlasl_y[n_train + n_val:],
        }

        print(f"  WLASL splits:")
        print(f"    Train: {wlasl_splits['X_train'].shape}")
        print(f"    Val: {wlasl_splits['X_val'].shape}")
        print(f"    Test: {wlasl_splits['X_test'].shape}")

        # Combine datasets
        print("\n6. Combining datasets...")
        extended_data = {}
        for split in ['train', 'val', 'test']:
            extended_data[f'X_{split}'] = np.concatenate([
                mvp_data[f'X_{split}'],
                wlasl_splits[f'X_{split}']
            ], axis=0)
            extended_data[f'y_{split}'] = np.concatenate([
                mvp_data[f'y_{split}'],
                wlasl_splits[f'y_{split}']
            ], axis=0)
            print(f"  {split}: {extended_data[f'X_{split}'].shape}")
    else:
        extended_data = mvp_data

    # Shuffle training data
    print("\n7. Shuffling training data...")
    indices = np.random.permutation(len(extended_data['X_train']))
    extended_data['X_train'] = extended_data['X_train'][indices]
    extended_data['y_train'] = extended_data['y_train'][indices]

    # Save extended dataset
    print(f"\n8. Saving to {output_dir}...")

    for split in ['train', 'val', 'test']:
        np.save(output_dir / f'X_{split}.npy', extended_data[f'X_{split}'])
        np.save(output_dir / f'y_{split}.npy', extended_data[f'y_{split}'])

    # Save merged label map
    with open(output_dir / 'label_map.json', 'w') as f:
        json.dump(merged_label_map, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("EXTENDED DATASET SUMMARY")
    print("=" * 60)

    print(f"\nMVP (Kaggle):")
    print(f"  Samples: {sum(len(mvp_data[f'X_{s}']) for s in ['train', 'val', 'test'])}")
    print(f"  Signs: {len(mvp_label_map)}")

    if wlasl_X is not None:
        print(f"\nWLASL:")
        print(f"  Samples: {len(wlasl_X)}")
        print(f"  Signs: {len(wlasl_label_map)}")

    print(f"\nExtended Dataset:")
    print(f"  Train: {len(extended_data['X_train'])}")
    print(f"  Val: {len(extended_data['X_val'])}")
    print(f"  Test: {len(extended_data['X_test'])}")
    total = sum(len(extended_data[f'X_{s}']) for s in ['train', 'val', 'test'])
    print(f"  Total: {total}")
    print(f"  Vocabulary: {len(merged_label_map)} signs")

    return extended_data, merged_label_map


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create extended dataset')
    parser.add_argument('--processed-dir', type=str,
                        default='data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--output-dir', type=str,
                        default=None,
                        help='Output directory (default: processed_dir/extended)')

    args = parser.parse_args()

    create_extended_dataset(args.processed_dir, args.output_dir)
