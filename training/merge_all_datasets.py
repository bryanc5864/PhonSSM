"""
Merge All ASL Datasets
Combines MVP (Kaggle), WLASL, and ASL Citizen into a unified dataset.
"""

import numpy as np
import json
from pathlib import Path


def remap_labels(y, old_label_map, new_label_map):
    """Remap labels from old to new label map."""
    old_idx_to_sign = {v: k for k, v in old_label_map.items()}
    y_new = np.zeros_like(y)
    for i, old_idx in enumerate(y):
        sign = old_idx_to_sign[old_idx]
        if sign in new_label_map:
            y_new[i] = new_label_map[sign]
        else:
            y_new[i] = -1  # Mark for removal
    return y_new


def merge_all_datasets(processed_dir, output_dir=None):
    """
    Merge all available ASL datasets.

    Expected files in processed_dir:
    - MVP: X_train.npy, y_train.npy, label_map.json
    - WLASL: X_wlasl.npy, y_wlasl.npy, wlasl_label_map.json
    - ASL Citizen: X_asl_citizen.npy, y_asl_citizen.npy, asl_citizen_label_map.json
    """
    processed_dir = Path(processed_dir)
    if output_dir is None:
        output_dir = processed_dir / 'merged'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MERGING ALL ASL DATASETS")
    print("=" * 60)

    datasets = {}
    label_maps = {}

    # 1. Load MVP data
    print("\n1. Loading available datasets...")

    # MVP (from extended if exists, otherwise base)
    if (processed_dir / 'extended' / 'X_train.npy').exists():
        mvp_dir = processed_dir / 'extended'
        print("   Using extended MVP+WLASL dataset")
    else:
        mvp_dir = processed_dir

    for split in ['train', 'val', 'test']:
        X_path = mvp_dir / f'X_{split}.npy'
        y_path = mvp_dir / f'y_{split}.npy'
        if X_path.exists():
            datasets[f'mvp_X_{split}'] = np.load(X_path)
            datasets[f'mvp_y_{split}'] = np.load(y_path)
            print(f"   MVP {split}: {datasets[f'mvp_X_{split}'].shape}")

    label_map_path = mvp_dir / 'label_map.json'
    if label_map_path.exists():
        with open(label_map_path, 'r') as f:
            label_maps['mvp'] = json.load(f)
        print(f"   MVP vocabulary: {len(label_maps['mvp'])} signs")

    # ASL Citizen
    asl_citizen_X = processed_dir / 'X_asl_citizen.npy'
    asl_citizen_y = processed_dir / 'y_asl_citizen.npy'
    asl_citizen_labels = processed_dir / 'asl_citizen_label_map.json'

    if asl_citizen_X.exists():
        datasets['citizen_X'] = np.load(asl_citizen_X)
        datasets['citizen_y'] = np.load(asl_citizen_y)
        with open(asl_citizen_labels, 'r') as f:
            label_maps['citizen'] = json.load(f)
        print(f"   ASL Citizen: {datasets['citizen_X'].shape}")
        print(f"   ASL Citizen vocabulary: {len(label_maps['citizen'])} signs")
    else:
        print("   ASL Citizen: Not found (skipping)")

    # 2. Create unified label map
    print("\n2. Creating unified label map...")

    all_signs = set()
    for name, lmap in label_maps.items():
        all_signs.update(lmap.keys())

    unified_signs = sorted(all_signs)
    unified_label_map = {s: i for i, s in enumerate(unified_signs)}

    print(f"   Total unique signs: {len(unified_label_map)}")

    # Count overlaps
    if 'mvp' in label_maps and 'citizen' in label_maps:
        overlap = set(label_maps['mvp'].keys()) & set(label_maps['citizen'].keys())
        print(f"   MVP-Citizen overlap: {len(overlap)} signs")

    # 3. Remap and combine
    print("\n3. Remapping labels and combining...")

    # Remap MVP
    if 'mvp' in label_maps:
        for split in ['train', 'val', 'test']:
            if f'mvp_y_{split}' in datasets:
                datasets[f'mvp_y_{split}'] = remap_labels(
                    datasets[f'mvp_y_{split}'],
                    label_maps['mvp'],
                    unified_label_map
                )

    # Remap and split ASL Citizen
    if 'citizen_X' in datasets:
        datasets['citizen_y'] = remap_labels(
            datasets['citizen_y'],
            label_maps['citizen'],
            unified_label_map
        )

        # Remove any samples with invalid labels
        valid_mask = datasets['citizen_y'] >= 0
        datasets['citizen_X'] = datasets['citizen_X'][valid_mask]
        datasets['citizen_y'] = datasets['citizen_y'][valid_mask]

        # Shuffle and split ASL Citizen (70/15/15)
        n_total = len(datasets['citizen_X'])
        indices = np.random.permutation(n_total)
        datasets['citizen_X'] = datasets['citizen_X'][indices]
        datasets['citizen_y'] = datasets['citizen_y'][indices]

        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)

        citizen_splits = {
            'train': (datasets['citizen_X'][:n_train],
                     datasets['citizen_y'][:n_train]),
            'val': (datasets['citizen_X'][n_train:n_train+n_val],
                   datasets['citizen_y'][n_train:n_train+n_val]),
            'test': (datasets['citizen_X'][n_train+n_val:],
                    datasets['citizen_y'][n_train+n_val:])
        }

        print(f"   ASL Citizen splits: train={n_train}, val={n_val}, test={n_total-n_train-n_val}")

    # 4. Combine all
    print("\n4. Combining datasets...")

    merged = {}
    for split in ['train', 'val', 'test']:
        X_parts = []
        y_parts = []

        # MVP
        if f'mvp_X_{split}' in datasets:
            X_parts.append(datasets[f'mvp_X_{split}'])
            y_parts.append(datasets[f'mvp_y_{split}'])

        # ASL Citizen
        if 'citizen_X' in datasets:
            X_parts.append(citizen_splits[split][0])
            y_parts.append(citizen_splits[split][1])

        if X_parts:
            merged[f'X_{split}'] = np.concatenate(X_parts, axis=0)
            merged[f'y_{split}'] = np.concatenate(y_parts, axis=0)
            print(f"   {split}: {merged[f'X_{split}'].shape}")

    # 5. Shuffle training data
    print("\n5. Shuffling training data...")
    if 'X_train' in merged:
        indices = np.random.permutation(len(merged['X_train']))
        merged['X_train'] = merged['X_train'][indices]
        merged['y_train'] = merged['y_train'][indices]

    # 6. Save
    print(f"\n6. Saving to {output_dir}...")

    for key, data in merged.items():
        np.save(output_dir / f'{key}.npy', data)

    with open(output_dir / 'label_map.json', 'w') as f:
        json.dump(unified_label_map, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("MERGED DATASET SUMMARY")
    print("=" * 60)

    total_samples = sum(len(merged[f'X_{s}']) for s in ['train', 'val', 'test'] if f'X_{s}' in merged)

    print(f"\nTotal samples: {total_samples}")
    print(f"Total vocabulary: {len(unified_label_map)} signs")
    print(f"\nSplits:")
    for split in ['train', 'val', 'test']:
        if f'X_{split}' in merged:
            print(f"  {split}: {len(merged[f'X_{split}'])}")

    print(f"\nOutput: {output_dir}")

    return merged, unified_label_map


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Merge all ASL datasets')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                        help='Directory containing processed datasets')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: processed_dir/merged)')

    args = parser.parse_args()
    merge_all_datasets(args.processed_dir, args.output_dir)
