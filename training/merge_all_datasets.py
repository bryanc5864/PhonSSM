"""
Merge All ASL Datasets
Combines all available ASL datasets into a unified training set:
- MVP (Kaggle ASL signs)
- WLASL (Word-Level ASL)
- ASL Citizen
- ASL Alphabet (fingerspelling)
- ASL MNIST (fingerspelling)
- ChicagoFSWild (fingerspelling in the wild)
"""

import numpy as np
import json
from pathlib import Path


def remap_labels(y, old_label_map, new_label_map):
    """Remap labels from old to new label map."""
    old_idx_to_sign = {v: k for k, v in old_label_map.items()}
    y_new = np.zeros_like(y)
    valid_mask = np.ones(len(y), dtype=bool)

    for i, old_idx in enumerate(y):
        if old_idx in old_idx_to_sign:
            sign = old_idx_to_sign[old_idx]
            if sign in new_label_map:
                y_new[i] = new_label_map[sign]
            else:
                valid_mask[i] = False
        else:
            valid_mask[i] = False

    return y_new, valid_mask


def load_dataset(processed_dir, prefix, label_map_name):
    """Load a dataset with its label map."""
    X_path = processed_dir / f'X_{prefix}.npy'
    y_path = processed_dir / f'y_{prefix}.npy'
    lm_path = processed_dir / f'{label_map_name}.json'

    if X_path.exists() and y_path.exists() and lm_path.exists():
        X = np.load(X_path)
        y = np.load(y_path)
        with open(lm_path, 'r') as f:
            label_map = json.load(f)
        return X, y, label_map
    return None, None, None


def merge_all_datasets(processed_dir, output_dir=None):
    """
    Merge all available ASL datasets with proper label remapping.
    """
    processed_dir = Path(processed_dir)
    if output_dir is None:
        output_dir = processed_dir / 'merged'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MERGING ALL ASL DATASETS")
    print("=" * 60)

    all_datasets = {}
    all_label_maps = {}

    # 1. Load all available datasets
    print("\n1. Loading available datasets...")

    # MVP+WLASL (from extended if exists)
    if (processed_dir / 'extended' / 'X_train.npy').exists():
        mvp_dir = processed_dir / 'extended'
        print("   Using extended MVP+WLASL dataset")
    else:
        mvp_dir = processed_dir

    # Load MVP splits
    for split in ['train', 'val', 'test']:
        X_path = mvp_dir / f'X_{split}.npy'
        y_path = mvp_dir / f'y_{split}.npy'
        if X_path.exists():
            all_datasets[f'mvp_{split}'] = {
                'X': np.load(X_path),
                'y': np.load(y_path)
            }
            print(f"   MVP {split}: {all_datasets[f'mvp_{split}']['X'].shape}")

    label_map_path = mvp_dir / 'label_map.json'
    if label_map_path.exists():
        with open(label_map_path, 'r') as f:
            all_label_maps['mvp'] = json.load(f)
        print(f"   MVP vocabulary: {len(all_label_maps['mvp'])} signs")

    # Other datasets (single files, need splitting)
    other_datasets = [
        ('asl_citizen', 'asl_citizen_label_map', 'ASL Citizen'),
        ('asl_alphabet', 'asl_alphabet_label_map', 'ASL Alphabet'),
        ('asl_mnist', 'asl_mnist_label_map', 'ASL MNIST'),
        ('chicagofswild', 'chicagofswild_label_map', 'ChicagoFSWild'),
        ('wlasl', 'wlasl_label_map', 'WLASL'),
    ]

    for prefix, lm_name, display_name in other_datasets:
        X, y, lm = load_dataset(processed_dir, prefix, lm_name)
        if X is not None:
            all_datasets[prefix] = {'X': X, 'y': y}
            all_label_maps[prefix] = lm
            print(f"   {display_name}: {X.shape}, vocab: {len(lm)} signs")

    # 2. Create unified label map
    print("\n2. Creating unified label map...")

    all_signs = set()
    for name, lmap in all_label_maps.items():
        all_signs.update(lmap.keys())

    # Sort for reproducibility
    unified_signs = sorted(all_signs)
    unified_label_map = {s: i for i, s in enumerate(unified_signs)}

    print(f"   Total unique signs: {len(unified_label_map)}")

    # Show some overlaps
    if 'mvp' in all_label_maps and 'asl_citizen' in all_label_maps:
        overlap = set(all_label_maps['mvp'].keys()) & set(all_label_maps['asl_citizen'].keys())
        print(f"   MVP-ASL Citizen overlap: {len(overlap)} signs")

    # 3. Remap all labels
    print("\n3. Remapping labels...")

    for name, lmap in all_label_maps.items():
        if name == 'mvp':
            # MVP has splits
            for split in ['train', 'val', 'test']:
                key = f'mvp_{split}'
                if key in all_datasets:
                    y_new, valid = remap_labels(all_datasets[key]['y'], lmap, unified_label_map)
                    all_datasets[key]['y'] = y_new[valid]
                    all_datasets[key]['X'] = all_datasets[key]['X'][valid]
                    print(f"   {key}: {np.sum(valid)}/{len(valid)} valid")
        else:
            if name in all_datasets:
                y_new, valid = remap_labels(all_datasets[name]['y'], lmap, unified_label_map)
                all_datasets[name]['y'] = y_new[valid]
                all_datasets[name]['X'] = all_datasets[name]['X'][valid]
                print(f"   {name}: {np.sum(valid)}/{len(valid)} valid")

    # 4. Split non-MVP datasets and combine
    print("\n4. Splitting and combining datasets...")

    train_X, train_y = [], []
    val_X, val_y = [], []
    test_X, test_y = [], []

    # Add MVP splits directly
    for split, (X_list, y_list) in [('train', (train_X, train_y)),
                                      ('val', (val_X, val_y)),
                                      ('test', (test_X, test_y))]:
        key = f'mvp_{split}'
        if key in all_datasets:
            X_list.append(all_datasets[key]['X'])
            y_list.append(all_datasets[key]['y'])

    # Split and add other datasets (80/10/10)
    for name in ['asl_citizen', 'asl_alphabet', 'asl_mnist', 'chicagofswild']:
        if name in all_datasets:
            X = all_datasets[name]['X']
            y = all_datasets[name]['y']

            n_total = len(X)
            indices = np.random.permutation(n_total)
            X = X[indices]
            y = y[indices]

            n_train = int(0.8 * n_total)
            n_val = int(0.1 * n_total)

            train_X.append(X[:n_train])
            train_y.append(y[:n_train])
            val_X.append(X[n_train:n_train+n_val])
            val_y.append(y[n_train:n_train+n_val])
            test_X.append(X[n_train+n_val:])
            test_y.append(y[n_train+n_val:])

            print(f"   {name}: train={n_train}, val={n_val}, test={n_total-n_train-n_val}")

    # Concatenate all
    X_train = np.concatenate(train_X, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    X_val = np.concatenate(val_X, axis=0)
    y_val = np.concatenate(val_y, axis=0)
    X_test = np.concatenate(test_X, axis=0)
    y_test = np.concatenate(test_y, axis=0)

    # 5. Shuffle training data
    print("\n5. Shuffling training data...")
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]

    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Test: {X_test.shape}")

    # Verify labels are in range
    print("\n6. Verifying label integrity...")
    max_label = len(unified_label_map) - 1
    for name, y in [('train', y_train), ('val', y_val), ('test', y_test)]:
        out_of_range = np.sum((y < 0) | (y > max_label))
        print(f"   {name}: min={y.min()}, max={y.max()}, out_of_range={out_of_range}")

    # 7. Save
    print(f"\n7. Saving to {output_dir}...")

    np.save(output_dir / 'X_train.npy', X_train)
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'X_val.npy', X_val)
    np.save(output_dir / 'y_val.npy', y_val)
    np.save(output_dir / 'X_test.npy', X_test)
    np.save(output_dir / 'y_test.npy', y_test)

    with open(output_dir / 'label_map.json', 'w') as f:
        json.dump(unified_label_map, f, indent=2)

    # Save stats
    stats = {
        'total_samples': len(X_train) + len(X_val) + len(X_test),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'vocabulary_size': len(unified_label_map),
        'shape': list(X_train.shape)
    }
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("MERGED DATASET SUMMARY")
    print("=" * 60)

    print(f"\nTotal samples: {stats['total_samples']}")
    print(f"Total vocabulary: {stats['vocabulary_size']} signs")
    print(f"\nSplits:")
    print(f"  train: {stats['train_samples']}")
    print(f"  val: {stats['val_samples']}")
    print(f"  test: {stats['test_samples']}")
    print(f"\nOutput: {output_dir}")

    return {'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test}, unified_label_map


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Merge all ASL datasets')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                        help='Directory containing processed datasets')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: processed_dir/merged)')

    args = parser.parse_args()
    merge_all_datasets(args.processed_dir, args.output_dir)
