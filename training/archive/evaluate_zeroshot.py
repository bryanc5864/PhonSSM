"""
Zero-shot evaluation of the main PhonSSM model on WLASL benchmarks.
No training - just evaluate the pretrained model on WLASL test sets.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.phonssm import PhonSSM, PhonSSMConfig


def load_main_model(checkpoint_path, device):
    """Load the main pretrained PhonSSM model."""
    print(f"Loading main model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint or use default
    if 'config' in checkpoint:
        config = PhonSSMConfig.from_dict(checkpoint['config'])
    else:
        # Default config for main model (single hand, 5565 signs)
        config = PhonSSMConfig(
            num_signs=5565,
            input_mode='single_hand',
            num_landmarks=21
        )

    model = PhonSSM(config).to(device)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"  Loaded model with {config.num_signs} classes")
    return model, config


def load_asl_citizen_for_zeroshot(main_label_map):
    """Load ASL Citizen data and find overlap with main model's vocabulary."""
    print(f"\nLoading ASL Citizen for zero-shot evaluation...")

    # Load ASL Citizen data
    X_all = np.load(PROJECT_ROOT / "data" / "processed" / "X_asl_citizen.npy")
    y_all = np.load(PROJECT_ROOT / "data" / "processed" / "y_asl_citizen.npy")

    with open(PROJECT_ROOT / "data" / "processed" / "asl_citizen_label_map.json") as f:
        citizen_label_map = json.load(f)

    citizen_idx_to_gloss = {v: k for k, v in citizen_label_map.items()}

    # Find overlap between ASL Citizen glosses and main model vocabulary
    main_glosses = set(main_label_map.keys())
    citizen_glosses = set(citizen_label_map.keys())
    overlap_glosses = list(citizen_glosses & main_glosses)

    print(f"  ASL Citizen glosses: {len(citizen_glosses)}")
    print(f"  Main model glosses: {len(main_glosses)}")
    print(f"  Overlap: {len(overlap_glosses)} ({100*len(overlap_glosses)/len(citizen_glosses):.1f}%)")

    if len(overlap_glosses) == 0:
        print("  ERROR: No overlap found!")
        return None

    # Create mapping: ASL Citizen gloss -> main model label
    overlap_set = set(overlap_glosses)
    gloss_to_main_label = {g: main_label_map[g] for g in overlap_glosses}

    # Filter data
    valid_indices = []
    mapped_labels = []

    for i, label in enumerate(y_all):
        gloss = citizen_idx_to_gloss.get(int(label), '')
        if gloss in overlap_set:
            valid_indices.append(i)
            mapped_labels.append(gloss_to_main_label[gloss])

    X_filtered = X_all[valid_indices]
    y_filtered = np.array(mapped_labels)

    # Split into test set
    from sklearn.model_selection import train_test_split

    _, X_test, _, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42
    )

    print(f"  Filtered samples: {len(X_filtered)}")
    print(f"  Test samples: {len(X_test)}")

    return {
        'X_test': X_test,
        'y_test': y_test,
        'num_overlap': len(overlap_glosses),
        'total_classes': len(citizen_glosses),
        'overlap_glosses': overlap_glosses,
        'dataset_name': 'ASL_Citizen'
    }


def load_wlasl_for_zeroshot(subset_size, main_label_map):
    """Load WLASL data and find overlap with main model's vocabulary."""
    print(f"\nLoading WLASL{subset_size} for zero-shot evaluation...")

    # Load WLASL JSON
    wlasl_json_path = PROJECT_ROOT / "data" / "raw" / "wlasl" / "start_kit" / "WLASL_v0.3.json"
    with open(wlasl_json_path) as f:
        wlasl_data = json.load(f)

    # Get top-K glosses for subset
    subset_glosses = [entry['gloss'].lower() for entry in wlasl_data[:subset_size]]

    # Load WLASL single-hand data (same format as main model)
    X_all = np.load(PROJECT_ROOT / "data" / "processed" / "X_wlasl.npy")
    y_all = np.load(PROJECT_ROOT / "data" / "processed" / "y_wlasl.npy")

    with open(PROJECT_ROOT / "data" / "processed" / "wlasl_label_map.json") as f:
        wlasl_label_map = json.load(f)

    wlasl_idx_to_gloss = {v: k for k, v in wlasl_label_map.items()}

    # Find overlap between WLASL glosses and main model vocabulary
    main_glosses = set(main_label_map.keys())
    overlap_glosses = [g for g in subset_glosses if g in main_glosses]

    print(f"  WLASL{subset_size} glosses: {len(subset_glosses)}")
    print(f"  Main model glosses: {len(main_glosses)}")
    print(f"  Overlap: {len(overlap_glosses)} ({100*len(overlap_glosses)/len(subset_glosses):.1f}%)")

    if len(overlap_glosses) == 0:
        print("  ERROR: No overlap found!")
        return None

    # Filter to overlapping glosses only
    overlap_set = set(overlap_glosses)

    # Create mapping: WLASL gloss -> main model label
    gloss_to_main_label = {g: main_label_map[g] for g in overlap_glosses}

    # Filter data
    valid_indices = []
    mapped_labels = []

    for i, label in enumerate(y_all):
        gloss = wlasl_idx_to_gloss.get(int(label), '')
        if gloss in overlap_set:
            valid_indices.append(i)
            mapped_labels.append(gloss_to_main_label[gloss])

    X_filtered = X_all[valid_indices]
    y_filtered = np.array(mapped_labels)

    # Split into train/val/test (use same ratios as benchmark)
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_filtered, y_filtered, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"  Filtered samples: {len(X_filtered)}")
    print(f"  Test samples: {len(X_test)}")

    return {
        'X_test': X_test,
        'y_test': y_test,
        'num_overlap': len(overlap_glosses),
        'total_classes': subset_size,
        'overlap_glosses': overlap_glosses,
        'dataset_name': f'WLASL{subset_size}'
    }


@torch.no_grad()
def evaluate_zeroshot(model, data, device):
    """Evaluate model on test data."""
    model.eval()

    X_test = torch.FloatTensor(data['X_test'])
    y_test = data['y_test']

    test_dataset = TensorDataset(X_test, torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=0)

    all_logits = []
    all_targets = []

    for X_batch, y_batch in tqdm(test_loader, desc='Evaluating'):
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        all_logits.append(outputs['logits'].cpu())
        all_targets.append(y_batch)

    logits = torch.cat(all_logits, dim=0).numpy()
    y_true = np.array(y_test)
    y_pred = logits.argmax(axis=-1)

    # Get unique labels for top-k accuracy
    unique_labels = np.unique(y_true)

    metrics = {}
    metrics['top1_accuracy'] = accuracy_score(y_true, y_pred) * 100

    # Top-5 and Top-10 within the valid label space
    for k in [5, 10]:
        if k <= len(unique_labels):
            # Get top-k predictions
            top_k_preds = np.argsort(logits, axis=-1)[:, -k:]
            correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
            metrics[f'top{k}_accuracy'] = correct.mean() * 100

    # Per-class accuracy
    per_class_acc = []
    for c in unique_labels:
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc.append((y_pred[mask] == c).mean())

    metrics['per_class_accuracy'] = np.mean(per_class_acc) * 100
    metrics['per_class_std'] = np.std(per_class_acc) * 100

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Zero-shot evaluation on WLASL/ASL Citizen')
    parser.add_argument('--dataset', type=str, default='wlasl', choices=['wlasl', 'asl_citizen', 'all'],
                        help='Dataset to evaluate on (wlasl, asl_citizen, or all)')
    parser.add_argument('--subset', type=int, default=100, help='WLASL subset (100/300/1000/2000)')
    parser.add_argument('--checkpoint', type=str,
                        default='models/phonssm/checkpoints/20260116_230934/best_model.pt',
                        help='Path to main model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print("ZERO-SHOT EVALUATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load main model's label map
    main_label_map_path = PROJECT_ROOT / "data" / "processed" / "merged" / "label_map.json"
    with open(main_label_map_path) as f:
        main_label_map = json.load(f)

    print(f"\nMain model vocabulary: {len(main_label_map)} signs")

    # Load model
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    model, config = load_main_model(checkpoint_path, device)

    # Determine which datasets to evaluate
    datasets_to_eval = []
    if args.dataset == 'wlasl' or args.dataset == 'all':
        datasets_to_eval.append(('wlasl', args.subset))
    if args.dataset == 'asl_citizen' or args.dataset == 'all':
        datasets_to_eval.append(('asl_citizen', None))

    # Create output directory
    output_dir = PROJECT_ROOT / "benchmarks" / "zeroshot"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset_name, subset in datasets_to_eval:
        print(f"\n{'='*60}")

        # Load data
        if dataset_name == 'wlasl':
            data = load_wlasl_for_zeroshot(subset, main_label_map)
            display_name = f'WLASL{subset}'
        else:
            data = load_asl_citizen_for_zeroshot(main_label_map)
            display_name = 'ASL_Citizen'

        if data is None:
            print(f"Cannot evaluate {display_name} - no vocabulary overlap")
            continue

        # Evaluate
        print(f"\nZERO-SHOT EVALUATION on {display_name}")
        print(f"{'='*60}")
        print(f"Evaluating on {data['num_overlap']}/{data['total_classes']} overlapping classes")

        metrics = evaluate_zeroshot(model, data, device)

        print(f"\n--- Results for {display_name} ---")
        print(f"Top-1 Accuracy:      {metrics['top1_accuracy']:.2f}%")
        if 'top5_accuracy' in metrics:
            print(f"Top-5 Accuracy:      {metrics['top5_accuracy']:.2f}%")
        if 'top10_accuracy' in metrics:
            print(f"Top-10 Accuracy:     {metrics['top10_accuracy']:.2f}%")
        print(f"Per-Class Accuracy:  {metrics['per_class_accuracy']:.2f}% (+/- {metrics['per_class_std']:.2f}%)")
        print(f"\nNote: Evaluated on {data['num_overlap']} overlapping classes out of {data['total_classes']}")

        # Save individual results
        results = {
            'dataset': display_name,
            'evaluation_type': 'zero-shot',
            'overlap_classes': data['num_overlap'],
            'total_classes': data['total_classes'],
            'test_samples': len(data['X_test']),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        output_path = output_dir / f"{display_name.lower()}_zeroshot.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")

        all_results[display_name] = metrics

    # Print summary if multiple datasets
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("ZERO-SHOT EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"\n{'Dataset':<20} {'Top-1':>10} {'Top-5':>10} {'Per-Class':>12}")
        print("-" * 55)
        for name, metrics in all_results.items():
            top5 = f"{metrics.get('top5_accuracy', 0):.2f}%" if 'top5_accuracy' in metrics else "N/A"
            print(f"{name:<20} {metrics['top1_accuracy']:>9.2f}% {top5:>10} {metrics['per_class_accuracy']:>11.2f}%")


if __name__ == "__main__":
    main()
