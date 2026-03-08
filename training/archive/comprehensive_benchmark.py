"""
Comprehensive Benchmark Script for SignSense Models
====================================================
Compares PhonSSM vs Bi-LSTM baseline across multiple metrics.

Metrics collected:
- Top-1, Top-3, Top-5, Top-10 Accuracy
- Macro/Weighted F1
- Per-class accuracy statistics
- Inference time
- Parameters & FLOPs
- Confusion analysis
- Few-shot performance (by samples per class)

Usage:
    python training/comprehensive_benchmark.py
    python training/comprehensive_benchmark.py --dataset merged --device cuda
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Metrics
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    top_k_accuracy_score, classification_report
)


def load_validation_data(data_dir: Path):
    """Load validation and test data."""
    print(f"Loading data from {data_dir}...")

    X_val = np.load(data_dir / "X_val.npy", allow_pickle=True)
    y_val = np.load(data_dir / "y_val.npy", allow_pickle=True)
    X_test = np.load(data_dir / "X_test.npy", allow_pickle=True)
    y_test = np.load(data_dir / "y_test.npy", allow_pickle=True)

    # Load label map if exists
    label_map = {}
    label_map_path = data_dir / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_map = json.load(f)

    # Compute class counts from training data
    y_train = np.load(data_dir / "y_train.npy", allow_pickle=True)
    class_counts = defaultdict(int)
    for label in y_train:
        class_counts[int(label)] += 1

    num_classes = len(label_map) if label_map else int(max(y_test) + 1)

    print(f"  Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {num_classes}")

    return {
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'num_classes': num_classes,
        'label_map': label_map,
        'class_counts': dict(class_counts)
    }


def compute_metrics(y_true, y_pred, logits, num_classes, class_counts=None):
    """Compute comprehensive metrics."""
    metrics = {}

    # === Primary Accuracy Metrics ===
    metrics['top1_accuracy'] = accuracy_score(y_true, y_pred) * 100

    # Top-K accuracy (need logits)
    for k in [3, 5, 10]:
        if k <= num_classes:
            metrics[f'top{k}_accuracy'] = top_k_accuracy_score(
                y_true, logits, k=k, labels=range(num_classes)
            ) * 100

    # === F1 Scores ===
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100

    # === Per-Class Accuracy ===
    per_class_acc = []
    per_class_counts = []
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            acc = (y_pred[mask] == c).mean()
            per_class_acc.append(acc)
            per_class_counts.append(mask.sum())

    metrics['per_class_accuracy_mean'] = np.mean(per_class_acc) * 100
    metrics['per_class_accuracy_std'] = np.std(per_class_acc) * 100
    metrics['per_class_accuracy_min'] = np.min(per_class_acc) * 100
    metrics['per_class_accuracy_max'] = np.max(per_class_acc) * 100

    # === Confusion Analysis ===
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    # Top confusions (off-diagonal)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    top_k = min(10, (cm_no_diag > 0).sum())
    if top_k > 0:
        flat_indices = np.argsort(cm_no_diag.ravel())[-top_k:]
        top_confusions = []
        for idx in flat_indices[::-1]:
            i, j = divmod(idx, num_classes)
            count = cm_no_diag[i, j]
            if count > 0:
                top_confusions.append({'true': int(i), 'pred': int(j), 'count': int(count)})
        metrics['top_confusions'] = top_confusions

    # === Few-Shot Performance ===
    if class_counts:
        bins = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
        few_shot_metrics = {}

        for low, high in bins:
            bin_mask = np.zeros(len(y_true), dtype=bool)
            for i, label in enumerate(y_true):
                count = class_counts.get(int(label), 0)
                if low <= count <= high:
                    bin_mask[i] = True

            if bin_mask.sum() > 0:
                bin_name = f"{low}-{int(high)}" if high != float('inf') else f"{low}+"
                bin_acc = accuracy_score(y_true[bin_mask], y_pred[bin_mask]) * 100
                few_shot_metrics[f'accuracy_{bin_name}_samples'] = bin_acc
                few_shot_metrics[f'count_{bin_name}_samples'] = int(bin_mask.sum())

        metrics['few_shot'] = few_shot_metrics

    return metrics


def benchmark_bilstm(data, device='cpu'):
    """Benchmark the Bi-LSTM baseline model."""
    print("\n" + "="*60)
    print("BENCHMARKING: Bi-LSTM Baseline")
    print("="*60)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf

    # Load model
    model_path = PROJECT_ROOT / "models" / "sign_classifier" / "sign_classifier.keras"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "models" / "sign_classifier" / "best_model.keras"

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Model info
    total_params = model.count_params()
    print(f"Parameters: {total_params:,}")

    # Prepare data
    X_test = data['X_test']
    y_test = data['y_test']

    # Inference
    print("Running inference...")

    # Warm-up
    _ = model.predict(X_test[:10], verbose=0)

    # Timed inference
    start_time = time.time()
    logits = model.predict(X_test, verbose=0)
    inference_time = time.time() - start_time

    y_pred = np.argmax(logits, axis=-1)

    # Compute metrics
    metrics = compute_metrics(
        y_test, y_pred, logits,
        data['num_classes'],
        data['class_counts']
    )

    # Add model info
    metrics['model'] = 'Bi-LSTM'
    metrics['parameters'] = total_params
    metrics['inference_time_total_s'] = inference_time
    metrics['inference_time_per_sample_ms'] = (inference_time / len(X_test)) * 1000
    metrics['throughput_samples_per_sec'] = len(X_test) / inference_time

    return metrics


def benchmark_phonssm(data, device='cuda'):
    """Benchmark the PhonSSM model."""
    print("\n" + "="*60)
    print("BENCHMARKING: PhonSSM")
    print("="*60)

    import torch
    from models.phonssm import PhonSSM, PhonSSMConfig

    # Device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    device = torch.device(device)
    print(f"Device: {device}")

    # Load model
    checkpoint_dir = PROJECT_ROOT / "models" / "phonssm" / "checkpoints"
    # Find latest checkpoint
    checkpoints = list(checkpoint_dir.glob("*/best_model.pt"))
    if not checkpoints:
        print("ERROR: No PhonSSM checkpoint found!")
        return None

    checkpoint_path = sorted(checkpoints)[-1]  # Latest
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    config_dict = checkpoint.get('config', {})
    config = PhonSSMConfig(**{k: v for k, v in config_dict.items()
                              if k in PhonSSMConfig.__dataclass_fields__})
    model = PhonSSM(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Prepare data
    X_test = torch.FloatTensor(data['X_test']).to(device)
    y_test = data['y_test']

    # Inference
    print("Running inference...")
    batch_size = 64
    all_logits = []

    # Warm-up
    with torch.no_grad():
        _ = model(X_test[:10])

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed inference
    start_time = time.time()
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            outputs = model(batch)
            all_logits.append(outputs['logits'].cpu().numpy())

    if device.type == 'cuda':
        torch.cuda.synchronize()
    inference_time = time.time() - start_time

    logits = np.vstack(all_logits)
    y_pred = np.argmax(logits, axis=-1)

    # Compute metrics
    metrics = compute_metrics(
        y_test, y_pred, logits,
        data['num_classes'],
        data['class_counts']
    )

    # Add model info
    metrics['model'] = 'PhonSSM'
    metrics['parameters'] = total_params
    metrics['inference_time_total_s'] = inference_time
    metrics['inference_time_per_sample_ms'] = (inference_time / len(X_test)) * 1000
    metrics['throughput_samples_per_sec'] = len(X_test) / inference_time

    # GPU memory if applicable
    if device.type == 'cuda':
        metrics['gpu_memory_mb'] = torch.cuda.max_memory_allocated() / 1024**2

    return metrics


def print_comparison_table(bilstm_metrics, phonssm_metrics):
    """Print side-by-side comparison."""
    print("\n" + "="*80)
    print("COMPARISON: Bi-LSTM vs PhonSSM")
    print("="*80)

    def fmt(v, decimals=2):
        if isinstance(v, float):
            return f"{v:.{decimals}f}"
        elif isinstance(v, int):
            return f"{v:,}"
        return str(v)

    def delta(b, p):
        if isinstance(b, (int, float)) and isinstance(p, (int, float)):
            d = p - b
            sign = "+" if d > 0 else ""
            return f"{sign}{d:.2f}"
        return "-"

    rows = [
        ("Model", "model", "model"),
        ("", "", ""),
        ("=== Accuracy ===", "", ""),
        ("Top-1 Accuracy (%)", "top1_accuracy", True),
        ("Top-3 Accuracy (%)", "top3_accuracy", True),
        ("Top-5 Accuracy (%)", "top5_accuracy", True),
        ("Top-10 Accuracy (%)", "top10_accuracy", True),
        ("", "", ""),
        ("=== F1 Scores ===", "", ""),
        ("Macro F1 (%)", "macro_f1", True),
        ("Weighted F1 (%)", "weighted_f1", True),
        ("", "", ""),
        ("=== Per-Class Stats ===", "", ""),
        ("Mean Accuracy (%)", "per_class_accuracy_mean", True),
        ("Std Accuracy (%)", "per_class_accuracy_std", False),
        ("Min Accuracy (%)", "per_class_accuracy_min", True),
        ("Max Accuracy (%)", "per_class_accuracy_max", True),
        ("", "", ""),
        ("=== Efficiency ===", "", ""),
        ("Parameters", "parameters", False),
        ("Inference (ms/sample)", "inference_time_per_sample_ms", False),
        ("Throughput (samples/s)", "throughput_samples_per_sec", True),
    ]

    print(f"{'Metric':<30} {'Bi-LSTM':>15} {'PhonSSM':>15} {'Delta':>12}")
    print("-" * 75)

    for row in rows:
        if len(row) == 3:
            label, key, show_delta = row
            if key == "":
                print(label)
                continue

            b_val = bilstm_metrics.get(key, "-")
            p_val = phonssm_metrics.get(key, "-") if phonssm_metrics else "-"

            d = delta(b_val, p_val) if show_delta and phonssm_metrics else "-"

            print(f"{label:<30} {fmt(b_val):>15} {fmt(p_val):>15} {d:>12}")

    # Few-shot comparison
    if 'few_shot' in bilstm_metrics:
        print("\n" + "="*80)
        print("FEW-SHOT PERFORMANCE (by training samples per class)")
        print("="*80)
        print(f"{'Samples/Class':<20} {'Bi-LSTM':>12} {'PhonSSM':>12} {'# Test':>10}")
        print("-" * 60)

        for key in sorted(bilstm_metrics['few_shot'].keys()):
            if key.startswith('accuracy_'):
                bin_name = key.replace('accuracy_', '').replace('_samples', '')
                count_key = f'count_{bin_name}_samples'

                b_acc = bilstm_metrics['few_shot'].get(key, 0)
                p_acc = phonssm_metrics['few_shot'].get(key, 0) if phonssm_metrics else 0
                count = bilstm_metrics['few_shot'].get(count_key, 0)

                print(f"{bin_name:<20} {b_acc:>11.2f}% {p_acc:>11.2f}% {count:>10}")


def save_results(bilstm_metrics, phonssm_metrics, output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        'timestamp': timestamp,
        'bilstm': {k: v for k, v in bilstm_metrics.items() if k != 'top_confusions'},
        'phonssm': {k: v for k, v in (phonssm_metrics or {}).items() if k != 'top_confusions'} if phonssm_metrics else None
    }

    output_path = output_dir / f"benchmark_results_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\nResults saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Benchmark')
    parser.add_argument('--dataset', type=str, default='merged',
                        choices=['merged', 'extended', 'processed'],
                        help='Dataset to use')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for PhonSSM (cuda or cpu)')
    parser.add_argument('--skip-bilstm', action='store_true',
                        help='Skip Bi-LSTM benchmark')
    parser.add_argument('--skip-phonssm', action='store_true',
                        help='Skip PhonSSM benchmark')

    args = parser.parse_args()

    print("="*80)
    print("SIGNSENSE COMPREHENSIVE BENCHMARK")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")

    # Load data
    if args.dataset == 'merged':
        data_dir = PROJECT_ROOT / "data" / "processed" / "merged"
    elif args.dataset == 'extended':
        data_dir = PROJECT_ROOT / "data" / "processed" / "extended"
    else:
        data_dir = PROJECT_ROOT / "data" / "processed"

    data = load_validation_data(data_dir)

    # Benchmark Bi-LSTM
    bilstm_metrics = None
    if not args.skip_bilstm:
        try:
            bilstm_metrics = benchmark_bilstm(data)
        except Exception as e:
            print(f"ERROR benchmarking Bi-LSTM: {e}")
            import traceback
            traceback.print_exc()

    # Benchmark PhonSSM
    phonssm_metrics = None
    if not args.skip_phonssm:
        try:
            phonssm_metrics = benchmark_phonssm(data, args.device)
        except Exception as e:
            print(f"ERROR benchmarking PhonSSM: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison
    if bilstm_metrics:
        print_comparison_table(bilstm_metrics, phonssm_metrics)

    # Save results
    output_dir = PROJECT_ROOT / "benchmarks"
    if bilstm_metrics or phonssm_metrics:
        save_results(bilstm_metrics or {}, phonssm_metrics, output_dir)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
