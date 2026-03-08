"""
Bi-LSTM Benchmark Script
========================
Evaluates the Bi-LSTM baseline with full metrics.

Usage:
    conda activate tf-gpu
    python training/benchmark_bilstm.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Metrics
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    top_k_accuracy_score
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "merged"
MODEL_DIR = PROJECT_ROOT / "models" / "sign_classifier"


def load_data():
    """Load test data."""
    print("Loading data...")

    X_test = np.load(DATA_DIR / "X_test.npy", allow_pickle=True)
    y_test = np.load(DATA_DIR / "y_test.npy", allow_pickle=True)
    X_val = np.load(DATA_DIR / "X_val.npy", allow_pickle=True)
    y_val = np.load(DATA_DIR / "y_val.npy", allow_pickle=True)
    y_train = np.load(DATA_DIR / "y_train.npy", allow_pickle=True)

    with open(DATA_DIR / "label_map.json") as f:
        label_map = json.load(f)

    # Class counts from training
    class_counts = defaultdict(int)
    for label in y_train:
        class_counts[int(label)] += 1

    num_classes = len(label_map)
    print(f"  Test: {X_test.shape}, Val: {X_val.shape}")
    print(f"  Classes: {num_classes}")

    return {
        'X_test': X_test, 'y_test': y_test,
        'X_val': X_val, 'y_val': y_val,
        'num_classes': num_classes,
        'label_map': label_map,
        'class_counts': dict(class_counts)
    }


def compute_metrics(y_true, y_pred, logits, num_classes, class_counts=None):
    """Compute comprehensive metrics."""
    metrics = {}

    # === Primary Accuracy Metrics ===
    metrics['top1_accuracy'] = accuracy_score(y_true, y_pred) * 100

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
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            acc = (y_pred[mask] == c).mean()
            per_class_acc.append(acc)

    metrics['per_class_accuracy_mean'] = np.mean(per_class_acc) * 100
    metrics['per_class_accuracy_std'] = np.std(per_class_acc) * 100
    metrics['per_class_accuracy_min'] = np.min(per_class_acc) * 100
    metrics['per_class_accuracy_max'] = np.max(per_class_acc) * 100

    # === Confusion Analysis ===
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
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
        few_shot = {}

        for low, high in bins:
            bin_mask = np.zeros(len(y_true), dtype=bool)
            for i, label in enumerate(y_true):
                count = class_counts.get(int(label), 0)
                if low <= count <= high:
                    bin_mask[i] = True

            if bin_mask.sum() > 0:
                bin_name = f"{low}-{int(high)}" if high != float('inf') else f"{low}+"
                bin_acc = accuracy_score(y_true[bin_mask], y_pred[bin_mask]) * 100
                few_shot[f'accuracy_{bin_name}_samples'] = bin_acc
                few_shot[f'count_{bin_name}_samples'] = int(bin_mask.sum())

        metrics['few_shot'] = few_shot

    return metrics


def main():
    print("=" * 60)
    print("BI-LSTM COMPREHENSIVE BENCHMARK")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")

    # Load data
    data = load_data()

    # Load model with manual weight extraction (Keras 2.15 .keras format compatibility)
    print("\nLoading model...")
    import h5py

    # Rebuild model architecture
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Input, LSTM, Bidirectional, Dense, Dropout,
        BatchNormalization, Masking
    )

    model = Sequential([
        Input(shape=(30, 63), name='landmarks'),
        Masking(mask_value=0.0),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)),
        Bidirectional(LSTM(64, dropout=0.2)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(data['num_classes'], activation='softmax', name='predictions')
    ])

    # Build model
    _ = model(tf.zeros((1, 30, 63)))

    # Extract weights from .keras file (which is a zip)
    temp_dir = PROJECT_ROOT / "temp_extract"
    temp_dir.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "best_model.keras"
    import zipfile
    with zipfile.ZipFile(model_path, 'r') as z:
        z.extract('model.weights.h5', temp_dir)

    weights_path = temp_dir / "model.weights.h5"

    # Manually load weights from HDF5 (handles Keras 2.15 format)
    with h5py.File(weights_path, 'r') as f:
        layer_mapping = {
            'bidirectional': 'bidirectional',
            'bidirectional_1': 'bidirectional_1',
            'dense': 'dense',
            'batch_normalization': 'batch_normalization',
            'dense_1': 'dense_1',
            'dense_2': 'predictions',
        }

        for h5_name, model_name in layer_mapping.items():
            if h5_name not in f['layers']:
                continue

            h5_layer = f['layers'][h5_name]
            model_layer = None
            for layer in model.layers:
                if layer.name == model_name:
                    model_layer = layer
                    break

            if model_layer is None or len(model_layer.get_weights()) == 0:
                continue

            if 'bidirectional' in h5_name:
                fw = h5_layer['forward_layer']['cell']['vars']
                bw = h5_layer['backward_layer']['cell']['vars']
                new_weights = [
                    np.array(fw['0']), np.array(fw['1']), np.array(fw['2']),
                    np.array(bw['0']), np.array(bw['1']), np.array(bw['2']),
                ]
                model_layer.set_weights(new_weights)
            elif 'batch_normalization' in h5_name:
                vars_group = h5_layer['vars']
                new_weights = [np.array(vars_group[str(i)]) for i in range(4)]
                model_layer.set_weights(new_weights)
            elif 'dense' in h5_name:
                vars_group = h5_layer['vars']
                new_weights = [np.array(vars_group['0']), np.array(vars_group['1'])]
                model_layer.set_weights(new_weights)

    print("Weights loaded successfully (manual extraction)")

    total_params = model.count_params()
    print(f"Parameters: {total_params:,}")

    # Run inference
    print("\nRunning inference on test set...")
    X_test = data['X_test']
    y_test = data['y_test']

    # Warm-up
    _ = model.predict(X_test[:10], verbose=0)

    # Timed inference
    start_time = time.time()
    logits = model.predict(X_test, verbose=1)
    inference_time = time.time() - start_time

    y_pred = np.argmax(logits, axis=-1)

    # Compute metrics
    print("\nComputing metrics...")
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
    metrics['test_samples'] = len(X_test)
    metrics['num_classes'] = data['num_classes']

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nModel: {metrics['model']}")
    print(f"Parameters: {metrics['parameters']:,}")
    print(f"Test Samples: {metrics['test_samples']:,}")
    print(f"Classes: {metrics['num_classes']:,}")

    print(f"\n--- Accuracy ---")
    print(f"Top-1:  {metrics['top1_accuracy']:.2f}%")
    print(f"Top-3:  {metrics['top3_accuracy']:.2f}%")
    print(f"Top-5:  {metrics['top5_accuracy']:.2f}%")
    print(f"Top-10: {metrics['top10_accuracy']:.2f}%")

    print(f"\n--- F1 Scores ---")
    print(f"Macro F1:    {metrics['macro_f1']:.2f}%")
    print(f"Weighted F1: {metrics['weighted_f1']:.2f}%")

    print(f"\n--- Per-Class Accuracy ---")
    print(f"Mean: {metrics['per_class_accuracy_mean']:.2f}%")
    print(f"Std:  {metrics['per_class_accuracy_std']:.2f}%")
    print(f"Min:  {metrics['per_class_accuracy_min']:.2f}%")
    print(f"Max:  {metrics['per_class_accuracy_max']:.2f}%")

    print(f"\n--- Efficiency ---")
    print(f"Inference Time: {metrics['inference_time_per_sample_ms']:.2f} ms/sample")
    print(f"Throughput: {metrics['throughput_samples_per_sec']:.1f} samples/sec")

    if 'few_shot' in metrics:
        print(f"\n--- Few-Shot Performance ---")
        for key in sorted(metrics['few_shot'].keys()):
            if key.startswith('accuracy_'):
                bin_name = key.replace('accuracy_', '').replace('_samples', '')
                count_key = f'count_{bin_name}_samples'
                acc = metrics['few_shot'][key]
                count = metrics['few_shot'].get(count_key, 0)
                print(f"  {bin_name:>10} samples: {acc:6.2f}% (n={count})")

    # Save results
    output_dir = PROJECT_ROOT / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"bilstm_benchmark_{timestamp}.json"

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    save_metrics = {k: convert(v) if not isinstance(v, dict) else
                    {k2: convert(v2) for k2, v2 in v.items()}
                    for k, v in metrics.items() if k != 'top_confusions'}

    with open(output_path, 'w') as f:
        json.dump(save_metrics, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
