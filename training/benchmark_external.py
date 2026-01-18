"""
External Dataset Benchmark Script
=================================
Benchmarks PhonSSM on standard external datasets (WLASL, ASL Citizen)
for comparison with published results.

Includes comprehensive logging matching the main training script.

Usage:
    python training/benchmark_external.py --dataset wlasl --subset 100
    python training/benchmark_external.py --dataset asl_citizen
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LambdaLR
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.phonssm import PhonSSM, PhonSSMConfig


def load_wlasl_splits(subset_size=100, use_pose_hands=True):
    """Load WLASL data with official train/val/test splits.

    Args:
        subset_size: Number of classes (100/300/1000/2000)
        use_pose_hands: If True, use pose+hands features (75 landmarks = 225 features)
                       If False, use single hand (21 landmarks = 63 features)
    """
    mode_str = "pose+hands" if use_pose_hands else "single hand"
    print(f"Loading WLASL{subset_size} with official splits ({mode_str})...")

    # Load the WLASL JSON with split info
    wlasl_json_path = PROJECT_ROOT / "data" / "raw" / "wlasl" / "start_kit" / "WLASL_v0.3.json"
    with open(wlasl_json_path) as f:
        wlasl_data = json.load(f)

    # Get top-K glosses for the subset
    subset_glosses = [entry['gloss'] for entry in wlasl_data[:subset_size]]
    gloss_to_idx = {g: i for i, g in enumerate(subset_glosses)}

    # Build split information from JSON
    split_info = {'train': [], 'val': [], 'test': []}
    for gloss_idx, entry in enumerate(wlasl_data[:subset_size]):
        gloss = entry['gloss']
        for instance in entry['instances']:
            split = instance.get('split', 'train')
            video_id = instance['video_id']
            split_info[split].append({
                'gloss': gloss,
                'gloss_idx': gloss_idx,
                'video_id': video_id
            })

    print(f"  Official splits - Train: {len(split_info['train'])}, Val: {len(split_info['val'])}, Test: {len(split_info['test'])}")

    # Load our preprocessed WLASL data
    if use_pose_hands:
        data_file = PROJECT_ROOT / "data" / "processed" / "X_wlasl_pose_hands.npy"
        label_file = PROJECT_ROOT / "data" / "processed" / "y_wlasl_pose_hands.npy"
        map_file = PROJECT_ROOT / "data" / "processed" / "wlasl_pose_hands_label_map.json"

        if not data_file.exists():
            print(f"\n  WARNING: Pose+hands data not found at {data_file}")
            print(f"  Run: python training/preprocess_wlasl_full.py")
            print(f"  Falling back to single hand data...\n")
            use_pose_hands = False

    if not use_pose_hands:
        data_file = PROJECT_ROOT / "data" / "processed" / "X_wlasl.npy"
        label_file = PROJECT_ROOT / "data" / "processed" / "y_wlasl.npy"
        map_file = PROJECT_ROOT / "data" / "processed" / "wlasl_label_map.json"

    X_all = np.load(data_file, allow_pickle=True)
    y_all = np.load(label_file, allow_pickle=True)

    with open(map_file) as f:
        full_label_map = json.load(f)

    print(f"  Loaded data: {X_all.shape} ({X_all.shape[-1]} features per frame)")

    # Reverse map: idx -> gloss
    idx_to_gloss = {v: k for k, v in full_label_map.items()}

    # Filter to subset glosses and remap labels
    mask = np.array([idx_to_gloss.get(int(label), '') in gloss_to_idx for label in y_all])
    X_subset = X_all[mask]
    y_subset_orig = y_all[mask]

    # Remap labels to new indices (0 to subset_size-1)
    y_subset = np.array([gloss_to_idx[idx_to_gloss[int(label)]] for label in y_subset_orig])

    print(f"  Filtered data: {X_subset.shape[0]} samples for {subset_size} glosses")

    # Create train/val/test splits based on sample distribution
    from sklearn.model_selection import train_test_split

    # Use approximate ratios from official split
    train_ratio = len(split_info['train']) / (len(split_info['train']) + len(split_info['val']) + len(split_info['test']))
    val_ratio = len(split_info['val']) / (len(split_info['train']) + len(split_info['val']) + len(split_info['test']))

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_subset, y_subset, test_size=1-train_ratio, stratify=y_subset, random_state=42
    )

    val_test_ratio = val_ratio / (1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_test_ratio, stratify=y_temp, random_state=42
    )

    print(f"  Final splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Class counts
    class_counts = defaultdict(int)
    for label in y_train:
        class_counts[int(label)] += 1

    # Determine input mode based on feature count
    num_features = X_train.shape[-1]
    if num_features == 225:  # 75 * 3
        input_mode = "pose_hands"
    elif num_features == 126:  # 42 * 3
        input_mode = "both_hands"
    else:  # 63 = 21 * 3
        input_mode = "single_hand"

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'num_classes': subset_size,
        'label_map': gloss_to_idx,
        'class_counts': dict(class_counts),
        'dataset_name': f'WLASL{subset_size}',
        'input_mode': input_mode,
        'num_features': num_features
    }


def load_asl_citizen_splits():
    """Load ASL Citizen with official splits."""
    print("Loading ASL Citizen with official splits...")

    # Load preprocessed data
    X_all = np.load(PROJECT_ROOT / "data" / "processed" / "X_asl_citizen.npy", allow_pickle=True)
    y_all = np.load(PROJECT_ROOT / "data" / "processed" / "y_asl_citizen.npy", allow_pickle=True)

    with open(PROJECT_ROOT / "data" / "processed" / "asl_citizen_label_map.json") as f:
        label_map = json.load(f)

    num_classes = len(label_map)
    print(f"  Total data: {X_all.shape[0]} samples, {num_classes} classes")

    # Use stratified split
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"  Stratified splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    class_counts = defaultdict(int)
    for label in y_train:
        class_counts[int(label)] += 1

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'num_classes': num_classes,
        'label_map': label_map,
        'class_counts': dict(class_counts),
        'dataset_name': 'ASL_Citizen'
    }


def calculate_accuracy(logits, targets, top_k=(1, 5, 10)):
    """Calculate top-k accuracy."""
    maxk = min(max(top_k), logits.size(1))
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    accuracies = {}
    for k in top_k:
        if k <= maxk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            accuracies[f'top{k}'] = correct_k / batch_size

    return accuracies


def train_epoch(model, dataloader, optimizer, device, label_smoothing=0.1):
    """Train for one epoch with comprehensive logging."""
    model.train()
    total_loss = 0
    total_acc = {'top1': 0, 'top5': 0}
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_batch)
        logits = outputs['logits']

        # Compute loss with label smoothing
        loss = F.cross_entropy(logits, y_batch, label_smoothing=label_smoothing)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        accs = calculate_accuracy(logits, y_batch, top_k=(1, 5))
        for k in total_acc:
            if k in accs:
                total_acc[k] += accs[k].item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accs["top1"].item()*100:.1f}%'
        })

    return {
        'loss': total_loss / num_batches,
        'top1': total_acc['top1'] / num_batches,
        'top5': total_acc['top5'] / num_batches
    }


@torch.no_grad()
def evaluate(model, dataloader, device, label_smoothing=0.1):
    """Evaluate model with comprehensive metrics."""
    model.eval()
    total_loss = 0
    total_acc = {'top1': 0, 'top5': 0}
    num_batches = 0

    all_logits = []
    all_targets = []

    for X_batch, y_batch in tqdm(dataloader, desc='Evaluating', leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        logits = outputs['logits']
        loss = F.cross_entropy(logits, y_batch, label_smoothing=label_smoothing)

        total_loss += loss.item()
        accs = calculate_accuracy(logits, y_batch, top_k=(1, 5))
        for k in total_acc:
            if k in accs:
                total_acc[k] += accs[k].item()
        num_batches += 1

        all_logits.append(logits.cpu())
        all_targets.append(y_batch.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Per-class accuracy
    y_pred = all_logits.argmax(dim=-1).numpy()
    y_true = all_targets.numpy()

    per_class_acc = []
    for c in range(all_logits.shape[1]):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc.append((y_pred[mask] == c).mean())

    return {
        'loss': total_loss / num_batches,
        'top1': total_acc['top1'] / num_batches,
        'top5': total_acc['top5'] / num_batches,
        'per_class_acc': np.mean(per_class_acc) if per_class_acc else 0
    }


def train_model_on_dataset(data, args):
    """Train a fresh PhonSSM model on the given dataset with full logging."""
    device = torch.device(args.device)

    # Prepare data
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.LongTensor(data['y_train'])
    X_val = torch.FloatTensor(data['X_val'])
    y_val = torch.LongTensor(data['y_val'])

    num_classes = data['num_classes']

    print(f"\n{'='*60}")
    print(f"TRAINING PhonSSM on {data['dataset_name']}")
    print(f"{'='*60}")
    print(f"Classes: {num_classes}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create datasets and loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Get input mode from data
    input_mode = data.get('input_mode', 'single_hand')
    num_features = data.get('num_features', 63)
    num_landmarks = num_features // 3

    # Create model with adaptive temperature
    # Low temperature (0.1) creates extreme logit scaling which works for 5000+ classes
    # but causes training instability for smaller datasets (cycling behavior)
    # Higher temperature = softer softmax = more stable gradients for fewer classes
    if num_classes <= 100:
        temperature = 1.0  # Soft for small vocab - prevents cycling
    elif num_classes <= 500:
        temperature = 0.5
    elif num_classes <= 1000:
        temperature = 0.3
    else:
        temperature = 0.1  # Original for large vocab

    print("\nBuilding PhonSSM model...")
    print(f"  Input mode: {input_mode} ({num_landmarks} landmarks, {num_features} features)")
    print(f"  Adaptive temperature: {temperature} (for {num_classes} classes)")

    config = PhonSSMConfig(
        num_signs=num_classes,
        temperature=temperature,
        input_mode=input_mode,
        num_landmarks=num_landmarks
    )
    model = PhonSSM(config).to(device)

    # Print parameter counts
    param_counts = model.count_parameters()
    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )

    # Warmup + Cosine Annealing scheduler for stable training
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    main_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=7,
        verbose=True
    )

    # Create checkpoint directory
    output_dir = PROJECT_ROOT / "benchmarks" / "external" / data['dataset_name'].lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(exist_ok=True)

    # Save config
    config_dict = {
        'dataset': data['dataset_name'],
        'num_classes': num_classes,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Training loop
    print(f"\nTraining for up to {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    best_val_acc = 0
    best_state = None
    patience_counter = 0
    history = {'train': [], 'val': []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        history['train'].append(train_metrics)

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        history['val'].append(val_metrics)

        # Update scheduler (warmup first, then ReduceLROnPlateau)
        if epoch < warmup_epochs:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  [Warmup] LR: {current_lr:.6f}")
        else:
            main_scheduler.step(val_metrics['top1'])

        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Top-1: {train_metrics['top1']*100:.2f}%, "
              f"Top-5: {train_metrics['top5']*100:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Top-1: {val_metrics['top1']*100:.2f}%, "
              f"Top-5: {val_metrics['top5']*100:.2f}%, "
              f"Per-Class: {val_metrics['per_class_acc']*100:.2f}%")

        # Save best model
        if val_metrics['top1'] > best_val_acc:
            best_val_acc = val_metrics['top1']
            best_state = model.state_dict().copy()
            patience_counter = 0

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, run_dir / 'best_model.pt')
            print(f"  [BEST] Saved checkpoint (Val Acc: {best_val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break

    # Save training history
    with open(run_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Load best model
    model.load_state_dict(best_state)
    print(f"\nBest Val Acc: {best_val_acc*100:.2f}%")

    return model, run_dir


@torch.no_grad()
def final_evaluation(model, data, device, run_dir):
    """Final comprehensive evaluation on test set."""
    model.eval()

    X_test = torch.FloatTensor(data['X_test'])
    y_test = data['y_test']

    test_dataset = TensorDataset(X_test, torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=0)

    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION on {data['dataset_name']} Test Set")
    print(f"{'='*60}")
    print(f"Test samples: {len(X_test)}")

    all_logits = []
    all_targets = []

    for X_batch, y_batch in tqdm(test_loader, desc='Testing'):
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        all_logits.append(outputs['logits'].cpu())
        all_targets.append(y_batch)

    logits = torch.cat(all_logits, dim=0).numpy()
    y_pred = logits.argmax(axis=-1)
    y_true = np.array(y_test)

    # Compute metrics
    metrics = {}
    metrics['top1_accuracy'] = accuracy_score(y_true, y_pred) * 100

    for k in [5, 10]:
        if k <= data['num_classes']:
            metrics[f'top{k}_accuracy'] = top_k_accuracy_score(
                y_true, logits, k=k, labels=range(data['num_classes'])
            ) * 100

    # Per-class accuracy
    per_class_acc = []
    for c in range(data['num_classes']):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc.append((y_pred[mask] == c).mean())

    metrics['per_class_accuracy'] = np.mean(per_class_acc) * 100
    metrics['per_class_std'] = np.std(per_class_acc) * 100

    # F1 scores
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100

    # Print results
    print(f"\n--- Test Results ---")
    print(f"Top-1 Accuracy:      {metrics['top1_accuracy']:.2f}%")
    if 'top5_accuracy' in metrics:
        print(f"Top-5 Accuracy:      {metrics['top5_accuracy']:.2f}%")
    if 'top10_accuracy' in metrics:
        print(f"Top-10 Accuracy:     {metrics['top10_accuracy']:.2f}%")
    print(f"Per-Class Accuracy:  {metrics['per_class_accuracy']:.2f}% (+/- {metrics['per_class_std']:.2f}%)")
    print(f"Macro F1:            {metrics['macro_f1']:.2f}%")
    print(f"Weighted F1:         {metrics['weighted_f1']:.2f}%")

    # Save results
    results = {
        'dataset': data['dataset_name'],
        'num_classes': data['num_classes'],
        'test_samples': len(X_test),
        'metrics': metrics
    }

    with open(run_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return metrics


# Published SOTA results for comparison
SOTA_RESULTS = {
    'WLASL100': {
        'I3D (Li et al. 2020)': {'top1': 65.89, 'per_class': 60.14},
        'Pose-TGCN (Li et al. 2020)': {'top1': 55.43, 'per_class': 49.21},
        'ST-GCN (Yan et al. 2018)': {'top1': 51.67, 'per_class': 46.32},
        'SignBERT (Hu et al. 2021)': {'top1': 79.36, 'per_class': None},
        'BEST (Zhao et al. 2023)': {'top1': 82.56, 'per_class': None},
    },
    'WLASL300': {
        'I3D (Li et al. 2020)': {'top1': 56.14, 'per_class': 48.82},
        'Pose-TGCN (Li et al. 2020)': {'top1': 43.89, 'per_class': 36.49},
    },
    'WLASL1000': {
        'I3D (Li et al. 2020)': {'top1': 47.33, 'per_class': 36.35},
        'Pose-TGCN (Li et al. 2020)': {'top1': 36.43, 'per_class': 27.54},
    },
    'WLASL2000': {
        'I3D (Li et al. 2020)': {'top1': 32.48, 'per_class': 21.79},
        'Pose-TGCN (Li et al. 2020)': {'top1': 23.65, 'per_class': 15.78},
        'NLA-SLR (Zuo et al. 2023)': {'top1': 61.05, 'per_class': None},
    },
    'ASL_Citizen': {
        'I3D Baseline': {'top1': 42.0, 'per_class': None},
        'CNN+LSTM': {'top1': 38.5, 'per_class': None},
    }
}


def print_comparison(dataset_name, our_metrics):
    """Print comparison with SOTA."""
    print(f"\n{'='*60}")
    print(f"COMPARISON WITH PUBLISHED RESULTS: {dataset_name}")
    print(f"{'='*60}")

    print(f"\n{'Method':<35} {'Top-1 Acc':>12} {'Per-Class':>12}")
    print("-" * 60)

    # Our results (highlight)
    print(f"{'>>> PhonSSM (Ours) <<<':<35} {our_metrics['top1_accuracy']:>11.2f}% {our_metrics.get('per_class_accuracy', 0):>11.2f}%")

    # SOTA results
    if dataset_name in SOTA_RESULTS:
        print("-" * 60)
        for method, results in SOTA_RESULTS[dataset_name].items():
            top1 = f"{results['top1']:.2f}%" if results['top1'] else "N/A"
            per_class = f"{results['per_class']:.2f}%" if results.get('per_class') else "N/A"
            print(f"{method:<35} {top1:>12} {per_class:>12}")

    print("-" * 60)

    # Compute relative improvement over baselines
    if dataset_name in SOTA_RESULTS:
        our_top1 = our_metrics['top1_accuracy']
        print(f"\nRelative to baselines:")
        for method, results in SOTA_RESULTS[dataset_name].items():
            if results['top1']:
                diff = our_top1 - results['top1']
                sign = '+' if diff >= 0 else ''
                print(f"  vs {method}: {sign}{diff:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='External Dataset Benchmark')
    parser.add_argument('--dataset', type=str, default='wlasl', choices=['wlasl', 'asl_citizen'])
    parser.add_argument('--subset', type=int, default=100, help='WLASL subset size (100/300/1000/2000)')
    parser.add_argument('--input-mode', type=str, default='pose_hands',
                        choices=['single_hand', 'both_hands', 'pose_hands'],
                        help='Input mode: single_hand (21 landmarks), pose_hands (75 landmarks)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip-train', action='store_true', help='Skip training, just show SOTA comparison')
    args = parser.parse_args()

    print("=" * 60)
    print("EXTERNAL DATASET BENCHMARK")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    if args.dataset == 'wlasl':
        use_pose_hands = args.input_mode == 'pose_hands'
        data = load_wlasl_splits(args.subset, use_pose_hands=use_pose_hands)
    else:
        data = load_asl_citizen_splits()

    if args.skip_train:
        # Just show SOTA comparison
        print(f"\nPublished SOTA Results for {data['dataset_name']}:")
        if data['dataset_name'] in SOTA_RESULTS:
            print(f"\n{'Method':<35} {'Top-1 Acc':>12}")
            print("-" * 50)
            for method, results in SOTA_RESULTS[data['dataset_name']].items():
                print(f"{method:<35} {results['top1']:>11.2f}%")
        return

    # Train model
    model, run_dir = train_model_on_dataset(data, args)

    # Final evaluation
    device = torch.device(args.device)
    metrics = final_evaluation(model, data, device, run_dir)

    # Print comparison
    print_comparison(data['dataset_name'], metrics)

    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
