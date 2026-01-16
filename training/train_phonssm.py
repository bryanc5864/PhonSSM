"""
PhonSSM Training Script
=======================
Train the Phonology-Aware State Space Model for sign language recognition.

Architecture:
    AGAN → PDM → BiSSM → HPC
    Input: (30 frames, 21 landmarks, 3 coords)
    Output: 5565 sign classes

Target metrics:
    - Top-1 accuracy: >40% (vs 28% baseline)
    - Top-5 accuracy: >70%
    - Parameters: ~2-3M

Usage:
    python training/train_phonssm.py
    python training/train_phonssm.py --epochs 100 --batch-size 32
    python training/train_phonssm.py --device cuda
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.phonssm import PhonSSM, PhonSSMConfig

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "merged"
MODEL_DIR = PROJECT_ROOT / "models" / "phonssm" / "checkpoints"


def load_data(data_dir: Path):
    """Load and prepare data for PyTorch."""
    print("Loading data...")

    X_train = np.load(data_dir / "X_train.npy", allow_pickle=True)
    y_train = np.load(data_dir / "y_train.npy", allow_pickle=True)
    X_val = np.load(data_dir / "X_val.npy", allow_pickle=True)
    y_val = np.load(data_dir / "y_val.npy", allow_pickle=True)
    X_test = np.load(data_dir / "X_test.npy", allow_pickle=True)
    y_test = np.load(data_dir / "y_test.npy", allow_pickle=True)

    with open(data_dir / "label_map.json") as f:
        label_map = json.load(f)

    num_classes = len(label_map)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Classes: {num_classes}")

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, label_map


def calculate_accuracy(logits, targets, top_k=(1, 3, 5)):
    """Calculate top-k accuracy."""
    maxk = max(top_k)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    accuracies = {}
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        accuracies[f'top{k}'] = correct_k / batch_size

    return accuracies


def train_epoch(model, dataloader, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = {'top1': 0, 'top3': 0, 'top5': 0}
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_batch)

        # Compute loss
        losses = model.compute_loss(outputs, y_batch, config.label_smoothing)
        loss = losses['total']

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metrics
        total_loss += loss.item()
        accs = calculate_accuracy(outputs['logits'], y_batch)
        for k in total_acc:
            total_acc[k] += accs[k].item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accs["top1"].item()*100:.1f}%'
        })

    return {
        'loss': total_loss / num_batches,
        'top1': total_acc['top1'] / num_batches,
        'top3': total_acc['top3'] / num_batches,
        'top5': total_acc['top5'] / num_batches
    }


@torch.no_grad()
def evaluate(model, dataloader, device, config):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_acc = {'top1': 0, 'top3': 0, 'top5': 0}
    num_batches = 0

    for X_batch, y_batch in tqdm(dataloader, desc='Evaluating', leave=False):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        losses = model.compute_loss(outputs, y_batch, config.label_smoothing)

        total_loss += losses['total'].item()
        accs = calculate_accuracy(outputs['logits'], y_batch)
        for k in total_acc:
            total_acc[k] += accs[k].item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'top1': total_acc['top1'] / num_batches,
        'top3': total_acc['top3'] / num_batches,
        'top5': total_acc['top5'] / num_batches
    }


def train(args):
    """Main training function."""
    print("=" * 60)
    print("PHONSSM TRAINING")
    print("=" * 60)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, label_map = load_data(
        Path(args.data_dir)
    )

    # Create config
    config = PhonSSMConfig(
        num_signs=num_classes,
        num_frames=X_train.shape[1],  # 30
        dropout=args.dropout,
        label_smoothing=args.label_smoothing
    )

    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create model
    print("\nBuilding PhonSSM model...")
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
        weight_decay=args.weight_decay
    )

    # Scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2
        )

    # Create checkpoint directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = MODEL_DIR / timestamp
    run_dir.mkdir(exist_ok=True)

    # Save config
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(vars(config), f, indent=2)

    # Training loop
    print(f"\nTraining for up to {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    best_val_acc = 0
    patience_counter = 0
    history = {'train': [], 'val': []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, config)
        history['train'].append(train_metrics)

        # Validate
        val_metrics = evaluate(model, val_loader, device, config)
        history['val'].append(val_metrics)

        # Update scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_metrics['top1'])
        else:
            scheduler.step()

        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Top-1: {train_metrics['top1']*100:.2f}%, "
              f"Top-5: {train_metrics['top5']*100:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Top-1: {val_metrics['top1']*100:.2f}%, "
              f"Top-5: {val_metrics['top5']*100:.2f}%")

        # Save best model
        if val_metrics['top1'] > best_val_acc:
            best_val_acc = val_metrics['top1']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': vars(config)
            }, run_dir / 'best_model.pt')
            print(f"New best model saved! Val accuracy: {best_val_acc*100:.2f}%")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['top1'],
                'config': vars(config)
            }, run_dir / f'checkpoint_epoch_{epoch+1}.pt')

    # Load best model for evaluation
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    checkpoint = torch.load(run_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device, config)
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Top-1 Accuracy: {test_metrics['top1']*100:.2f}%")
    print(f"  Top-3 Accuracy: {test_metrics['top3']*100:.2f}%")
    print(f"  Top-5 Accuracy: {test_metrics['top5']*100:.2f}%")

    # Save final results
    results = {
        'test_metrics': test_metrics,
        'best_val_acc': best_val_acc,
        'epochs_trained': len(history['train']),
        'parameters': param_counts,
        'label_map': label_map
    }
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, torch.Tensor)) else x)

    # Save training history
    with open(run_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2, default=lambda x: float(x))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Test accuracy: {test_metrics['top1']*100:.2f}%")
    print(f"Model saved to: {run_dir}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PhonSSM')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                        help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay for AdamW')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['plateau', 'cosine'],
                        help='Learning rate scheduler')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers (0 for Windows)')

    args = parser.parse_args()
    train(args)
