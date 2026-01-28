"""
Confusion Matrix Analysis for PhonSSM
=====================================
Generates confusion matrices and identifies most confused sign pairs.
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.phonssm import PhonSSM, PhonSSMConfig


def load_model_and_data(subset=100):
    """Load PhonSSM model and WLASL test data."""
    from sklearn.model_selection import train_test_split

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find best checkpoint
    if subset == 100:
        ckpt_path = PROJECT_ROOT / "benchmarks/external/wlasl100/20260118_073336/best_model.pt"
    elif subset == 1000:
        ckpt_path = PROJECT_ROOT / "benchmarks/external/wlasl1000/20260120_060624/best_model.pt"
    elif subset == 2000:
        ckpt_path = PROJECT_ROOT / "benchmarks/external/wlasl2000/20260119_020829/best_model.pt"
    else:
        raise ValueError(f"No checkpoint for subset {subset}")

    # Load WLASL official data
    wlasl_json = PROJECT_ROOT / "data/raw/wlasl/start_kit/WLASL_v0.3.json"
    with open(wlasl_json) as f:
        wlasl_data = json.load(f)

    # Get subset glosses
    subset_glosses = [entry['gloss'] for entry in wlasl_data[:subset]]
    gloss_to_idx = {g: i for i, g in enumerate(subset_glosses)}
    idx_to_label = {i: g for i, g in enumerate(subset_glosses)}

    # Load pose+hands data
    X_all = np.load(PROJECT_ROOT / "data/processed/X_wlasl_pose_hands.npy")
    y_all = np.load(PROJECT_ROOT / "data/processed/y_wlasl_pose_hands.npy")

    with open(PROJECT_ROOT / "data/processed/wlasl_pose_hands_label_map.json") as f:
        full_label_map = json.load(f)

    # Reverse map: idx -> gloss
    idx_to_gloss = {v: k for k, v in full_label_map.items()}

    # Filter to subset glosses
    mask = np.array([idx_to_gloss.get(int(label), '') in gloss_to_idx for label in y_all])
    X_subset = X_all[mask]
    y_subset_orig = y_all[mask]

    # Remap labels to new indices (0 to subset_size-1)
    y_subset = np.array([gloss_to_idx[idx_to_gloss[int(label)]] for label in y_subset_orig])

    # Split (match benchmark ratios)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_subset, y_subset, test_size=0.32, stratify=y_subset, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.55, stratify=y_temp, random_state=42
    )

    # Create model
    config = PhonSSMConfig(
        num_signs=subset,
        input_mode="pose_hands",
        num_landmarks=75
    )
    model = PhonSSM(config).to(device)

    # Load weights
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Loaded WLASL{subset}: {len(X_test)} test samples, {subset} classes")

    return model, X_test, y_test, idx_to_label, device


def get_predictions(model, X_test, device, batch_size=64):
    """Run inference and get predictions."""
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs['logits'], dim=-1)
            preds = probs.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.array(all_preds), np.vstack(all_probs)


def analyze_confusion(y_true, y_pred, idx_to_label, top_k=20):
    """Analyze confusion matrix and find most confused pairs."""
    cm = confusion_matrix(y_true, y_pred)

    # Find most confused pairs (off-diagonal)
    confused_pairs = []
    n_classes = len(cm)
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    'true': idx_to_label[i],
                    'pred': idx_to_label[j],
                    'count': int(cm[i, j]),
                    'true_total': int(cm[i].sum()),
                    'error_rate': cm[i, j] / max(cm[i].sum(), 1)
                })

    # Sort by count
    confused_pairs.sort(key=lambda x: -x['count'])

    return cm, confused_pairs[:top_k]


def plot_confusion_matrix(cm, idx_to_label, output_path, subset):
    """Plot confusion matrix heatmap."""
    n_classes = len(cm)

    if n_classes <= 100:
        # Full matrix for small datasets
        fig, ax = plt.subplots(figsize=(20, 20))

        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(cm_norm, annot=False, cmap='Blues', ax=ax,
                    xticklabels=[idx_to_label[i] for i in range(n_classes)],
                    yticklabels=[idx_to_label[i] for i in range(n_classes)])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'PhonSSM Confusion Matrix - WLASL{subset}')
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(fontsize=6)
    else:
        # Aggregated view for large datasets
        fig, ax = plt.subplots(figsize=(12, 10))

        # Compute per-class accuracy
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc)

        # Histogram of per-class accuracies
        ax.hist(per_class_acc, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(per_class_acc.mean(), color='red', linestyle='--',
                   label=f'Mean: {per_class_acc.mean():.2%}')
        ax.set_xlabel('Per-Class Accuracy')
        ax.set_ylabel('Number of Classes')
        ax.set_title(f'PhonSSM Per-Class Accuracy Distribution - WLASL{subset}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, default=100, choices=[100, 2000])
    args = parser.parse_args()

    print("=" * 60)
    print(f"CONFUSION MATRIX ANALYSIS - WLASL{args.subset}")
    print("=" * 60)

    # Setup output directory
    output_dir = PROJECT_ROOT / "analysis" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    model, X_test, y_test, idx_to_label, device = load_model_and_data(args.subset)

    # Get predictions
    print("\nRunning inference...")
    y_pred, probs = get_predictions(model, X_test, device)

    # Compute accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"Test Accuracy: {accuracy:.2%}")

    # Analyze confusion
    print("\nAnalyzing confusion matrix...")
    cm, confused_pairs = analyze_confusion(y_test, y_pred, idx_to_label)

    # Print most confused pairs
    print(f"\nTop 20 Most Confused Pairs:")
    print("-" * 60)
    print(f"{'True':<15} {'Predicted':<15} {'Count':<8} {'Error Rate':<10}")
    print("-" * 60)
    for pair in confused_pairs:
        print(f"{pair['true']:<15} {pair['pred']:<15} {pair['count']:<8} {pair['error_rate']:.1%}")

    # Save confusion matrix plot
    plot_path = output_dir / f"confusion_matrix_wlasl{args.subset}.png"
    plot_confusion_matrix(cm, idx_to_label, plot_path, args.subset)

    # Save confused pairs to JSON
    json_path = output_dir / f"confused_pairs_wlasl{args.subset}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'accuracy': float(accuracy),
            'num_classes': len(idx_to_label),
            'test_samples': len(y_test),
            'confused_pairs': confused_pairs
        }, f, indent=2)
    print(f"Saved: {json_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
