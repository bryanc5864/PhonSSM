"""
AGAN Attention Heatmap Visualization
=====================================
Extracts graph attention weights and visualizes on skeleton.
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
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split

from models.phonssm import PhonSSM, PhonSSMConfig


# Landmark indices for pose+hands (75 total)
POSE_LANDMARKS = list(range(33))  # MediaPipe pose
LEFT_HAND_LANDMARKS = list(range(33, 54))  # 21 landmarks
RIGHT_HAND_LANDMARKS = list(range(54, 75))  # 21 landmarks

# Body part groupings
BODY_PARTS = {
    'face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Face landmarks
    'torso': [11, 12, 23, 24],  # Shoulders and hips
    'left_arm': [13, 15, 17, 19, 21],  # Left arm
    'right_arm': [14, 16, 18, 20, 22],  # Right arm
    'left_hand': LEFT_HAND_LANDMARKS,
    'right_hand': RIGHT_HAND_LANDMARKS,
}

# Hand landmark names
HAND_LANDMARKS = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_mcp', 'index_pip', 'index_dip', 'index_tip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
    'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
]


def load_model_and_data(subset=100):
    """Load PhonSSM model and sample data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if subset == 100:
        ckpt_path = PROJECT_ROOT / "benchmarks/external/wlasl100/20260118_073336/best_model.pt"
    else:
        ckpt_path = PROJECT_ROOT / "benchmarks/external/wlasl2000/20260119_020829/best_model.pt"

    # Load data
    wlasl_json = PROJECT_ROOT / "data/raw/wlasl/start_kit/WLASL_v0.3.json"
    with open(wlasl_json) as f:
        wlasl_data = json.load(f)

    subset_glosses = [entry['gloss'] for entry in wlasl_data[:subset]]
    gloss_to_idx = {g: i for i, g in enumerate(subset_glosses)}
    idx_to_label = {i: g for i, g in enumerate(subset_glosses)}

    X_all = np.load(PROJECT_ROOT / "data/processed/X_wlasl_pose_hands.npy")
    y_all = np.load(PROJECT_ROOT / "data/processed/y_wlasl_pose_hands.npy")

    with open(PROJECT_ROOT / "data/processed/wlasl_pose_hands_label_map.json") as f:
        full_label_map = json.load(f)

    idx_to_gloss = {v: k for k, v in full_label_map.items()}

    mask = np.array([idx_to_gloss.get(int(label), '') in gloss_to_idx for label in y_all])
    X_subset = X_all[mask]
    y_subset = np.array([gloss_to_idx[idx_to_gloss[int(label)]] for label in y_all[mask]])

    # Create model
    config = PhonSSMConfig(
        num_signs=subset,
        input_mode="pose_hands",
        num_landmarks=75
    )
    model = PhonSSM(config).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, X_subset, y_subset, idx_to_label, device


def extract_attention_weights(model, x, device):
    """Extract attention weights from AGAN module."""
    model.eval()

    # Hook to capture attention weights
    attention_weights = []

    def hook_fn(module, input, output):
        # GAT attention is in the output
        if hasattr(module, 'att_weights'):
            attention_weights.append(module.att_weights.detach().cpu())

    # Register hooks on attention layers
    hooks = []
    for name, module in model.named_modules():
        if 'agan' in name.lower() and hasattr(module, 'attention'):
            hooks.append(module.attention.register_forward_hook(hook_fn))

    # Forward pass
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
        outputs = model(x_tensor)

    # Remove hooks
    for h in hooks:
        h.remove()

    # If no attention weights captured, compute from model directly
    if not attention_weights:
        # Extract from AGAN module manually
        attention_weights = extract_agan_attention(model, x, device)

    return attention_weights, outputs


def extract_agan_attention(model, x, device):
    """Extract attention from AGAN by analyzing intermediate outputs."""
    model.eval()

    # Get AGAN module
    agan = model.agan if hasattr(model, 'agan') else None
    if agan is None:
        return None

    with torch.no_grad():
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)

        # Reshape for AGAN: (B, T, 225) -> (B, T, 75, 3)
        B, T, F = x_tensor.shape
        x_reshaped = x_tensor.view(B, T, 75, 3)

        # Get attention scores if available
        if hasattr(agan, 'get_attention_weights'):
            return agan.get_attention_weights(x_reshaped)

        # Otherwise, compute node importance from output gradients
        x_reshaped.requires_grad_(True)
        out = agan(x_reshaped)

        # Compute importance as gradient magnitude
        importance = torch.zeros(75)
        for i in range(75):
            if out.grad_fn is not None:
                grad = torch.autograd.grad(out.sum(), x_reshaped, retain_graph=True)[0]
                importance[i] = grad[:, :, i, :].abs().mean()

        return importance.numpy()

    return None


def compute_landmark_importance(model, X_samples, y_samples, device, n_samples=100):
    """Compute average landmark importance across samples."""
    model.eval()

    # Sample data
    if len(X_samples) > n_samples:
        indices = np.random.choice(len(X_samples), n_samples, replace=False)
        X_samples = X_samples[indices]
        y_samples = y_samples[indices]

    importance_scores = np.zeros(75)
    count = 0

    for i in range(len(X_samples)):
        x = X_samples[i]
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
        x_tensor.requires_grad_(True)

        with torch.enable_grad():
            outputs = model(x_tensor)
            logits = outputs['logits']

            # Get gradient w.r.t. predicted class
            pred_class = logits.argmax(dim=-1)
            loss = logits[0, pred_class]
            loss.backward()

            # Compute importance from gradients
            grad = x_tensor.grad.detach().cpu().numpy()[0]  # (30, 225)
            grad = grad.reshape(30, 75, 3)  # (30, 75, 3)

            # Average over time and coordinates
            sample_importance = np.abs(grad).mean(axis=(0, 2))  # (75,)
            importance_scores += sample_importance
            count += 1

        model.zero_grad()

    importance_scores /= count
    return importance_scores


def plot_skeleton_heatmap(importance, output_path, title="Landmark Importance"):
    """Plot importance scores on a simplified skeleton diagram."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Normalize importance
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

    # Body part importance
    body_importance = {}
    for part, indices in BODY_PARTS.items():
        valid_indices = [i for i in indices if i < len(importance)]
        if valid_indices:
            body_importance[part] = importance[valid_indices].mean()

    # Plot 1: Body part bar chart
    ax1 = axes[0]
    parts = list(body_importance.keys())
    values = [body_importance[p] for p in parts]
    colors = plt.colormaps['RdYlBu_r'](values)
    ax1.barh(parts, values, color=colors)
    ax1.set_xlabel('Importance')
    ax1.set_title('Body Part Importance')
    ax1.set_xlim(0, 1)

    # Plot 2: Right hand detail
    ax2 = axes[1]
    right_hand_imp = importance[RIGHT_HAND_LANDMARKS]
    finger_groups = {
        'Thumb': [1, 2, 3, 4],
        'Index': [5, 6, 7, 8],
        'Middle': [9, 10, 11, 12],
        'Ring': [13, 14, 15, 16],
        'Pinky': [17, 18, 19, 20],
        'Wrist': [0]
    }
    finger_imp = {k: right_hand_imp[v].mean() for k, v in finger_groups.items()}
    fingers = list(finger_imp.keys())
    values = [finger_imp[f] for f in fingers]
    colors = plt.colormaps['RdYlBu_r'](values)
    ax2.barh(fingers, values, color=colors)
    ax2.set_xlabel('Importance')
    ax2.set_title('Right Hand - Finger Importance')
    ax2.set_xlim(0, 1)

    # Plot 3: Left hand detail
    ax3 = axes[2]
    left_hand_imp = importance[LEFT_HAND_LANDMARKS]
    finger_imp = {k: left_hand_imp[v].mean() for k, v in finger_groups.items()}
    fingers = list(finger_imp.keys())
    values = [finger_imp[f] for f in fingers]
    colors = plt.colormaps['RdYlBu_r'](values)
    ax3.barh(fingers, values, color=colors)
    ax3.set_xlabel('Importance')
    ax3.set_title('Left Hand - Finger Importance')
    ax3.set_xlim(0, 1)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_class_attention(importance_by_class, idx_to_label, output_path, top_n=10):
    """Plot attention patterns for different sign classes."""
    # Select diverse classes
    classes = list(importance_by_class.keys())[:top_n]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for idx, cls in enumerate(classes):
        ax = axes[idx]
        imp = importance_by_class[cls]

        # Body part aggregation
        body_imp = {}
        for part, indices in BODY_PARTS.items():
            valid_indices = [i for i in indices if i < len(imp)]
            if valid_indices:
                body_imp[part] = imp[valid_indices].mean()

        parts = list(body_imp.keys())
        values = [body_imp[p] for p in parts]

        # Normalize
        values = np.array(values)
        values = (values - values.min()) / (values.max() - values.min() + 1e-8)

        colors = plt.colormaps['RdYlBu_r'](values)
        ax.barh(parts, values, color=colors)
        ax.set_title(idx_to_label[cls], fontsize=10)
        ax.set_xlim(0, 1)
        ax.tick_params(axis='y', labelsize=8)

    plt.suptitle('Attention Patterns by Sign Class', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print(f"ATTENTION HEATMAP ANALYSIS - WLASL{args.subset}")
    print("=" * 60)

    output_dir = PROJECT_ROOT / "analysis" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    model, X, y, idx_to_label, device = load_model_and_data(args.subset)

    # Compute overall landmark importance
    print("\nComputing landmark importance (this may take a moment)...")
    importance = compute_landmark_importance(model, X, y, device, n_samples=200)

    # Plot overall heatmap
    plot_skeleton_heatmap(
        importance,
        output_dir / f"attention_heatmap_wlasl{args.subset}.png",
        f"PhonSSM Attention - WLASL{args.subset}"
    )

    # Compute per-class importance
    print("\nComputing per-class attention patterns...")
    importance_by_class = {}
    unique_classes = np.unique(y)[:20]  # Top 20 classes

    for cls in unique_classes:
        mask = y == cls
        X_cls = X[mask]
        y_cls = y[mask]
        if len(X_cls) >= 5:
            importance_by_class[cls] = compute_landmark_importance(
                model, X_cls, y_cls, device, n_samples=min(20, len(X_cls))
            )

    # Plot per-class attention
    if importance_by_class:
        plot_per_class_attention(
            importance_by_class, idx_to_label,
            output_dir / f"attention_per_class_wlasl{args.subset}.png"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
