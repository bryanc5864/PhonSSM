"""
t-SNE Visualization of PhonSSM Embeddings
==========================================
Extracts phonological subspace embeddings and visualizes with t-SNE.
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
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from models.phonssm import PhonSSM, PhonSSMConfig


def load_model_and_data(subset=100):
    """Load PhonSSM model and WLASL test data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find best checkpoint
    if subset == 100:
        ckpt_path = PROJECT_ROOT / "benchmarks/external/wlasl100/20260118_073336/best_model.pt"
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

    idx_to_gloss = {v: k for k, v in full_label_map.items()}

    # Filter to subset glosses
    mask = np.array([idx_to_gloss.get(int(label), '') in gloss_to_idx for label in y_all])
    X_subset = X_all[mask]
    y_subset_orig = y_all[mask]
    y_subset = np.array([gloss_to_idx[idx_to_gloss[int(label)]] for label in y_subset_orig])

    # Sample for visualization (t-SNE is slow for large datasets)
    if len(X_subset) > 3000:
        indices = np.random.choice(len(X_subset), 3000, replace=False)
        X_subset = X_subset[indices]
        y_subset = y_subset[indices]

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

    print(f"Loaded WLASL{subset}: {len(X_subset)} samples for visualization")

    return model, X_subset, y_subset, idx_to_label, device


def extract_embeddings(model, X, device, batch_size=64):
    """Extract phonological component embeddings."""
    model.eval()

    all_embeddings = {
        'handshape': [],
        'location': [],
        'movement': [],
        'orientation': [],
        'final': []  # Final representation before classifier
    }

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
            outputs = model(batch)

            # Extract phonological components
            if 'phonological_components' in outputs:
                for comp_name, comp_tensor in outputs['phonological_components'].items():
                    # Mean pool over time
                    comp_mean = comp_tensor.mean(dim=1)  # (B, D)
                    all_embeddings[comp_name].append(comp_mean.cpu().numpy())

            # Final embedding (before classifier)
            if 'embedding' in outputs:
                all_embeddings['final'].append(outputs['embedding'].cpu().numpy())

    # Concatenate
    for key in all_embeddings:
        if all_embeddings[key]:
            all_embeddings[key] = np.vstack(all_embeddings[key])
        else:
            all_embeddings[key] = None

    return all_embeddings


def run_tsne(embeddings, perplexity=30, max_iter=1000):
    """Run t-SNE on embeddings."""
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, random_state=42)
    return tsne.fit_transform(embeddings)


def plot_tsne(coords, labels, idx_to_label, title, output_path, top_n_classes=20):
    """Plot t-SNE visualization."""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Get most frequent classes for coloring
    unique, counts = np.unique(labels, return_counts=True)
    top_classes = unique[np.argsort(-counts)[:top_n_classes]]

    # Color map
    cmap = plt.colormaps['tab20']
    colors = [cmap(i % 20) for i in range(len(top_classes))]
    class_to_color = {c: colors[i] for i, c in enumerate(top_classes)}

    # Plot other classes in gray first
    other_mask = ~np.isin(labels, top_classes)
    ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
               c='lightgray', alpha=0.3, s=10, label='Other')

    # Plot top classes
    for cls in top_classes:
        mask = labels == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[class_to_color[cls]], alpha=0.7, s=30,
                   label=idx_to_label[cls])

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_component_comparison(embeddings_dict, labels, idx_to_label, output_dir, subset):
    """Plot t-SNE for each phonological component side by side."""
    components = ['handshape', 'location', 'movement', 'orientation']
    available = [c for c in components if embeddings_dict.get(c) is not None]

    if not available:
        print("No phonological component embeddings available")
        return

    n_cols = len(available)
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    # Get top classes
    unique, counts = np.unique(labels, return_counts=True)
    top_classes = unique[np.argsort(-counts)[:10]]
    cmap = plt.colormaps['tab10']
    class_to_color = {c: cmap(i) for i, c in enumerate(top_classes)}

    for idx, comp in enumerate(available):
        print(f"Running t-SNE for {comp}...")
        coords = run_tsne(embeddings_dict[comp], perplexity=min(30, len(labels)//5))

        ax = axes[idx]

        # Plot other classes in gray
        other_mask = ~np.isin(labels, top_classes)
        ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
                   c='lightgray', alpha=0.3, s=10)

        # Plot top classes
        for cls in top_classes:
            mask = labels == cls
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[class_to_color[cls]], alpha=0.7, s=20,
                       label=idx_to_label[cls])

        ax.set_title(f'{comp.capitalize()} Subspace', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # Single legend
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='center right', bbox_to_anchor=(1.15, 0.5))

    plt.suptitle(f'PhonSSM Phonological Subspaces - WLASL{subset}', fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = output_dir / f"tsne_components_wlasl{subset}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, default=100, choices=[100, 2000])
    args = parser.parse_args()

    print("=" * 60)
    print(f"t-SNE VISUALIZATION - WLASL{args.subset}")
    print("=" * 60)

    output_dir = PROJECT_ROOT / "analysis" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    model, X, y, idx_to_label, device = load_model_and_data(args.subset)

    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = extract_embeddings(model, X, device)

    # Plot component comparison
    print("\nCreating component comparison plot...")
    plot_component_comparison(embeddings, y, idx_to_label, output_dir, args.subset)

    # Plot final embedding
    if embeddings['final'] is not None:
        print("\nRunning t-SNE on final embeddings...")
        coords = run_tsne(embeddings['final'])
        plot_tsne(coords, y, idx_to_label,
                  f'PhonSSM Final Embeddings - WLASL{args.subset}',
                  output_dir / f"tsne_final_wlasl{args.subset}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
