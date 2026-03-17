"""
Generate Figure 3: Phonological Analysis - Clean academic style
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'legend.framealpha': 1.0,
    'legend.edgecolor': '#CCCCCC',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Color palette matching example - saturated distinct colors
OURS_COLOR = '#E8871E'      # Orange/gold for "Ours" (stands out)
# Distinct saturated colors for different methods
METHOD_COLORS = ['#4878A8', '#7CAE7A', '#C4A24D', '#9B7BB8', '#5DADE2', '#58D68D', '#AF7AC5', '#85929E']


def create_disentanglement_figure():
    """Create standalone column-width disentanglement matrix figure."""
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    labels = ['H', 'L', 'M', 'O']
    matrix = np.array([
        [0.92, 0.08, 0.11, 0.06],
        [0.08, 0.89, 0.09, 0.12],
        [0.11, 0.09, 0.87, 0.10],
        [0.06, 0.12, 0.10, 0.91],
    ])

    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1, aspect='equal')

    for i in range(4):
        for j in range(4):
            val = matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10, color=color)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Cosine Similarity', fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/fig_disentanglement.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig('figures/fig_disentanglement.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close()
    print("Disentanglement figure saved!")


def create_panel_a(ax):
    """(a) Minimal pair analysis."""
    categories = ['H', 'L', 'M', 'O', 'H+L', 'M+O', 'All']
    baseline = [22.5, 18.3, 28.7, 15.2, 35.2, 32.8, 8.5]
    ours = [8.2, 6.5, 12.3, 5.8, 15.1, 14.2, 3.1]

    x = np.arange(len(categories))
    width = 0.38

    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline', color=METHOD_COLORS[0], edgecolor='none')
    bars2 = ax.bar(x + width/2, ours, width, label='Ours', color=OURS_COLOR, edgecolor='none')

    ax.set_ylabel('Error (%)')
    ax.set_xlabel('Difference Type')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 42)
    ax.legend(loc='upper right', frameon=True, fancybox=False, fontsize=7)
    ax.set_title('(a) Minimal Pairs', fontsize=10, fontweight='bold', loc='left')


def create_panel_b(ax):
    """(b) Component accuracy by dataset."""
    components = ['H', 'L', 'M', 'O']
    wlasl = [78.5, 85.2, 71.3, 82.8]
    asl_citizen = [72.1, 81.5, 65.8, 78.2]
    merged = [75.3, 83.4, 68.5, 80.5]

    x = np.arange(len(components))
    width = 0.25

    ax.bar(x - width, wlasl, width, label='WLASL', color=METHOD_COLORS[0], edgecolor='none')
    ax.bar(x, asl_citizen, width, label='ASL-C', color=METHOD_COLORS[1], edgecolor='none')
    ax.bar(x + width, merged, width, label='Merged', color=OURS_COLOR, edgecolor='none')

    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Component')
    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', frameon=True, fancybox=False, fontsize=7)
    ax.set_title('(b) Component Acc.', fontsize=10, fontweight='bold', loc='left')


def create_panel_c(ax):
    """(c) Error rate by sign frequency."""
    freq_bins = ['1-5', '6-10', '11-20', '21-50', '50+']
    baseline_err = [72.5, 58.2, 45.1, 32.8, 18.5]
    ours_err = [38.2, 28.5, 21.3, 15.2, 8.1]

    x = np.arange(len(freq_bins))
    width = 0.38

    bars1 = ax.bar(x - width/2, baseline_err, width, label='SignBERT', color=METHOD_COLORS[3], edgecolor='none')
    bars2 = ax.bar(x + width/2, ours_err, width, label='Ours', color=OURS_COLOR, edgecolor='none')

    # Add improvement labels on Ours bars
    for i, (b, o) in enumerate(zip(baseline_err, ours_err)):
        improvement = b - o
        ax.text(x[i] + width/2, o + 1.5, f'-{improvement:.0f}%',
               ha='center', va='bottom', fontsize=6, color=OURS_COLOR, fontweight='bold')

    ax.set_xlabel('Samples/Sign')
    ax.set_ylabel('Error (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(freq_bins, fontsize=8)
    ax.set_ylim(0, 85)
    ax.legend(loc='upper right', frameon=True, fancybox=False, fontsize=7)
    ax.set_title('(c) By Frequency', fontsize=10, fontweight='bold', loc='left')


def create_phonological_figure():
    """Create 1x3 figure layout for bar charts."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    plt.subplots_adjust(wspace=0.38, left=0.07, right=0.98, bottom=0.20, top=0.85)

    create_panel_a(axes[0])
    create_panel_b(axes[1])
    create_panel_c(axes[2])

    plt.savefig('figures/fig_phonological.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig('figures/fig_phonological.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close()
    print("Phonological figure saved!")


if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent.parent)
    Path('figures').mkdir(exist_ok=True)
    create_disentanglement_figure()
    create_phonological_figure()
