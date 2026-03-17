"""
Generate Figure 2: Main Results - Clean academic style
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 8,
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
BASELINE_COLOR = '#888888'  # Gray for baselines
# Distinct saturated colors for different methods
METHOD_COLORS = ['#4878A8', '#7CAE7A', '#C4A24D', '#9B7BB8', '#5DADE2', '#58D68D', '#AF7AC5', '#85929E']


def create_panel_a(ax):
    """(a) Benchmark comparison."""
    methods = ['I3D', 'ST-GCN', 'SignBERT', 'SAM-SLR', 'DSTA-SLR', 'Ours']
    values = [32.5, 35.8, 47.5, 51.5, 53.7, 72.1]
    colors = [METHOD_COLORS[i % len(METHOD_COLORS)] for i in range(len(methods)-1)] + [OURS_COLOR]

    x = np.arange(len(methods))
    bars = ax.bar(x, values, color=colors, edgecolor='none', width=0.7)

    # Add value labels on top (black, evenly spaced)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val:.1f}', ha='center', va='bottom', fontsize=6, fontweight='normal', color='black')

    ax.set_ylabel('Top-1 Acc. (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
    ax.set_ylim(0, 82)
    ax.set_title('(a) WLASL-2000', fontsize=9, fontweight='bold', loc='left')


def create_panel_b(ax):
    """(b) Scaling across splits."""
    splits = ['100', '300', '1000', '2000']
    methods_data = {
        'SignBERT': [79.1, 70.4, 55.0, 47.5],
        'SAM-SLR': [77.9, 67.7, 51.5, 51.5],
        'DSTA-SLR': [82.4, 80.0, 67.8, 53.7],
        'Ours': [88.4, 74.4, 62.9, 72.1],
    }
    markers = ['s', '^', 'd', 'o']
    colors_line = [BASELINE_COLOR, BASELINE_COLOR, BASELINE_COLOR, OURS_COLOR]
    linestyles = ['--', '--', '--', '-']

    x = np.arange(len(splits))
    for (name, vals), marker, color, ls in zip(methods_data.items(), markers, colors_line, linestyles):
        lw = 2.0 if name == 'Ours' else 1.2
        ms = 6 if name == 'Ours' else 4
        ax.plot(x, vals, marker=marker, color=color, linewidth=lw, linestyle=ls,
               markersize=ms, label=name, markeredgecolor='white', markeredgewidth=0.8)

    ax.set_xlabel('WLASL Split')
    ax.set_ylabel('Top-1 Acc. (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(25, 95)
    ax.legend(loc='lower left', frameon=True, fancybox=False, fontsize=6)
    ax.set_title('(b) Vocabulary Scaling', fontsize=9, fontweight='bold', loc='left')


def create_panel_c(ax):
    """(c) Few-shot learning - real data from paper Table 2."""
    # Real data: performance by training samples per sign
    sample_bins = ['1-5', '6-10', '11-20', '21-50', '51-100', '101+']
    bilstm = [4.08, 3.70, 5.29, 10.13, 1.52, 52.66]
    ours = [13.27, 9.26, 12.71, 24.03, 26.03, 92.82]

    x = np.arange(len(sample_bins))
    width = 0.38

    bars1 = ax.bar(x - width/2, bilstm, width, label='Bi-LSTM', color=METHOD_COLORS[0], edgecolor='none')
    bars2 = ax.bar(x + width/2, ours, width, label='Ours', color=OURS_COLOR, edgecolor='none')

    # Add value labels on top (black, evenly spaced)
    for bar, val in zip(bars1, bilstm):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val:.1f}', ha='center', va='bottom', fontsize=5, color='black')
    for bar, val in zip(bars2, ours):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val:.1f}', ha='center', va='bottom', fontsize=5, color='black')

    ax.set_xlabel('Samples/Sign')
    ax.set_ylabel('Top-1 Acc. (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(sample_bins, fontsize=7)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper left', frameon=True, fancybox=False, fontsize=6)
    ax.set_title('(c) Few-Shot', fontsize=9, fontweight='bold', loc='left')


def create_panel_d(ax):
    """(d) Ablation study on WLASL100."""
    # Real ablation results from paper (WLASL100)
    components = ['Full', '-Ortho', '-BiSSM', '-AGAN', '-PDM']
    values = [88.37, 85.92, 82.17, 79.84, 76.49]
    colors = [OURS_COLOR] + [METHOD_COLORS[i] for i in range(4)]

    x = np.arange(len(components))
    bars = ax.bar(x, values, color=colors, edgecolor='none', width=0.65)

    # Add value labels on top (black, evenly spaced)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val:.1f}', ha='center', va='bottom', fontsize=6, color='black')

    ax.set_ylabel('Top-1 Acc. (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=45, ha='right', fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_title('(d) Ablation', fontsize=9, fontweight='bold', loc='left')


def create_results_figure():
    """Create 2x2 figure."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2))
    plt.subplots_adjust(wspace=0.32, hspace=0.50, left=0.08, right=0.98, bottom=0.12, top=0.94)

    create_panel_a(axes[0, 0])
    create_panel_b(axes[0, 1])
    create_panel_c(axes[1, 0])
    create_panel_d(axes[1, 1])

    plt.savefig('figures/fig_results.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig('figures/fig_results.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close()
    print("Results figure saved!")


if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent.parent)
    Path('figures').mkdir(exist_ok=True)
    create_results_figure()
