"""
Generate enhanced architecture figure for PhonSSM paper.

Improvements over current version:
1. Visual hand skeleton input
2. Graph structure visualization in AGAN
3. Four colored streams in PDM
4. Temporal arrows in BiSSM
5. Prototype hierarchy in HPC
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
from pathlib import Path

# Style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'input': '#6c757d',
    'agan': '#2E86AB',
    'pdm': '#A23B72',
    'bissm': '#F18F01',
    'hpc': '#2E7D32',
    'handshape': '#2E86AB',
    'location': '#5FAD56',
    'movement': '#F18F01',
    'orientation': '#9B51E0',
    'bone': '#555555',
    'joint': '#E63946',
}

# MediaPipe hand connections (simplified for visualization)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),  # Palm
]


def get_hand_landmarks():
    """Generate hand landmarks for visualization."""
    landmarks = np.zeros((21, 2))
    landmarks[0] = [0, 0]
    # Thumb
    landmarks[1] = [-0.3, 0.15]
    landmarks[2] = [-0.45, 0.35]
    landmarks[3] = [-0.55, 0.55]
    landmarks[4] = [-0.6, 0.75]
    # Index
    landmarks[5] = [-0.15, 0.4]
    landmarks[6] = [-0.18, 0.65]
    landmarks[7] = [-0.2, 0.85]
    landmarks[8] = [-0.22, 1.0]
    # Middle
    landmarks[9] = [0, 0.42]
    landmarks[10] = [0, 0.7]
    landmarks[11] = [0, 0.92]
    landmarks[12] = [0, 1.1]
    # Ring
    landmarks[13] = [0.15, 0.4]
    landmarks[14] = [0.17, 0.63]
    landmarks[15] = [0.19, 0.82]
    landmarks[16] = [0.2, 0.97]
    # Pinky
    landmarks[17] = [0.3, 0.35]
    landmarks[18] = [0.32, 0.52]
    landmarks[19] = [0.34, 0.67]
    landmarks[20] = [0.35, 0.8]
    return landmarks


def draw_mini_hand(ax, center, scale=0.15, alpha=0.8):
    """Draw a small hand skeleton at the given center."""
    hand = get_hand_landmarks()
    hand = hand * scale + np.array(center)

    for i, j in HAND_CONNECTIONS:
        ax.plot([hand[i, 0], hand[j, 0]], [hand[i, 1], hand[j, 1]],
                color=COLORS['bone'], linewidth=0.8, alpha=alpha)

    for idx, (x, y) in enumerate(hand):
        size = 3 if idx in [4, 8, 12, 16, 20] else 1.5
        ax.scatter(x, y, c=COLORS['joint'], s=size, zorder=5, alpha=alpha)


def draw_graph_nodes(ax, center, radius=0.12, n_nodes=8):
    """Draw a small graph structure representing anatomical connectivity."""
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    nodes = np.array([[center[0] + radius * np.cos(a),
                       center[1] + radius * np.sin(a)] for a in angles])

    # Draw edges (anatomical connections)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),
             (0, 4), (1, 5), (2, 6)]
    for i, j in edges:
        ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]],
                color='white', linewidth=0.8, alpha=0.6)

    # Draw nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], c='white', s=8, zorder=5, alpha=0.9)


def draw_four_streams(ax, center, width=0.4, height=0.25):
    """Draw four parallel colored streams for PDM."""
    colors = [COLORS['handshape'], COLORS['location'],
              COLORS['movement'], COLORS['orientation']]
    labels = ['H', 'L', 'M', 'O']

    stream_height = height / 5
    gap = height / 20

    for i, (color, label) in enumerate(zip(colors, labels)):
        y = center[1] + height/2 - (i + 0.5) * (stream_height + gap)
        rect = Rectangle((center[0] - width/2, y - stream_height/2),
                         width, stream_height, facecolor=color,
                         edgecolor='white', linewidth=0.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(center[0] - width/2 - 0.03, y, label, fontsize=5,
               ha='right', va='center', color='white', fontweight='bold')


def draw_temporal_arrows(ax, center, width=0.3, height=0.2):
    """Draw bidirectional temporal arrows for BiSSM."""
    # Forward arrow
    ax.annotate('', xy=(center[0] + width/3, center[1] + 0.03),
               xytext=(center[0] - width/3, center[1] + 0.03),
               arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    # Backward arrow
    ax.annotate('', xy=(center[0] - width/3, center[1] - 0.03),
               xytext=(center[0] + width/3, center[1] - 0.03),
               arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    # Time label
    ax.text(center[0], center[1] - 0.08, 't', fontsize=6, ha='center',
           va='top', color='white', style='italic')


def draw_prototype_hierarchy(ax, center, width=0.35, height=0.22):
    """Draw hierarchical prototype structure for HPC."""
    # Component-level prototypes (top row)
    top_y = center[1] + height/4
    for i, color in enumerate([COLORS['handshape'], COLORS['location'],
                               COLORS['movement'], COLORS['orientation']]):
        x = center[0] - width/3 + i * width/4.5
        ax.scatter(x, top_y, c=color, s=15, edgecolors='white',
                  linewidth=0.5, zorder=5)

    # Sign-level prototype (bottom)
    bottom_y = center[1] - height/4
    ax.scatter(center[0], bottom_y, c='white', s=25, marker='s',
              edgecolors='black', linewidth=0.5, zorder=5)

    # Connecting lines
    for i in range(4):
        x = center[0] - width/3 + i * width/4.5
        ax.plot([x, center[0]], [top_y - 0.02, bottom_y + 0.03],
               color='white', linewidth=0.6, alpha=0.6)


def create_enhanced_architecture():
    """Create the enhanced architecture figure."""
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 2.2)
    ax.axis('off')

    # Component specifications: (x_center, width, height, color, label, sublabel)
    components = [
        (0.7, 0.9, 1.4, COLORS['input'], 'Input', '$\\mathbf{X} \\in \\mathbb{R}^{T \\times 21 \\times 3}$'),
        (2.0, 0.9, 1.4, COLORS['agan'], 'AGAN', 'Graph Attention'),
        (3.3, 0.9, 1.4, COLORS['pdm'], 'PDM', 'Disentanglement'),
        (4.6, 0.9, 1.4, COLORS['bissm'], 'BiSSM', 'Temporal SSM'),
        (5.9, 0.9, 1.4, COLORS['hpc'], 'HPC', 'Prototypes'),
    ]

    y_center = 1.1

    for x, w, h, color, label, sublabel in components:
        # Main box
        box = FancyBboxPatch((x - w/2, y_center - h/2), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             facecolor=color, edgecolor='black',
                             linewidth=1, alpha=0.92)
        ax.add_patch(box)

        # Label at top
        ax.text(x, y_center + h/2 - 0.15, label, ha='center', va='top',
               fontsize=9, fontweight='bold', color='white')

        # Sublabel
        ax.text(x, y_center + h/2 - 0.35, sublabel, ha='center', va='top',
               fontsize=6, color='white', alpha=0.9)

    # Add visual elements inside each box

    # Input: hand skeleton
    draw_mini_hand(ax, (0.7, 0.85), scale=0.18, alpha=0.9)

    # AGAN: graph structure
    draw_graph_nodes(ax, (2.0, 0.85), radius=0.18, n_nodes=8)

    # PDM: four streams
    draw_four_streams(ax, (3.3, 0.8), width=0.55, height=0.45)

    # BiSSM: temporal arrows
    draw_temporal_arrows(ax, (4.6, 0.85), width=0.5)

    # HPC: prototype hierarchy
    draw_prototype_hierarchy(ax, (5.9, 0.85), width=0.5, height=0.4)

    # Arrows between components
    arrow_y = y_center
    arrow_props = dict(arrowstyle='->', color='black', lw=1.5,
                       connectionstyle='arc3,rad=0')

    for i in range(len(components) - 1):
        x1 = components[i][0] + components[i][1]/2 + 0.02
        x2 = components[i+1][0] - components[i+1][1]/2 - 0.02
        ax.annotate('', xy=(x2, arrow_y), xytext=(x1, arrow_y),
                   arrowprops=arrow_props)

    # Output arrow and label
    ax.annotate('', xy=(6.7, arrow_y), xytext=(6.35, arrow_y),
               arrowprops=arrow_props)
    ax.text(6.85, arrow_y, 'ŷ', fontsize=11, ha='left', va='center',
           fontweight='bold')

    # Parameter count
    ax.text(6.8, 0.15, '3.2M params', fontsize=7, ha='right',
           style='italic', color='gray')

    # Stage labels at bottom
    stage_labels = ['Spatial\nEncoding', 'Phonological\nDecomposition',
                   'Temporal\nModeling', 'Metric\nClassification']
    stage_x = [1.35, 2.65, 4.0, 5.25]

    for x, label in zip(stage_x, stage_labels):
        ax.text(x, 0.08, label, ha='center', va='bottom', fontsize=6,
               color='gray', linespacing=0.9)

    # Bracket for stages 1-2
    ax.annotate('', xy=(0.7, 0.25), xytext=(2.0, 0.25),
               arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.savefig('figures/fig_architecture_enhanced.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig('figures/fig_architecture_enhanced.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close()
    print("Enhanced architecture figure saved!")


def create_vertical_architecture():
    """
    Create a vertical (column-width) version of the architecture.
    Better for single-column placement in the paper.
    """
    fig, ax = plt.subplots(figsize=(3.3, 4.5))
    ax.set_xlim(0, 3.3)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    # Vertical layout
    components = [
        (4.1, COLORS['input'], 'Input', '$\\mathbf{X} \\in \\mathbb{R}^{T \\times 21 \\times 3}$'),
        (3.3, COLORS['agan'], 'AGAN', 'Anatomical Graph Attention'),
        (2.5, COLORS['pdm'], 'PDM', 'Phonological Disentanglement'),
        (1.7, COLORS['bissm'], 'BiSSM', 'Bidirectional State Space'),
        (0.9, COLORS['hpc'], 'HPC', 'Hierarchical Prototypes'),
    ]

    x_center = 1.65
    box_w, box_h = 2.6, 0.55

    for y, color, label, sublabel in components:
        box = FancyBboxPatch((x_center - box_w/2, y - box_h/2), box_w, box_h,
                             boxstyle="round,pad=0.02,rounding_size=0.04",
                             facecolor=color, edgecolor='black',
                             linewidth=1, alpha=0.92)
        ax.add_patch(box)

        ax.text(x_center, y + 0.08, label, ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
        ax.text(x_center, y - 0.12, sublabel, ha='center', va='center',
               fontsize=6, color='white', alpha=0.9)

    # Arrows
    for i in range(len(components) - 1):
        y1 = components[i][0] - box_h/2 - 0.02
        y2 = components[i+1][0] + box_h/2 + 0.02
        ax.annotate('', xy=(x_center, y2), xytext=(x_center, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.3))

    # Output
    ax.annotate('', xy=(x_center, 0.25), xytext=(x_center, 0.62),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.3))
    ax.text(x_center, 0.15, 'Prediction: ŷ', ha='center', fontsize=8)

    # Parameter count
    ax.text(3.1, 0.1, '3.2M params', fontsize=6, ha='right',
           style='italic', color='gray')

    # Add mini visuals to the right of each box
    # (keeping it clean - these are optional)

    plt.savefig('figures/fig_architecture_vertical.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig('figures/fig_architecture_vertical.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close()
    print("Vertical architecture figure saved!")


if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent.parent)  # Change to paper/ directory

    print("=" * 60)
    print("Generating Enhanced Architecture Figures for PhonSSM Paper")
    print("=" * 60)

    Path('figures').mkdir(exist_ok=True)

    print("\n1. Creating horizontal (full-width) architecture...")
    create_enhanced_architecture()

    print("\n2. Creating vertical (column-width) architecture...")
    create_vertical_architecture()

    print("\n" + "=" * 60)
    print("Done! Generated figures:")
    print("  - figures/fig_architecture_enhanced.pdf (horizontal)")
    print("  - figures/fig_architecture_vertical.pdf (vertical)")
    print("=" * 60)
