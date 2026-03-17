"""
Generate components for the Motivation Figure (New Figure 1) for PhonSSM paper.

This script generates:
1. Hand skeleton visualizations showing minimal pairs
2. Embedding space visualizations (entangled vs disentangled)
3. Individual components to be composited in Figma

For hybrid approach: Python generates data-driven elements, Figma composes final layout.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
from pathlib import Path

# Style configuration matching paper
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'phonssm': '#2E86AB',
    'baseline': '#5C6670',
    'handshape': '#2E86AB',    # Blue
    'location': '#5FAD56',      # Green
    'movement': '#F18F01',      # Orange
    'orientation': '#9B51E0',   # Purple
    'accent': '#A23B72',
    'skeleton': '#333333',
    'joint': '#E63946',
}

# MediaPipe hand landmark connections (21 landmarks)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]

def get_hand_landmarks_flat():
    """
    Generate a realistic flat hand (like ASL 'B' handshape).
    Returns 21 (x, y) coordinates for MediaPipe hand landmarks.
    """
    landmarks = np.zeros((21, 2))

    # Wrist at origin
    landmarks[0] = [0, 0]

    # Thumb (angled out)
    landmarks[1] = [-0.3, 0.15]
    landmarks[2] = [-0.45, 0.35]
    landmarks[3] = [-0.55, 0.55]
    landmarks[4] = [-0.6, 0.75]

    # Index finger
    landmarks[5] = [-0.15, 0.4]
    landmarks[6] = [-0.18, 0.65]
    landmarks[7] = [-0.2, 0.85]
    landmarks[8] = [-0.22, 1.0]

    # Middle finger (longest)
    landmarks[9] = [0, 0.42]
    landmarks[10] = [0, 0.7]
    landmarks[11] = [0, 0.92]
    landmarks[12] = [0, 1.1]

    # Ring finger
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


def draw_hand_skeleton(ax, landmarks, color=COLORS['skeleton'],
                       joint_color=COLORS['joint'], label=None,
                       location_marker=None, alpha=1.0):
    """
    Draw a hand skeleton on the given axes.

    Args:
        ax: matplotlib axes
        landmarks: (21, 2) array of (x, y) coordinates
        color: color for bones
        joint_color: color for joint markers
        label: optional label below the hand
        location_marker: optional (x, y, text) for location annotation
    """
    # Draw bones
    for i, j in HAND_CONNECTIONS:
        ax.plot([landmarks[i, 0], landmarks[j, 0]],
                [landmarks[i, 1], landmarks[j, 1]],
                color=color, linewidth=2.5, alpha=alpha, solid_capstyle='round')

    # Draw joints
    for idx, (x, y) in enumerate(landmarks):
        # Fingertips are slightly larger
        size = 8 if idx in [4, 8, 12, 16, 20] else 5
        ax.scatter(x, y, c=joint_color, s=size, zorder=5, alpha=alpha)

    # Wrist marker
    ax.scatter(landmarks[0, 0], landmarks[0, 1], c=joint_color, s=12,
               zorder=6, marker='s', alpha=alpha)

    if label:
        ax.text(0, -0.25, label, ha='center', va='top', fontsize=14,
                fontweight='bold')

    if location_marker:
        x, y, text = location_marker
        ax.annotate(text, xy=(x, y), fontsize=10, ha='center', va='bottom',
                   color=COLORS['location'], fontweight='bold',
                   path_effects=[pe.withStroke(linewidth=2, foreground='white')])


def create_minimal_pair_visualization():
    """
    Create Panel A: Minimal pair visualization (MOTHER vs FATHER).
    Shows two hands with identical handshape but different locations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))

    hand = get_hand_landmarks_flat()

    # MOTHER - hand at chin level
    ax1 = axes[0]
    hand_mother = hand.copy()
    hand_mother[:, 1] += 0.3  # Shift up slightly (chin area)

    draw_hand_skeleton(ax1, hand_mother, label="MOTHER")

    # Add chin indicator
    ax1.plot([-0.6, 0.6], [-0.1, -0.1], 'k--', alpha=0.3, linewidth=1)
    ax1.text(0.65, -0.1, 'chin', fontsize=8, va='center', alpha=0.5)

    # Location highlight
    ax1.add_patch(plt.Circle((0, 0.1), 0.15, fill=False,
                             color=COLORS['location'], linewidth=2, linestyle='--'))

    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-0.5, 1.6)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # FATHER - hand at forehead level
    ax2 = axes[1]
    hand_father = hand.copy()
    hand_father[:, 1] += 0.8  # Shift up more (forehead area)

    draw_hand_skeleton(ax2, hand_father, label="FATHER")

    # Add forehead indicator
    ax2.plot([-0.6, 0.6], [0.4, 0.4], 'k--', alpha=0.3, linewidth=1)
    ax2.text(0.65, 0.4, 'forehead', fontsize=8, va='center', alpha=0.5)

    # Location highlight
    ax2.add_patch(plt.Circle((0, 0.6), 0.15, fill=False,
                             color=COLORS['location'], linewidth=2, linestyle='--'))

    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-0.5, 1.6)
    ax2.set_aspect('equal')
    ax2.axis('off')

    # Add annotation showing the difference
    fig.text(0.5, 0.02, 'Same handshape, different LOCATION',
             ha='center', fontsize=11, style='italic', color=COLORS['location'])

    plt.tight_layout()
    return fig


def create_embedding_comparison():
    """
    Create Panels B & D: Embedding space comparison.
    B: Entangled (standard model) - signs scattered randomly
    D: Disentangled (PhonSSM) - organized by phonological features
    """
    np.random.seed(42)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Generate sign data points
    n_signs = 50

    # === Panel B: Entangled (Standard Model) ===
    ax_b = axes[0]

    # Random scatter for standard model
    x_std = np.random.randn(n_signs) * 1.5
    y_std = np.random.randn(n_signs) * 1.5

    ax_b.scatter(x_std, y_std, c=COLORS['baseline'], alpha=0.5, s=40)

    # Highlight MOTHER/FATHER as far apart (confused)
    mother_idx, father_idx = 10, 35
    ax_b.scatter(x_std[mother_idx], y_std[mother_idx], c=COLORS['accent'],
                s=100, marker='*', edgecolors='white', linewidth=1, zorder=10)
    ax_b.scatter(x_std[father_idx], y_std[father_idx], c=COLORS['accent'],
                s=100, marker='*', edgecolors='white', linewidth=1, zorder=10)

    ax_b.annotate('MOTHER', (x_std[mother_idx], y_std[mother_idx]),
                 xytext=(10, 10), textcoords='offset points', fontsize=9)
    ax_b.annotate('FATHER', (x_std[father_idx], y_std[father_idx]),
                 xytext=(10, -15), textcoords='offset points', fontsize=9)

    # Draw confusion arrow
    ax_b.annotate('', xy=(x_std[father_idx], y_std[father_idx]),
                 xytext=(x_std[mother_idx], y_std[mother_idx]),
                 arrowprops=dict(arrowstyle='<->', color=COLORS['accent'],
                               lw=1.5, ls='--'))

    ax_b.set_title('B. Standard Model: Entangled', fontweight='bold', loc='left')
    ax_b.set_xlabel('Dimension 1')
    ax_b.set_ylabel('Dimension 2')
    ax_b.text(0.5, -0.12, 'Minimal pairs scattered randomly',
             transform=ax_b.transAxes, ha='center', fontsize=9,
             style='italic', color=COLORS['baseline'])

    # === Panel D: Disentangled (PhonSSM) ===
    ax_d = axes[1]

    # Create clustered structure by phonological features
    # Group by location (y-axis) and handshape (color)
    locations = ['chin', 'forehead', 'chest', 'neutral']
    loc_y = {'chin': -1.5, 'forehead': 1.5, 'chest': -0.5, 'neutral': 0.5}

    for i, loc in enumerate(locations):
        n_loc = n_signs // 4
        x_loc = np.random.randn(n_loc) * 0.4 + (i - 1.5) * 0.8
        y_loc = np.random.randn(n_loc) * 0.3 + loc_y[loc]
        ax_d.scatter(x_loc, y_loc, alpha=0.5, s=40, label=loc)

    # MOTHER and FATHER now close (same handshape) but separated on location axis
    ax_d.scatter(-0.8, -1.5, c=COLORS['accent'], s=100, marker='*',
                edgecolors='white', linewidth=1, zorder=10)
    ax_d.scatter(-0.6, 1.5, c=COLORS['accent'], s=100, marker='*',
                edgecolors='white', linewidth=1, zorder=10)

    ax_d.annotate('MOTHER', (-0.8, -1.5), xytext=(-40, -10),
                 textcoords='offset points', fontsize=9)
    ax_d.annotate('FATHER', (-0.6, 1.5), xytext=(-40, 10),
                 textcoords='offset points', fontsize=9)

    # Arrow showing they're in same handshape cluster but different location
    ax_d.annotate('', xy=(-0.6, 1.2), xytext=(-0.8, -1.2),
                 arrowprops=dict(arrowstyle='<->', color=COLORS['location'],
                               lw=2))
    ax_d.text(-1.5, 0, 'Location\naxis', fontsize=8, ha='center',
             color=COLORS['location'], rotation=90, va='center')

    ax_d.set_title('D. PhonSSM: Organized by Phonology', fontweight='bold', loc='left')
    ax_d.set_xlabel('Handshape dimension')
    ax_d.set_ylabel('Location dimension')
    ax_d.text(0.5, -0.12, 'Minimal pairs correctly separated',
             transform=ax_d.transAxes, ha='center', fontsize=9,
             style='italic', color=COLORS['phonssm'])

    # Add location labels
    for loc, y in loc_y.items():
        ax_d.text(2.2, y, loc, fontsize=8, va='center', alpha=0.7)

    ax_d.set_xlim(-2.5, 2.5)
    ax_d.set_ylim(-2.5, 2.5)

    plt.tight_layout()
    return fig


def create_decomposition_diagram():
    """
    Create Panel C: Phonological decomposition schematic.
    Shows input -> 4 parallel streams (H, L, M, O).
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Input box
    input_box = plt.Rectangle((0.5, 2), 1.5, 2, fill=True,
                               facecolor='#f0f0f0', edgecolor='black', linewidth=1.5)
    ax.add_patch(input_box)
    ax.text(1.25, 3, 'Input\nX', ha='center', va='center', fontsize=10)

    # Draw small hand skeleton in input box
    hand = get_hand_landmarks_flat()
    hand_scaled = hand * 0.4 + np.array([1.25, 3])
    for i, j in HAND_CONNECTIONS[:10]:  # Simplified
        ax.plot([hand_scaled[i, 0], hand_scaled[j, 0]],
                [hand_scaled[i, 1], hand_scaled[j, 1]],
                color='gray', linewidth=0.8, alpha=0.5)

    # Arrow from input
    ax.annotate('', xy=(2.8, 3), xytext=(2, 3),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Four parallel streams
    components = [
        ('H', COLORS['handshape'], 'Handshape', 4.8),
        ('L', COLORS['location'], 'Location', 3.6),
        ('M', COLORS['movement'], 'Movement', 2.4),
        ('O', COLORS['orientation'], 'Orientation', 1.2),
    ]

    for letter, color, name, y in components:
        # Box
        box = plt.Rectangle((3, y - 0.4), 3.5, 0.7, fill=True,
                            facecolor=color, edgecolor='black',
                            linewidth=1.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(4.75, y, f'{letter}: {name}', ha='center', va='center',
               fontsize=9, color='white', fontweight='bold')

        # Arrows
        ax.annotate('', xy=(3, y), xytext=(2.8, 3),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        ax.annotate('', xy=(7.5, 3), xytext=(6.5, y),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    # Orthogonality symbol
    ax.text(7, 1.8, '⊥', fontsize=14, ha='center', va='center')
    ax.text(7, 1.2, 'orthogonal', fontsize=7, ha='center', style='italic')

    # Output
    output_box = plt.Rectangle((7.5, 2), 2, 2, fill=True,
                                facecolor=COLORS['phonssm'],
                                edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(output_box)
    ax.text(8.5, 3, 'Disentangled\nFeatures', ha='center', va='center',
           fontsize=9, color='white', fontweight='bold')

    ax.set_title('C. PhonSSM: Explicit Decomposition', fontweight='bold',
                loc='left', pad=10)

    plt.tight_layout()
    return fig


def create_all_components():
    """Generate all figure components and save as separate files."""
    output_dir = Path('figures/motivation_components')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating motivation figure components...")

    # Panel A: Minimal pairs
    fig_a = create_minimal_pair_visualization()
    fig_a.savefig(output_dir / 'panel_a_minimal_pairs.pdf', bbox_inches='tight')
    fig_a.savefig(output_dir / 'panel_a_minimal_pairs.png', bbox_inches='tight', dpi=300)
    plt.close(fig_a)
    print("  [OK] Panel A: Minimal pairs saved")

    # Panels B & D: Embedding comparison
    fig_bd = create_embedding_comparison()
    fig_bd.savefig(output_dir / 'panels_bd_embeddings.pdf', bbox_inches='tight')
    fig_bd.savefig(output_dir / 'panels_bd_embeddings.png', bbox_inches='tight', dpi=300)
    plt.close(fig_bd)
    print("  [OK] Panels B & D: Embedding comparison saved")

    # Panel C: Decomposition diagram
    fig_c = create_decomposition_diagram()
    fig_c.savefig(output_dir / 'panel_c_decomposition.pdf', bbox_inches='tight')
    fig_c.savefig(output_dir / 'panel_c_decomposition.png', bbox_inches='tight', dpi=300)
    plt.close(fig_c)
    print("  [OK] Panel C: Decomposition diagram saved")

    print(f"\nAll components saved to {output_dir}/")
    print("\nNext steps:")
    print("1. Import these PNGs into Figma")
    print("2. Arrange in 2x2 grid layout")
    print("3. Add panel labels (A, B, C, D)")
    print("4. Add overall figure caption")
    print("5. Export final figure as PDF")


def create_combined_figure():
    """
    Create the complete 4-panel motivation figure.
    This is an alternative to compositing in Figma.
    """
    fig = plt.figure(figsize=(10, 8))

    # Panel A: Minimal pairs (top-left)
    ax_a1 = fig.add_axes([0.05, 0.55, 0.2, 0.4])
    ax_a2 = fig.add_axes([0.27, 0.55, 0.2, 0.4])

    hand = get_hand_landmarks_flat()

    # MOTHER
    hand_mother = hand.copy()
    hand_mother[:, 1] += 0.3
    draw_hand_skeleton(ax_a1, hand_mother)
    ax_a1.text(0, -0.2, 'MOTHER', ha='center', va='top', fontsize=11, fontweight='bold')
    ax_a1.plot([-0.5, 0.5], [-0.05, -0.05], 'k--', alpha=0.3, linewidth=1)
    ax_a1.text(0.55, -0.05, 'chin', fontsize=7, va='center', alpha=0.5)
    ax_a1.set_xlim(-0.8, 0.8)
    ax_a1.set_ylim(-0.4, 1.4)
    ax_a1.set_aspect('equal')
    ax_a1.axis('off')

    # FATHER
    hand_father = hand.copy()
    hand_father[:, 1] += 0.7
    draw_hand_skeleton(ax_a2, hand_father)
    ax_a2.text(0, -0.2, 'FATHER', ha='center', va='top', fontsize=11, fontweight='bold')
    ax_a2.plot([-0.5, 0.5], [0.35, 0.35], 'k--', alpha=0.3, linewidth=1)
    ax_a2.text(0.55, 0.35, 'forehead', fontsize=7, va='center', alpha=0.5)
    ax_a2.set_xlim(-0.8, 0.8)
    ax_a2.set_ylim(-0.4, 1.4)
    ax_a2.set_aspect('equal')
    ax_a2.axis('off')

    # Panel A title and caption
    fig.text(0.05, 0.96, 'A. The Problem: Minimal Pairs', fontsize=11, fontweight='bold')
    fig.text(0.25, 0.52, 'Only LOCATION differs', ha='center', fontsize=9,
             style='italic', color=COLORS['location'])

    # Panel B: Entangled embeddings (top-right)
    ax_b = fig.add_axes([0.55, 0.55, 0.4, 0.4])
    np.random.seed(42)
    n = 40
    x_std = np.random.randn(n) * 1.2
    y_std = np.random.randn(n) * 1.2
    ax_b.scatter(x_std, y_std, c=COLORS['baseline'], alpha=0.4, s=30)

    # Highlight minimal pair (far apart = confused)
    ax_b.scatter(-1.5, 1.2, c=COLORS['accent'], s=80, marker='*', zorder=10)
    ax_b.scatter(1.3, -0.8, c=COLORS['accent'], s=80, marker='*', zorder=10)
    ax_b.annotate('MOTHER', (-1.5, 1.2), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax_b.annotate('FATHER', (1.3, -0.8), xytext=(5, -10), textcoords='offset points', fontsize=8)
    ax_b.annotate('', xy=(1.3, -0.8), xytext=(-1.5, 1.2),
                 arrowprops=dict(arrowstyle='<->', color=COLORS['accent'], lw=1.5, ls='--'))
    ax_b.text(0, 0.5, '?', fontsize=20, ha='center', va='center', color=COLORS['accent'], alpha=0.5)

    ax_b.set_title('B. Standard Model: Entangled', fontweight='bold', loc='left', fontsize=11)
    ax_b.set_xlabel('Dimension 1', fontsize=9)
    ax_b.set_ylabel('Dimension 2', fontsize=9)
    ax_b.tick_params(labelsize=8)

    # Panel C: Decomposition (bottom-left)
    ax_c = fig.add_axes([0.05, 0.08, 0.42, 0.38])
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 5)
    ax_c.axis('off')

    # Input
    input_box = plt.Rectangle((0.2, 1.5), 1.3, 2),
    ax_c.add_patch(plt.Rectangle((0.2, 1.5), 1.3, 2, fill=True,
                                  facecolor='#f0f0f0', edgecolor='black', linewidth=1))
    ax_c.text(0.85, 2.5, 'Input', ha='center', va='center', fontsize=9)

    # Arrow
    ax_c.annotate('', xy=(2.2, 2.5), xytext=(1.5, 2.5),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # Four streams
    components = [
        ('H', COLORS['handshape'], 4.0),
        ('L', COLORS['location'], 3.0),
        ('M', COLORS['movement'], 2.0),
        ('O', COLORS['orientation'], 1.0),
    ]

    for letter, color, y in components:
        box = plt.Rectangle((2.3, y - 0.35), 2.8, 0.6, fill=True,
                            facecolor=color, edgecolor='black', linewidth=1, alpha=0.85)
        ax_c.add_patch(box)
        ax_c.text(3.7, y, letter, ha='center', va='center', fontsize=10,
                 color='white', fontweight='bold')
        ax_c.annotate('', xy=(2.3, y), xytext=(2.2, 2.5),
                     arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
        ax_c.annotate('', xy=(6.0, 2.5), xytext=(5.1, y),
                     arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # Output
    ax_c.add_patch(plt.Rectangle((6.0, 1.5), 1.8, 2, fill=True,
                                  facecolor=COLORS['phonssm'], edgecolor='black',
                                  linewidth=1, alpha=0.85))
    ax_c.text(6.9, 2.5, 'Factored\nFeatures', ha='center', va='center',
             fontsize=9, color='white', fontweight='bold')

    ax_c.text(5.5, 0.6, '⊥ orthogonal', fontsize=8, ha='center', style='italic')

    fig.text(0.05, 0.47, 'C. PhonSSM: Explicit Decomposition', fontsize=11, fontweight='bold')

    # Panel D: Organized embeddings (bottom-right)
    ax_d = fig.add_axes([0.55, 0.08, 0.4, 0.38])
    np.random.seed(123)

    # Clustered by location
    locs = [('chin', -1.2, COLORS['location']),
            ('forehead', 1.2, COLORS['phonssm']),
            ('chest', -0.3, COLORS['movement']),
            ('neutral', 0.4, COLORS['orientation'])]

    for loc, y_center, c in locs:
        n_loc = 10
        x_loc = np.random.randn(n_loc) * 0.3 + np.random.uniform(-1, 1)
        y_loc = np.random.randn(n_loc) * 0.2 + y_center
        ax_d.scatter(x_loc, y_loc, alpha=0.4, s=30, c=c)

    # MOTHER and FATHER now correctly organized
    ax_d.scatter(-0.5, -1.2, c=COLORS['accent'], s=80, marker='*', zorder=10)
    ax_d.scatter(-0.3, 1.2, c=COLORS['accent'], s=80, marker='*', zorder=10)
    ax_d.annotate('MOTHER', (-0.5, -1.2), xytext=(-50, -5), textcoords='offset points', fontsize=8)
    ax_d.annotate('FATHER', (-0.3, 1.2), xytext=(-50, 5), textcoords='offset points', fontsize=8)

    # Arrow showing location separation
    ax_d.annotate('', xy=(-0.3, 0.9), xytext=(-0.5, -0.9),
                 arrowprops=dict(arrowstyle='<->', color=COLORS['location'], lw=2))
    ax_d.text(-1.5, 0, 'Location\naxis', fontsize=8, ha='center',
             color=COLORS['location'], rotation=90, va='center')

    # Check mark (using matplotlib symbol)
    ax_d.plot([0.8, 1.0, 1.4], [-0.3, -0.5, 0.1], color=COLORS['phonssm'],
             linewidth=3, alpha=0.5, solid_capstyle='round')

    ax_d.set_title('D. PhonSSM: Organized by Phonology', fontweight='bold', loc='left', fontsize=11)
    ax_d.set_xlabel('Handshape dimension', fontsize=9)
    ax_d.set_ylabel('Location dimension', fontsize=9)
    ax_d.tick_params(labelsize=8)
    ax_d.set_xlim(-2, 2)
    ax_d.set_ylim(-2, 2)

    plt.savefig('figures/fig_motivation.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig('figures/fig_motivation.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close()
    print("Combined motivation figure saved to figures/fig_motivation.pdf")


if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent.parent)  # Change to paper/ directory

    print("=" * 60)
    print("Generating Motivation Figure Components for PhonSSM Paper")
    print("=" * 60)

    Path('figures').mkdir(exist_ok=True)

    # Generate individual components (for Figma compositing)
    create_all_components()

    # Also generate combined figure (pure Python option)
    print("\nGenerating combined figure...")
    create_combined_figure()

    print("\n" + "=" * 60)
    print("Done! Two options available:")
    print("1. Use figures/fig_motivation.pdf (complete Python-generated)")
    print("2. Composite figures/motivation_components/* in Figma")
    print("=" * 60)
