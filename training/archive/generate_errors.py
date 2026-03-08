"""
SignSense Synthetic Error Generation Module
Generates labeled error samples for training the Error Diagnosis Network.

Key Innovation: No public dataset contains labeled errors, so we generate them
programmatically. This enables supervised training of error detection models.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import json


# Error type definitions
ERROR_TYPES = {
    'handshape': [
        'finger_not_extended',
        'fingers_not_curled',
        'wrong_handshape',
        'thumb_position'
    ],
    'location': [
        'hand_too_high',
        'hand_too_low',
        'hand_too_left',
        'hand_too_right',
        'wrong_location'
    ],
    'movement': [
        'too_fast',
        'too_slow',
        'wrong_direction',
        'incomplete',
        'extra_movement'
    ],
    'orientation': [
        'palm_wrong_direction',
        'wrist_rotation'
    ]
}

# Flatten to list of all error types
ALL_ERROR_TYPES = []
for category, errors in ERROR_TYPES.items():
    for error in errors:
        ALL_ERROR_TYPES.append(f"{category}:{error}")

ERROR_TO_IDX = {e: i for i, e in enumerate(ALL_ERROR_TYPES)}
IDX_TO_ERROR = {i: e for e, i in ERROR_TO_IDX.items()}

# Component indices
COMPONENTS = ['handshape', 'location', 'movement', 'orientation']
COMPONENT_TO_IDX = {c: i for i, c in enumerate(COMPONENTS)}


# =============================================================================
# Handshape Error Generators
# =============================================================================

def generate_finger_not_extended(landmarks):
    """Curl a finger that should be extended."""
    modified = landmarks.copy().reshape(-1, 21, 3)

    # Randomly choose a finger (index=5-8, middle=9-12, ring=13-16, pinky=17-20)
    finger_bases = [5, 9, 13, 17]
    finger_idx = np.random.choice(finger_bases)

    # Curl finger toward palm (landmark 0 = wrist)
    wrist = modified[:, 0, :]  # Shape: (30, 3)
    for i in range(finger_idx, finger_idx + 4):
        # Move landmarks toward wrist
        t = (i - finger_idx + 1) / 4  # 0.25, 0.5, 0.75, 1.0
        modified[:, i, :] = modified[:, i, :] * (1 - t * 0.6) + wrist * (t * 0.6)

    return modified.reshape(-1, 63), "handshape:finger_not_extended"


def generate_fingers_not_curled(landmarks):
    """Extend fingers that should be curled."""
    modified = landmarks.copy().reshape(-1, 21, 3)

    # Extend all fingertips
    for tip in [4, 8, 12, 16, 20]:
        base = tip - 3
        # Direction from base to tip
        direction = modified[:, tip] - modified[:, base]
        # Extend further
        modified[:, tip] = modified[:, base] + direction * 1.5

    return modified.reshape(-1, 63), "handshape:fingers_not_curled"


def generate_wrong_handshape(landmarks):
    """Randomly perturb finger positions."""
    modified = landmarks.copy().reshape(-1, 21, 3)

    # Add random perturbations to finger landmarks
    for i in range(4, 21):  # Skip wrist and palm base
        perturbation = np.random.uniform(-0.15, 0.15, size=(len(modified), 3))
        modified[:, i] += perturbation

    return modified.reshape(-1, 63), "handshape:wrong_handshape"


def generate_thumb_position(landmarks):
    """Move thumb to incorrect position."""
    modified = landmarks.copy().reshape(-1, 21, 3)

    # Thumb landmarks: 1, 2, 3, 4
    # Rotate thumb around wrist
    angle = np.random.uniform(30, 60) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    for i in range(1, 5):
        x = modified[:, i, 0]
        y = modified[:, i, 1]
        modified[:, i, 0] = x * cos_a - y * sin_a
        modified[:, i, 1] = x * sin_a + y * cos_a

    return modified.reshape(-1, 63), "handshape:thumb_position"


# =============================================================================
# Location Error Generators
# =============================================================================

def generate_hand_too_high(landmarks):
    """Shift hand position up."""
    modified = landmarks.copy().reshape(-1, 21, 3)
    shift = np.array([0, -0.15, 0])  # Negative Y is up in image coords
    modified = modified + shift
    return modified.reshape(-1, 63), "location:hand_too_high"


def generate_hand_too_low(landmarks):
    """Shift hand position down."""
    modified = landmarks.copy().reshape(-1, 21, 3)
    shift = np.array([0, 0.15, 0])  # Positive Y is down
    modified = modified + shift
    return modified.reshape(-1, 63), "location:hand_too_low"


def generate_hand_too_left(landmarks):
    """Shift hand position left."""
    modified = landmarks.copy().reshape(-1, 21, 3)
    shift = np.array([-0.15, 0, 0])
    modified = modified + shift
    return modified.reshape(-1, 63), "location:hand_too_left"


def generate_hand_too_right(landmarks):
    """Shift hand position right."""
    modified = landmarks.copy().reshape(-1, 21, 3)
    shift = np.array([0.15, 0, 0])
    modified = modified + shift
    return modified.reshape(-1, 63), "location:hand_too_right"


def generate_wrong_location(landmarks):
    """Shift hand to completely wrong location."""
    modified = landmarks.copy().reshape(-1, 21, 3)
    shift = np.random.uniform(-0.25, 0.25, size=3)
    shift[2] = shift[2] * 0.5  # Less Z shift
    modified = modified + shift
    return modified.reshape(-1, 63), "location:wrong_location"


# =============================================================================
# Movement Error Generators
# =============================================================================

def generate_too_fast(landmarks):
    """Speed up movement by subsampling frames."""
    n_frames = len(landmarks) // 63  # Assumes flattened input
    modified = landmarks.copy().reshape(n_frames, 21, 3)

    # Take every other frame
    indices = np.linspace(0, n_frames - 1, n_frames // 2).astype(int)
    fast = modified[indices]

    # Pad back to original length by repeating
    result = np.repeat(fast, 2, axis=0)[:n_frames]

    return result.reshape(-1, 63), "movement:too_fast"


def generate_too_slow(landmarks):
    """Slow down movement by interpolating."""
    n_frames = len(landmarks) // 63
    modified = landmarks.copy().reshape(n_frames, 21, 3)

    # Use only first half of frames, stretched
    half = n_frames // 2
    indices = np.linspace(0, half - 1, n_frames).astype(int)
    slow = modified[indices]

    return slow.reshape(-1, 63), "movement:too_slow"


def generate_wrong_direction(landmarks):
    """Reverse the movement direction."""
    n_frames = len(landmarks) // 63
    modified = landmarks.copy().reshape(n_frames, 21, 3)

    # Reverse frame order
    reversed_lm = modified[::-1].copy()

    return reversed_lm.reshape(-1, 63), "movement:wrong_direction"


def generate_incomplete(landmarks):
    """Truncate movement early."""
    n_frames = len(landmarks) // 63
    modified = landmarks.copy().reshape(n_frames, 21, 3)

    # Stop at 60% of the movement
    cutoff = int(n_frames * 0.6)
    modified[cutoff:] = modified[cutoff - 1]  # Freeze

    return modified.reshape(-1, 63), "movement:incomplete"


def generate_extra_movement(landmarks):
    """Add spurious movement at end."""
    n_frames = len(landmarks) // 63
    modified = landmarks.copy().reshape(n_frames, 21, 3)

    # Add oscillation in last 20% of frames
    start = int(n_frames * 0.8)
    for i in range(start, n_frames):
        t = (i - start) / (n_frames - start)
        oscillation = np.sin(t * 4 * np.pi) * 0.1
        modified[i] += oscillation

    return modified.reshape(-1, 63), "movement:extra_movement"


# =============================================================================
# Orientation Error Generators
# =============================================================================

def generate_palm_wrong_direction(landmarks):
    """Rotate hand to wrong palm orientation."""
    n_frames = len(landmarks) // 63
    modified = landmarks.copy().reshape(n_frames, 21, 3)

    # Rotate around Y-axis (changes palm direction)
    angle = np.random.uniform(45, 90) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    for f in range(n_frames):
        for i in range(21):
            x, z = modified[f, i, 0], modified[f, i, 2]
            modified[f, i, 0] = x * cos_a - z * sin_a
            modified[f, i, 2] = x * sin_a + z * cos_a

    return modified.reshape(-1, 63), "orientation:palm_wrong_direction"


def generate_wrist_rotation(landmarks):
    """Rotate wrist incorrectly."""
    n_frames = len(landmarks) // 63
    modified = landmarks.copy().reshape(n_frames, 21, 3)

    # Rotate around X-axis
    angle = np.random.uniform(20, 45) * np.pi / 180
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    for f in range(n_frames):
        for i in range(21):
            y, z = modified[f, i, 1], modified[f, i, 2]
            modified[f, i, 1] = y * cos_a - z * sin_a
            modified[f, i, 2] = y * sin_a + z * cos_a

    return modified.reshape(-1, 63), "orientation:wrist_rotation"


# =============================================================================
# Main Error Generation
# =============================================================================

# Mapping from error type to generator function
ERROR_GENERATORS = {
    'handshape:finger_not_extended': generate_finger_not_extended,
    'handshape:fingers_not_curled': generate_fingers_not_curled,
    'handshape:wrong_handshape': generate_wrong_handshape,
    'handshape:thumb_position': generate_thumb_position,
    'location:hand_too_high': generate_hand_too_high,
    'location:hand_too_low': generate_hand_too_low,
    'location:hand_too_left': generate_hand_too_left,
    'location:hand_too_right': generate_hand_too_right,
    'location:wrong_location': generate_wrong_location,
    'movement:too_fast': generate_too_fast,
    'movement:too_slow': generate_too_slow,
    'movement:wrong_direction': generate_wrong_direction,
    'movement:incomplete': generate_incomplete,
    'movement:extra_movement': generate_extra_movement,
    'orientation:palm_wrong_direction': generate_palm_wrong_direction,
    'orientation:wrist_rotation': generate_wrist_rotation,
}


def generate_error_sample(landmarks, error_type=None):
    """
    Generate an error sample from correct landmarks.

    Args:
        landmarks: (30, 63) correct landmark sequence
        error_type: Specific error to generate (None = random)

    Returns:
        modified_landmarks: (30, 63) landmarks with error
        error_type: String indicating error type
        component_scores: (4,) scores for each component
        error_label: (15,) multi-hot error label
    """
    if error_type is None:
        error_type = np.random.choice(list(ERROR_GENERATORS.keys()))

    generator = ERROR_GENERATORS[error_type]
    modified, error_name = generator(landmarks.flatten())
    modified = modified.reshape(30, 63)

    # Create component scores (0 = wrong, 1 = correct)
    component_scores = np.ones(4, dtype=np.float32)
    error_category = error_name.split(':')[0]
    component_scores[COMPONENT_TO_IDX[error_category]] = 0.0

    # Add some noise to scores
    component_scores += np.random.uniform(-0.1, 0.1, size=4)
    component_scores = np.clip(component_scores, 0, 1)

    # Create multi-hot error label
    error_label = np.zeros(len(ALL_ERROR_TYPES), dtype=np.float32)
    error_label[ERROR_TO_IDX[error_name]] = 1.0

    return modified, error_name, component_scores, error_label


def augment_with_errors(X, y, errors_per_sample=3, include_correct=True, seed=42):
    """
    Augment dataset with synthetic error examples.

    Args:
        X: (N, 30, 63) correct landmark sequences
        y: (N,) sign labels
        errors_per_sample: Number of error variants per correct sample
        include_correct: Whether to include original correct samples
        seed: Random seed

    Returns:
        X_aug: Augmented landmarks
        y_aug: Sign labels
        is_correct: (M,) boolean array
        component_scores: (M, 4) component quality scores
        error_labels: (M, 15) multi-hot error labels
    """
    np.random.seed(seed)

    X_aug = []
    y_aug = []
    is_correct = []
    component_scores = []
    error_labels = []

    # Include correct samples
    if include_correct:
        for i in range(len(X)):
            X_aug.append(X[i])
            y_aug.append(y[i])
            is_correct.append(True)
            component_scores.append(np.ones(4, dtype=np.float32))
            error_labels.append(np.zeros(len(ALL_ERROR_TYPES), dtype=np.float32))

    # Generate error samples
    print(f"Generating {errors_per_sample} error variants per sample...")
    for i in tqdm(range(len(X)), desc="Generating errors"):
        for _ in range(errors_per_sample):
            modified, error_name, comp_scores, err_label = generate_error_sample(X[i])

            X_aug.append(modified)
            y_aug.append(y[i])
            is_correct.append(False)
            component_scores.append(comp_scores)
            error_labels.append(err_label)

    return (
        np.array(X_aug, dtype=np.float32),
        np.array(y_aug, dtype=np.int32),
        np.array(is_correct, dtype=bool),
        np.array(component_scores, dtype=np.float32),
        np.array(error_labels, dtype=np.float32)
    )


def save_error_metadata(output_dir):
    """Save error type mappings."""
    output_dir = Path(output_dir)

    metadata = {
        'error_types': ALL_ERROR_TYPES,
        'error_to_idx': ERROR_TO_IDX,
        'components': COMPONENTS,
        'component_to_idx': COMPONENT_TO_IDX,
        'error_categories': ERROR_TYPES
    }

    with open(output_dir / 'error_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved error metadata to {output_dir / 'error_metadata.json'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic error samples')
    parser.add_argument('--input-dir', type=str,
                        default='data/processed',
                        help='Directory with X_train.npy, etc.')
    parser.add_argument('--output-dir', type=str,
                        default='data/processed',
                        help='Output directory')
    parser.add_argument('--errors-per-sample', type=int, default=3,
                        help='Number of error variants per correct sample')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Load training data
    print("Loading training data...")
    X_train = np.load(input_dir / 'X_train.npy')
    y_train = np.load(input_dir / 'y_train.npy')
    print(f"Loaded: X_train={X_train.shape}, y_train={y_train.shape}")

    # Generate errors for training set
    print(f"\nGenerating errors ({args.errors_per_sample} per sample)...")
    X_aug, y_aug, is_correct, comp_scores, err_labels = augment_with_errors(
        X_train, y_train,
        errors_per_sample=args.errors_per_sample
    )

    print(f"\nAugmented dataset:")
    print(f"  Total samples: {len(X_aug)}")
    print(f"  Correct: {is_correct.sum()}")
    print(f"  Errors: {(~is_correct).sum()}")

    # Save
    print("\nSaving...")
    np.save(output_dir / 'X_train_with_errors.npy', X_aug)
    np.save(output_dir / 'y_train_with_errors.npy', y_aug)
    np.save(output_dir / 'is_correct_train.npy', is_correct)
    np.save(output_dir / 'component_scores_train.npy', comp_scores)
    np.save(output_dir / 'error_labels_train.npy', err_labels)

    save_error_metadata(output_dir)

    print("\nDone!")
