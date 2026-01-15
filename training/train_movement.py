"""
Movement Pattern Analyzer Training Script
==========================================
1D CNN that evaluates movement quality: speed, smoothness, and completeness.

Architecture:
    Input: (30 frames, 9 features) - position(3) + velocity(3) + acceleration(3)
    → Conv1D(32, k=3) → ReLU → MaxPool
    → Conv1D(64, k=3) → ReLU → MaxPool
    → Conv1D(128, k=3) → ReLU → GlobalAvgPool
    → Two output heads:
        1. Movement type (6 classes): static, linear, circular, arc, zigzag, compound
        2. Quality scores (3 values): speed, smoothness, completeness

Target metrics:
    - Movement type accuracy: >85%
    - Parameters: ~45K

Usage:
    python training/train_movement.py
    python training/train_movement.py --epochs 100 --batch-size 128
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, BatchNormalization,
    MaxPooling1D, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.utils import to_categorical

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "merged"
MODEL_DIR = PROJECT_ROOT / "models" / "movement_analyzer"

# Movement types
MOVEMENT_TYPES = ['static', 'linear', 'circular', 'arc', 'zigzag', 'compound']


def compute_motion_features(landmarks):
    """
    Compute position, velocity, and acceleration from landmark sequences.

    Input: (N, 30, 63) - N sequences of 30 frames with 63 features (21 landmarks × 3)
    Output: (N, 30, 9) - position(3) + velocity(3) + acceleration(3)

    We use the wrist (landmark 0) as the reference point for motion.
    """
    # Extract wrist position (first 3 features per frame)
    # landmarks shape: (N, 30, 63)
    position = landmarks[:, :, :3]  # (N, 30, 3) - wrist x, y, z

    # Compute velocity (first derivative)
    velocity = np.zeros_like(position)
    velocity[:, 1:, :] = position[:, 1:, :] - position[:, :-1, :]

    # Compute acceleration (second derivative)
    acceleration = np.zeros_like(velocity)
    acceleration[:, 1:, :] = velocity[:, 1:, :] - velocity[:, :-1, :]

    # Concatenate: position + velocity + acceleration
    motion_features = np.concatenate([position, velocity, acceleration], axis=2)

    return motion_features  # (N, 30, 9)


def classify_movement_type(motion_features):
    """
    Classify movement type based on motion characteristics.
    Returns movement type labels and quality scores.

    This is a heuristic-based labeling for training data generation.
    """
    n_samples = len(motion_features)
    labels = []
    quality_scores = []

    for i in range(n_samples):
        position = motion_features[i, :, :3]
        velocity = motion_features[i, :, 3:6]
        acceleration = motion_features[i, :, 6:9]

        # Compute motion statistics
        total_displacement = np.linalg.norm(position[-1] - position[0])
        path_length = np.sum(np.linalg.norm(np.diff(position, axis=0), axis=1))
        avg_speed = np.mean(np.linalg.norm(velocity, axis=1))
        speed_variance = np.var(np.linalg.norm(velocity, axis=1))
        avg_acceleration = np.mean(np.linalg.norm(acceleration, axis=1))

        # Classify based on heuristics
        if avg_speed < 0.01:
            movement_type = 0  # static
        elif path_length > 0 and total_displacement / (path_length + 1e-6) > 0.8:
            movement_type = 1  # linear
        elif avg_acceleration > 0.02 and speed_variance < 0.01:
            movement_type = 2  # circular
        elif avg_acceleration > 0.01:
            movement_type = 3  # arc
        elif speed_variance > 0.02:
            movement_type = 4  # zigzag
        else:
            movement_type = 5  # compound

        labels.append(movement_type)

        # Compute quality scores
        speed_score = min(1.0, avg_speed / 0.1)  # Normalized speed
        smoothness_score = max(0, 1.0 - speed_variance * 10)  # Less variance = smoother
        completeness_score = min(1.0, path_length / 0.5)  # Normalized path length

        quality_scores.append([speed_score, smoothness_score, completeness_score])

    return np.array(labels), np.array(quality_scores, dtype=np.float32)


def load_and_prepare_data(data_dir):
    """Load data and compute motion features."""
    data_dir = Path(data_dir)

    print("Loading landmark data...")
    X_train = np.load(data_dir / "X_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    X_test = np.load(data_dir / "X_test.npy")

    print(f"Raw shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("\nComputing motion features...")
    X_train_motion = compute_motion_features(X_train)
    X_val_motion = compute_motion_features(X_val)
    X_test_motion = compute_motion_features(X_test)

    print(f"Motion shapes - Train: {X_train_motion.shape}")

    print("\nClassifying movement types...")
    y_train_type, y_train_quality = classify_movement_type(X_train_motion)
    y_val_type, y_val_quality = classify_movement_type(X_val_motion)
    y_test_type, y_test_quality = classify_movement_type(X_test_motion)

    # Print distribution
    unique, counts = np.unique(y_train_type, return_counts=True)
    print("\nMovement type distribution:")
    for u, c in zip(unique, counts):
        print(f"  {MOVEMENT_TYPES[u]}: {c} ({c/len(y_train_type)*100:.1f}%)")

    return {
        'X_train': X_train_motion,
        'X_val': X_val_motion,
        'X_test': X_test_motion,
        'y_train_type': y_train_type,
        'y_val_type': y_val_type,
        'y_test_type': y_test_type,
        'y_train_quality': y_train_quality,
        'y_val_quality': y_val_quality,
        'y_test_quality': y_test_quality
    }


def build_model(input_shape, num_movement_types=6, num_quality_scores=3):
    """
    Build Movement Pattern Analyzer model.

    Lightweight 1D CNN with two output heads.
    """
    inputs = Input(shape=input_shape, name='motion_features')

    # Conv1D backbone
    x = Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)

    x = Dropout(0.3)(x)

    # Head 1: Movement type classification
    type_out = Dense(32, activation='relu')(x)
    type_out = Dense(num_movement_types, activation='softmax', name='movement_type')(type_out)

    # Head 2: Quality scores (speed, smoothness, completeness)
    quality_out = Dense(32, activation='relu')(x)
    quality_out = Dense(num_quality_scores, activation='sigmoid', name='quality')(quality_out)

    model = Model(inputs, [type_out, quality_out])

    return model


def train(args):
    """Main training function."""
    print("=" * 60)
    print("MOVEMENT PATTERN ANALYZER TRAINING")
    print("=" * 60)

    # Load and prepare data
    data = load_and_prepare_data(args.data_dir)

    input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])  # (30, 9)
    num_types = len(MOVEMENT_TYPES)

    print(f"\nInput shape: {input_shape}")
    print(f"Movement types: {num_types}")

    # Convert type labels to one-hot
    y_train_type_cat = to_categorical(data['y_train_type'], num_types)
    y_val_type_cat = to_categorical(data['y_val_type'], num_types)
    y_test_type_cat = to_categorical(data['y_test_type'], num_types)

    # Build model
    print("\nBuilding model...")
    model = build_model(input_shape, num_types, 3)
    model.summary()

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss={
            'movement_type': 'categorical_crossentropy',
            'quality': 'mse'
        },
        loss_weights={
            'movement_type': 1.0,
            'quality': 0.5
        },
        metrics={
            'movement_type': ['accuracy'],
            'quality': ['mae']
        }
    )

    # Callbacks
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_DIR / 'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=MODEL_DIR / 'logs' / timestamp,
            histogram_freq=1
        )
    ]

    # Train
    print(f"\nTraining for up to {args.epochs} epochs...")

    history = model.fit(
        data['X_train'],
        {'movement_type': y_train_type_cat, 'quality': data['y_train_quality']},
        validation_data=(
            data['X_val'],
            {'movement_type': y_val_type_cat, 'quality': data['y_val_quality']}
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results = model.evaluate(
        data['X_test'],
        {'movement_type': y_test_type_cat, 'quality': data['y_test_quality']},
        verbose=0
    )

    print(f"Total Loss: {results[0]:.4f}")
    print(f"Movement Type Accuracy: {results[3]:.4f} ({results[3]*100:.2f}%)")
    print(f"Quality MAE: {results[4]:.4f}")

    # Save model
    model.save(MODEL_DIR / 'movement_analyzer.keras')
    print(f"\nModel saved to {MODEL_DIR / 'movement_analyzer.keras'}")

    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(MODEL_DIR / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)

    # Save model info
    model_info = {
        'input_shape': list(input_shape),
        'num_movement_types': num_types,
        'movement_types': MOVEMENT_TYPES,
        'parameters': int(model.count_params()),
        'test_type_accuracy': float(results[3]),
        'test_quality_mae': float(results[4]),
        'epochs_trained': len(history.history['loss'])
    }
    with open(MODEL_DIR / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model parameters: {model.count_params():,}")
    print(f"Saved to: {MODEL_DIR}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Movement Pattern Analyzer')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                        help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')

    args = parser.parse_args()
    train(args)
