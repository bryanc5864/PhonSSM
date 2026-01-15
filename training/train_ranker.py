"""
Feedback Ranker Training Script
================================
MLP that prioritizes which feedback corrections to show first.

Architecture:
    Input: (22,) features:
        - 4 component scores (handshape, location, movement, orientation)
        - 16 error type probabilities
        - 1 user skill level
        - 1 sign difficulty
    → Dense(64, ReLU)
    → Dense(32, ReLU)
    → Dense(1, Sigmoid) - priority score

Target metrics:
    - Ranking accuracy: >80%
    - Parameters: ~50K

Usage:
    python training/train_ranker.py
    python training/train_ranker.py --epochs 100 --batch-size 256
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "feedback_ranker"

# Error severity weights (higher = more important to fix first)
ERROR_SEVERITY = {
    'handshape:finger_not_extended': 0.9,
    'handshape:fingers_not_curled': 0.85,
    'handshape:wrong_handshape': 0.95,
    'handshape:thumb_position': 0.7,
    'location:hand_too_high': 0.8,
    'location:hand_too_low': 0.8,
    'location:hand_too_left': 0.75,
    'location:hand_too_right': 0.75,
    'location:wrong_location': 0.85,
    'movement:too_fast': 0.6,
    'movement:too_slow': 0.5,
    'movement:wrong_direction': 0.9,
    'movement:incomplete': 0.85,
    'movement:extra_movement': 0.4,
    'orientation:palm_wrong_direction': 0.8,
    'orientation:wrist_rotation': 0.7
}


def generate_ranking_data(data_dir, num_samples=100000):
    """
    Generate training data for the feedback ranker.

    Creates synthetic examples with error combinations and computes
    priority scores based on error severity and context.
    """
    data_dir = Path(data_dir)

    print("Loading error data...")
    error_labels = np.load(data_dir / "error_labels_train.npy")
    component_scores = np.load(data_dir / "component_scores_train.npy")

    with open(data_dir / "error_metadata.json") as f:
        error_metadata = json.load(f)

    error_types = error_metadata.get('error_types', list(ERROR_SEVERITY.keys()))
    num_error_types = len(error_types)

    print(f"Error types: {num_error_types}")
    print(f"Generating {num_samples} ranking samples...")

    X_data = []
    y_data = []

    # Create severity lookup
    severity_weights = np.array([
        ERROR_SEVERITY.get(et, 0.5) for et in error_types
    ])

    for i in range(num_samples):
        # Sample from real error data or generate synthetic
        if i < len(error_labels):
            error_probs = error_labels[i].astype(np.float32)
            comp_scores = component_scores[i]
        else:
            # Generate synthetic error combinations
            error_probs = np.random.random(num_error_types).astype(np.float32)
            error_probs = (error_probs > 0.7).astype(np.float32) * np.random.random(num_error_types)
            comp_scores = np.random.random(4).astype(np.float32)

        # Add context features
        user_skill = np.random.random()  # 0 = beginner, 1 = expert
        sign_difficulty = np.random.random()  # 0 = easy, 1 = hard

        # Combine features
        features = np.concatenate([
            comp_scores,           # 4 component scores
            error_probs,           # 16 error probabilities
            [user_skill],          # 1 user skill level
            [sign_difficulty]      # 1 sign difficulty
        ])

        X_data.append(features)

        # Compute priority score
        # Higher priority for: high severity errors, beginner users, difficult signs
        base_priority = np.sum(error_probs * severity_weights)
        skill_modifier = 1.0 + (1.0 - user_skill) * 0.3  # Beginners need more help
        difficulty_modifier = 1.0 + sign_difficulty * 0.2  # Hard signs need more feedback

        # Also consider component scores (lower score = higher priority)
        component_priority = np.sum(1.0 - comp_scores) / 4.0

        priority = (base_priority * skill_modifier * difficulty_modifier + component_priority) / 3.0
        priority = np.clip(priority, 0, 1)

        y_data.append(priority)

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.float32)

    print(f"Generated X shape: {X.shape}")
    print(f"Priority range: [{y.min():.3f}, {y.max():.3f}]")

    return X, y, error_types


def build_model(input_dim):
    """
    Build Feedback Ranker model.

    Simple MLP for ranking priority scores.
    """
    model = Sequential([
        Input(shape=(input_dim,), name='feedback_features'),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(16, activation='relu'),

        Dense(1, activation='sigmoid', name='priority')
    ])

    return model


def train(args):
    """Main training function."""
    print("=" * 60)
    print("FEEDBACK RANKER TRAINING")
    print("=" * 60)

    # Generate training data
    X, y, error_types = generate_ranking_data(args.data_dir, args.num_samples)

    # Split into train/val/test
    n_total = len(X)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")

    # Build model
    print("\nBuilding model...")
    model = build_model(input_dim)
    model.summary()

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
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
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {results[0]:.4f}")
    print(f"Test MAE: {results[1]:.4f}")
    print(f"Test RMSE: {results[2]:.4f}")

    # Compute ranking accuracy (pairwise comparison)
    print("\nComputing ranking accuracy...")
    y_pred = model.predict(X_test, verbose=0).flatten()

    correct_pairs = 0
    total_pairs = 0

    # Sample pairs for evaluation
    n_pairs = min(10000, len(X_test) * (len(X_test) - 1) // 2)
    pair_indices = np.random.choice(len(X_test), size=(n_pairs, 2), replace=True)

    for i, j in pair_indices:
        if i == j:
            continue
        total_pairs += 1

        # Check if ranking order is preserved
        if (y_test[i] > y_test[j]) == (y_pred[i] > y_pred[j]):
            correct_pairs += 1
        elif y_test[i] == y_test[j]:
            correct_pairs += 1  # Ties are acceptable

    ranking_accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0
    print(f"Ranking Accuracy: {ranking_accuracy:.4f} ({ranking_accuracy*100:.2f}%)")

    # Save model
    model.save(MODEL_DIR / 'feedback_ranker.keras')
    print(f"\nModel saved to {MODEL_DIR / 'feedback_ranker.keras'}")

    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(MODEL_DIR / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)

    # Save model info
    model_info = {
        'input_dim': input_dim,
        'parameters': int(model.count_params()),
        'test_mse': float(results[0]),
        'test_mae': float(results[1]),
        'test_rmse': float(results[2]),
        'ranking_accuracy': float(ranking_accuracy),
        'epochs_trained': len(history.history['loss']),
        'error_types': error_types,
        'feature_names': (
            ['comp_handshape', 'comp_location', 'comp_movement', 'comp_orientation'] +
            [f'error_{i}' for i in range(len(error_types))] +
            ['user_skill', 'sign_difficulty']
        )
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
    parser = argparse.ArgumentParser(description='Train Feedback Ranker')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                        help='Directory containing error data')
    parser.add_argument('--num-samples', type=int, default=100000,
                        help='Number of training samples to generate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')

    args = parser.parse_args()
    train(args)
