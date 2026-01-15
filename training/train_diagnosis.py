"""
Error Diagnosis Network Training Script
========================================
Multi-task CNN-LSTM that identifies what's wrong with an incorrect sign.

Architecture:
    Input: (30 frames, 63 features)
    → Shared Backbone: Conv1D(64) → Conv1D(128) → LSTM(128)
    → Three output heads:
        1. Component scores (4): Handshape, Location, Movement, Orientation
        2. Error types (16): Multi-label classification
        3. Overall correctness (1): Binary classification

Target metrics:
    - Component MAE: <0.12
    - Error type F1: >0.70
    - Parameters: ~620K

Usage:
    python training/train_diagnosis.py
    python training/train_diagnosis.py --epochs 100 --batch-size 64
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
    Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling1D, Concatenate, MaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "error_diagnosis"


def load_data(data_dir):
    """Load training data with error labels."""
    data_dir = Path(data_dir)

    print("Loading data...")

    # Load features with errors
    X_train = np.load(data_dir / "X_train_with_errors.npy")
    y_train = np.load(data_dir / "y_train_with_errors.npy")

    # Load error-specific labels
    error_labels = np.load(data_dir / "error_labels_train.npy")
    component_scores = np.load(data_dir / "component_scores_train.npy")
    is_correct = np.load(data_dir / "is_correct_train.npy")

    # Load error metadata
    with open(data_dir / "error_metadata.json") as f:
        error_metadata = json.load(f)

    print(f"X shape: {X_train.shape}")
    print(f"Error labels shape: {error_labels.shape}")
    print(f"Component scores shape: {component_scores.shape}")
    print(f"Is correct shape: {is_correct.shape}")

    # Split into train/val (80/20 of the error data)
    n_total = len(X_train)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    data = {
        'X_train': X_train[train_idx],
        'X_val': X_train[val_idx],
        'error_train': error_labels[train_idx],
        'error_val': error_labels[val_idx],
        'component_train': component_scores[train_idx],
        'component_val': component_scores[val_idx],
        'correct_train': is_correct[train_idx],
        'correct_val': is_correct[val_idx]
    }

    print(f"\nTrain: {len(data['X_train'])}, Val: {len(data['X_val'])}")

    return data, error_metadata


def build_model(input_shape, num_error_types=16, num_components=4):
    """
    Build Error Diagnosis multi-task model.

    Three output heads:
    1. Component scores (regression): How well each component is performed
    2. Error types (multi-label): Which specific errors are present
    3. Correctness (binary): Overall correct or incorrect
    """
    inputs = Input(shape=input_shape, name='landmarks')

    # Shared backbone: Conv1D → Conv1D → LSTM
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = LSTM(128, return_sequences=False, dropout=0.2)(x)
    x = Dropout(0.3)(x)

    # Head 1: Component scores (4 values: handshape, location, movement, orientation)
    comp = Dense(64, activation='relu')(x)
    comp = Dropout(0.2)(comp)
    comp = Dense(num_components, activation='sigmoid', name='components')(comp)

    # Head 2: Error types (16 multi-label classification)
    err = Dense(64, activation='relu')(x)
    err = Dropout(0.2)(err)
    err = Dense(num_error_types, activation='sigmoid', name='errors')(err)

    # Head 3: Overall correctness (binary)
    correct = Dense(32, activation='relu')(x)
    correct = Dropout(0.2)(correct)
    correct = Dense(1, activation='sigmoid', name='correct')(correct)

    model = Model(inputs, [comp, err, correct])

    return model


def train(args):
    """Main training function."""
    print("=" * 60)
    print("ERROR DIAGNOSIS NETWORK TRAINING")
    print("=" * 60)

    # Load data
    data, error_metadata = load_data(args.data_dir)

    # Input shape
    input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
    num_error_types = data['error_train'].shape[1]
    num_components = data['component_train'].shape[1]

    print(f"\nInput shape: {input_shape}")
    print(f"Error types: {num_error_types}")
    print(f"Components: {num_components}")

    # Build model
    print("\nBuilding model...")
    model = build_model(input_shape, num_error_types, num_components)
    model.summary()

    # Compile with multiple losses
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss={
            'components': 'mse',
            'errors': 'binary_crossentropy',
            'correct': 'binary_crossentropy'
        },
        loss_weights={
            'components': 1.0,
            'errors': 1.0,
            'correct': 0.5
        },
        metrics={
            'components': ['mae'],
            'errors': ['binary_accuracy', tf.keras.metrics.AUC(name='auc')],
            'correct': ['accuracy']
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

    # Prepare training data
    train_inputs = data['X_train']
    train_outputs = {
        'components': data['component_train'],
        'errors': data['error_train'],
        'correct': data['correct_train'].reshape(-1, 1)
    }

    val_inputs = data['X_val']
    val_outputs = {
        'components': data['component_val'],
        'errors': data['error_val'],
        'correct': data['correct_val'].reshape(-1, 1)
    }

    # Train
    print(f"\nTraining for up to {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")

    history = model.fit(
        train_inputs, train_outputs,
        validation_data=(val_inputs, val_outputs),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results = model.evaluate(val_inputs, val_outputs, verbose=0)

    # Parse results (order: loss, comp_loss, err_loss, correct_loss, comp_mae, err_acc, err_auc, correct_acc)
    print(f"Total Loss: {results[0]:.4f}")
    print(f"Component MAE: {results[4]:.4f}")
    print(f"Error Type Accuracy: {results[5]:.4f}")
    print(f"Error Type AUC: {results[6]:.4f}")
    print(f"Correctness Accuracy: {results[7]:.4f}")

    # Save model
    model.save(MODEL_DIR / 'error_diagnosis.keras')
    print(f"\nModel saved to {MODEL_DIR / 'error_diagnosis.keras'}")

    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(MODEL_DIR / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)

    # Save model info
    model_info = {
        'input_shape': list(input_shape),
        'num_error_types': num_error_types,
        'num_components': num_components,
        'parameters': int(model.count_params()),
        'component_mae': float(results[4]),
        'error_accuracy': float(results[5]),
        'error_auc': float(results[6]),
        'correctness_accuracy': float(results[7]),
        'epochs_trained': len(history.history['loss']),
        'error_types': error_metadata.get('error_types', [])
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
    parser = argparse.ArgumentParser(description='Train Error Diagnosis Network')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                        help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')

    args = parser.parse_args()
    train(args)
