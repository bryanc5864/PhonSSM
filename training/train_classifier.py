"""
Sign Classifier Training Script
================================
Bi-directional LSTM that identifies which sign the user is attempting.

Architecture:
    Input: (30 frames, 63 features) - 21 landmarks × 3 coords
    → Bidirectional LSTM (128 units)
    → Bidirectional LSTM (64 units)
    → Dense (128, ReLU) + Dropout(0.3)
    → Dense (num_classes, Softmax)

Target metrics:
    - Top-1 accuracy: >85%
    - Top-3 accuracy: >95%
    - Parameters: ~385K

Usage:
    python training/train_classifier.py
    python training/train_classifier.py --epochs 100 --batch-size 64
    python training/train_classifier.py --data-dir data/processed/merged
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout,
    BatchNormalization, Masking
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "merged"
MODEL_DIR = PROJECT_ROOT / "models" / "sign_classifier"


def load_data(data_dir):
    """Load training, validation, and test data."""
    data_dir = Path(data_dir)

    print("Loading data...")
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    with open(data_dir / "label_map.json") as f:
        label_map = json.load(f)

    num_classes = len(label_map)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Classes: {num_classes}")

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, label_map


def build_model(input_shape, num_classes):
    """
    Build Sign Classifier model.

    Architecture from tech doc:
    - Bidirectional LSTM (128) → Bidirectional LSTM (64)
    - Dense (128) + Dropout → Dense (num_classes)
    """
    model = Sequential([
        Input(shape=input_shape, name='landmarks'),

        # Mask zero-padded frames
        Masking(mask_value=0.0),

        # Bidirectional LSTM layers
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)),
        Bidirectional(LSTM(64, dropout=0.2)),

        # Dense layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dropout(0.2),

        # Output layer
        Dense(num_classes, activation='softmax', name='predictions')
    ])

    return model


def train(args):
    """Main training function."""
    print("=" * 60)
    print("SIGN CLASSIFIER TRAINING")
    print("=" * 60)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, label_map = load_data(args.data_dir)

    # Input shape
    input_shape = (X_train.shape[1], X_train.shape[2])  # (30, 63)
    print(f"Input shape: {input_shape}")

    # Convert labels to one-hot
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Compute class weights for imbalanced data
    class_weights = None
    if args.use_class_weights:
        print("Computing class weights...")
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        print(f"Class weights computed for {len(classes)} classes")

    # Build model
    print("\nBuilding model...")
    model = build_model(input_shape, num_classes)
    model.summary()

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
        ]
    )

    # Callbacks
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
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
            monitor='val_accuracy',
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
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    test_results = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
    print(f"Test Top-3 Accuracy: {test_results[2]:.4f} ({test_results[2]*100:.2f}%)")
    print(f"Test Top-5 Accuracy: {test_results[3]:.4f} ({test_results[3]*100:.2f}%)")

    # Save final model
    model.save(MODEL_DIR / 'sign_classifier.keras')
    print(f"\nModel saved to {MODEL_DIR / 'sign_classifier.keras'}")

    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(MODEL_DIR / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)

    # Save model info
    model_info = {
        'input_shape': list(input_shape),
        'num_classes': num_classes,
        'parameters': int(model.count_params()),
        'test_accuracy': float(test_results[1]),
        'test_top3_accuracy': float(test_results[2]),
        'test_top5_accuracy': float(test_results[3]),
        'epochs_trained': len(history.history['loss']),
        'label_map': label_map
    }
    with open(MODEL_DIR / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final test accuracy: {test_results[1]*100:.2f}%")
    print(f"Model parameters: {model.count_params():,}")
    print(f"Saved to: {MODEL_DIR}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Sign Classifier')
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
    parser.add_argument('--use-class-weights', action='store_true',
                        help='Use class weights for imbalanced data')

    args = parser.parse_args()
    train(args)
