"""
Test script to verify training setup before running full training.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "merged"

def test_tensorflow_gpu():
    """Test 1: TensorFlow and GPU detection."""
    print("=" * 50)
    print("TEST 1: TensorFlow and GPU")
    print("=" * 50)

    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")

    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")

    if len(gpus) == 0:
        print("WARNING: No GPU detected!")
        return False

    # Quick GPU test
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
    print(f"GPU computation test: PASSED")
    return True


def test_data_loading():
    """Test 2: Data loading and integrity."""
    print("\n" + "=" * 50)
    print("TEST 2: Data Loading")
    print("=" * 50)

    # Check files exist
    required_files = ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy', 'label_map.json']
    for f in required_files:
        path = DATA_DIR / f
        if not path.exists():
            print(f"ERROR: Missing {path}")
            return False
        print(f"Found: {f}")

    # Load data
    X_train = np.load(DATA_DIR / 'X_train.npy')
    y_train = np.load(DATA_DIR / 'y_train.npy')
    X_val = np.load(DATA_DIR / 'X_val.npy')
    y_val = np.load(DATA_DIR / 'y_val.npy')

    with open(DATA_DIR / 'label_map.json') as f:
        label_map = json.load(f)

    num_classes = len(label_map)

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  num_classes: {num_classes}")

    # Verify labels are in range
    if y_train.max() >= num_classes:
        print(f"ERROR: y_train max ({y_train.max()}) >= num_classes ({num_classes})")
        return False
    if y_val.max() >= num_classes:
        print(f"ERROR: y_val max ({y_val.max()}) >= num_classes ({num_classes})")
        return False

    print(f"\nLabel range check: PASSED")
    print(f"  y_train: [{y_train.min()}, {y_train.max()}]")
    print(f"  y_val: [{y_val.min()}, {y_val.max()}]")

    return True


def test_model_building():
    """Test 3: Model architecture."""
    print("\n" + "=" * 50)
    print("TEST 3: Model Building")
    print("=" * 50)

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
    )

    # Load to get dimensions
    X_train = np.load(DATA_DIR / 'X_train.npy')
    with open(DATA_DIR / 'label_map.json') as f:
        label_map = json.load(f)

    input_shape = (X_train.shape[1], X_train.shape[2])  # (30, 63)
    num_classes = len(label_map)

    print(f"Building model for input_shape={input_shape}, num_classes={num_classes}")

    # Build the same model as train_classifier.py
    model = Sequential([
        Input(shape=input_shape, name='landmarks'),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)),
        Bidirectional(LSTM(64, return_sequences=False, dropout=0.2)),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax', name='predictions')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Model built successfully!")
    print(f"Total parameters: {model.count_params():,}")

    return True


def test_training_step():
    """Test 4: Single training step."""
    print("\n" + "=" * 50)
    print("TEST 4: Training Step (small batch)")
    print("=" * 50)

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
    )

    # Load small subset
    X_train = np.load(DATA_DIR / 'X_train.npy')[:100]
    y_train = np.load(DATA_DIR / 'y_train.npy')[:100]

    with open(DATA_DIR / 'label_map.json') as f:
        label_map = json.load(f)

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(label_map)

    # Smaller model for quick test
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Running 1 epoch on 100 samples...")
    history = model.fit(
        X_train, y_train,
        epochs=1,
        batch_size=32,
        verbose=1
    )

    print(f"Training step completed!")
    print(f"Loss: {history.history['loss'][0]:.4f}")

    return True


def main():
    print("=" * 50)
    print("SIGNSENSE TRAINING SETUP TEST")
    print("=" * 50)

    results = {}

    # Test 1: TensorFlow and GPU
    try:
        results['gpu'] = test_tensorflow_gpu()
    except Exception as e:
        print(f"ERROR: {e}")
        results['gpu'] = False

    # Test 2: Data loading
    try:
        results['data'] = test_data_loading()
    except Exception as e:
        print(f"ERROR: {e}")
        results['data'] = False

    # Test 3: Model building
    try:
        results['model'] = test_model_building()
    except Exception as e:
        print(f"ERROR: {e}")
        results['model'] = False

    # Test 4: Training step
    try:
        results['training'] = test_training_step()
    except Exception as e:
        print(f"ERROR: {e}")
        results['training'] = False

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed! Ready for training.")
        return 0
    else:
        print("\nSome tests failed. Please fix issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
