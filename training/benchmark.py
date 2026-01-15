"""Quick benchmark to estimate training time."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import json
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "merged"

# Load data info
X_train = np.load(DATA_DIR / 'X_train.npy')
y_train = np.load(DATA_DIR / 'y_train.npy')
with open(DATA_DIR / 'label_map.json') as f:
    num_classes = len(json.load(f))

n_samples = len(X_train)
batch_size = 64
steps_per_epoch = n_samples // batch_size

print(f"Building model...")

# Build actual model
model = Sequential([
    Input(shape=(30, 63)),
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)),
    Bidirectional(LSTM(64, return_sequences=False, dropout=0.2)),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Benchmark on 1000 samples
print(f"Benchmarking on 1000 samples...")
X_bench = X_train[:1000]
y_bench = y_train[:1000]

start = time.time()
model.fit(X_bench, y_bench, batch_size=64, epochs=1, verbose=1)
elapsed = time.time() - start

time_per_step = elapsed / (1000 // 64)
time_per_epoch = time_per_step * steps_per_epoch

print(f"\n{'='*50}")
print(f"TRAINING TIME ESTIMATE")
print(f"{'='*50}")
print(f"Training samples: {n_samples:,}")
print(f"Steps per epoch: {steps_per_epoch:,}")
print(f"Time per step: {time_per_step:.3f}s")
print(f"Est. time per epoch: {time_per_epoch/60:.1f} min")
print(f"Est. 50 epochs: {time_per_epoch*50/60:.1f} min ({time_per_epoch*50/3600:.1f} hrs)")
print(f"\nNote: Early stopping may reduce total time significantly.")
