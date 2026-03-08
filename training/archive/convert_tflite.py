"""
TensorFlow Lite Conversion Script
==================================
Converts trained Keras models to TFLite format for mobile deployment.

Supports quantization options:
- float32: Full precision (largest size)
- float16: Half precision (50% smaller, minimal accuracy loss)
- int8: Full integer quantization (smallest, requires representative dataset)

Usage:
    python training/convert_tflite.py
    python training/convert_tflite.py --quantize float16
    python training/convert_tflite.py --model sign_classifier --quantize int8
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import json
import argparse
from pathlib import Path

import tensorflow as tf

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "merged"

# Model configurations
MODELS = {
    'sign_classifier': {
        'keras_path': MODELS_DIR / 'sign_classifier' / 'sign_classifier.keras',
        'tflite_path': MODELS_DIR / 'sign_classifier' / 'sign_classifier.tflite',
        'quantize': 'int8',
        'target_size_mb': 1.6
    },
    'error_diagnosis': {
        'keras_path': MODELS_DIR / 'error_diagnosis' / 'error_diagnosis.keras',
        'tflite_path': MODELS_DIR / 'error_diagnosis' / 'error_diagnosis.tflite',
        'quantize': 'float16',
        'target_size_mb': 2.4
    },
    'movement_analyzer': {
        'keras_path': MODELS_DIR / 'movement_analyzer' / 'movement_analyzer.keras',
        'tflite_path': MODELS_DIR / 'movement_analyzer' / 'movement_analyzer.tflite',
        'quantize': 'int8',
        'target_size_mb': 0.8
    },
    'feedback_ranker': {
        'keras_path': MODELS_DIR / 'feedback_ranker' / 'feedback_ranker.keras',
        'tflite_path': MODELS_DIR / 'feedback_ranker' / 'feedback_ranker.tflite',
        'quantize': 'int8',
        'target_size_mb': 0.2
    }
}


def representative_dataset_generator(data_path, num_samples=100):
    """Generate representative dataset for int8 quantization."""
    X = np.load(data_path)

    # Randomly sample
    indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)

    for i in indices:
        yield [X[i:i+1].astype(np.float32)]


def convert_to_tflite(keras_path, tflite_path, quantize='float16', representative_data=None):
    """
    Convert Keras model to TFLite.

    Args:
        keras_path: Path to Keras model (.keras or .h5)
        tflite_path: Output path for TFLite model
        quantize: Quantization type ('float32', 'float16', 'int8')
        representative_data: Path to representative data (for int8)

    Returns:
        Size of converted model in MB
    """
    print(f"\nLoading model from {keras_path}...")
    model = tf.keras.models.load_model(keras_path)

    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantize == 'float16':
        converter.target_spec.supported_types = [tf.float16]
        print("Applying float16 quantization...")

    elif quantize == 'int8':
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.float32  # Keep input as float
        converter.inference_output_type = tf.float32  # Keep output as float

        if representative_data is not None:
            print("Applying int8 quantization with representative dataset...")
            converter.representative_dataset = lambda: representative_dataset_generator(representative_data)
        else:
            print("Applying int8 quantization (dynamic range)...")

    elif quantize == 'float32':
        print("No quantization (float32)...")
        converter.optimizations = []

    # Convert
    print("Converting...")
    tflite_model = converter.convert()

    # Save
    tflite_path = Path(tflite_path)
    tflite_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"Saved to {tflite_path}")
    print(f"Size: {size_mb:.2f} MB")

    return size_mb


def verify_tflite(tflite_path, test_input_shape):
    """Verify TFLite model works correctly."""
    print(f"\nVerifying {tflite_path}...")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output(s): {len(output_details)}")

    # Test inference
    test_input = np.random.randn(*test_input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    for i, output in enumerate(output_details):
        result = interpreter.get_tensor(output['index'])
        print(f"Output {i} shape: {result.shape}, dtype: {result.dtype}")

    print("Verification passed!")
    return True


def convert_all(args):
    """Convert all models to TFLite."""
    print("=" * 60)
    print("TFLITE CONVERSION")
    print("=" * 60)

    results = []

    for model_name, config in MODELS.items():
        if args.model and args.model != model_name:
            continue

        keras_path = config['keras_path']
        tflite_path = config['tflite_path']
        quantize = args.quantize or config['quantize']

        print(f"\n{'='*60}")
        print(f"Converting: {model_name}")
        print(f"{'='*60}")

        if not keras_path.exists():
            print(f"WARNING: Model not found at {keras_path}")
            print("Train the model first, then convert.")
            continue

        # Determine representative data path
        rep_data = None
        if quantize == 'int8':
            rep_data = DATA_DIR / 'X_train.npy'
            if not rep_data.exists():
                rep_data = None

        try:
            size_mb = convert_to_tflite(keras_path, tflite_path, quantize, rep_data)

            results.append({
                'model': model_name,
                'quantize': quantize,
                'size_mb': size_mb,
                'target_mb': config['target_size_mb'],
                'status': 'success'
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'model': model_name,
                'status': 'failed',
                'error': str(e)
            })

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)

    total_size = 0
    for r in results:
        if r['status'] == 'success':
            status = f"{r['size_mb']:.2f} MB"
            if r['size_mb'] <= r['target_mb']:
                status += " ✓"
            else:
                status += f" (target: {r['target_mb']} MB)"
            total_size += r['size_mb']
        else:
            status = f"FAILED: {r.get('error', 'unknown')}"

        print(f"  {r['model']}: {status}")

    print(f"\nTotal size: {total_size:.2f} MB")
    print(f"Target total: ~5 MB")

    # Save conversion info
    info_path = MODELS_DIR / 'tflite_info.json'
    with open(info_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nConversion info saved to {info_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert models to TFLite')
    parser.add_argument('--model', type=str, choices=list(MODELS.keys()),
                        help='Specific model to convert (default: all)')
    parser.add_argument('--quantize', type=str, choices=['float32', 'float16', 'int8'],
                        help='Quantization type (default: per-model config)')

    args = parser.parse_args()
    convert_all(args)
