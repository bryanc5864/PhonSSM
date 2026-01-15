"""
Master Training Script
======================
Trains all SignSense models in sequence.

Models:
1. Sign Classifier (Bi-LSTM) - ~385K params
2. Error Diagnosis Network (Multi-task CNN-LSTM) - ~620K params
3. Movement Pattern Analyzer (1D CNN) - ~45K params
4. Feedback Ranker (MLP) - ~50K params

After training, converts all models to TFLite format.

Usage:
    python training/train_all.py
    python training/train_all.py --skip-classifier  # Skip specific models
    python training/train_all.py --epochs 50 --batch-size 32  # Custom params
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent


def run_training(script_name, extra_args=None):
    """Run a training script and return success status."""
    script_path = PROJECT_ROOT / "training" / script_name

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Train all SignSense models')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs for all models')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for all models')
    parser.add_argument('--skip-classifier', action='store_true',
                        help='Skip Sign Classifier training')
    parser.add_argument('--skip-diagnosis', action='store_true',
                        help='Skip Error Diagnosis training')
    parser.add_argument('--skip-movement', action='store_true',
                        help='Skip Movement Analyzer training')
    parser.add_argument('--skip-ranker', action='store_true',
                        help='Skip Feedback Ranker training')
    parser.add_argument('--skip-convert', action='store_true',
                        help='Skip TFLite conversion')
    parser.add_argument('--convert-only', action='store_true',
                        help='Only run TFLite conversion')

    args = parser.parse_args()

    print("=" * 60)
    print("SIGNSENSE MODEL TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}
    common_args = ['--epochs', str(args.epochs), '--batch-size', str(args.batch_size)]

    if not args.convert_only:
        # 1. Sign Classifier
        if not args.skip_classifier:
            print("\n[1/4] SIGN CLASSIFIER")
            results['sign_classifier'] = run_training(
                'train_classifier.py',
                common_args + ['--use-class-weights']
            )
        else:
            print("\n[1/4] SIGN CLASSIFIER - SKIPPED")
            results['sign_classifier'] = 'skipped'

        # 2. Error Diagnosis Network
        if not args.skip_diagnosis:
            print("\n[2/4] ERROR DIAGNOSIS NETWORK")
            results['error_diagnosis'] = run_training(
                'train_diagnosis.py',
                common_args
            )
        else:
            print("\n[2/4] ERROR DIAGNOSIS NETWORK - SKIPPED")
            results['error_diagnosis'] = 'skipped'

        # 3. Movement Pattern Analyzer
        if not args.skip_movement:
            print("\n[3/4] MOVEMENT PATTERN ANALYZER")
            results['movement_analyzer'] = run_training(
                'train_movement.py',
                ['--epochs', str(args.epochs), '--batch-size', '128']
            )
        else:
            print("\n[3/4] MOVEMENT PATTERN ANALYZER - SKIPPED")
            results['movement_analyzer'] = 'skipped'

        # 4. Feedback Ranker
        if not args.skip_ranker:
            print("\n[4/4] FEEDBACK RANKER")
            results['feedback_ranker'] = run_training(
                'train_ranker.py',
                ['--epochs', str(args.epochs), '--batch-size', '256']
            )
        else:
            print("\n[4/4] FEEDBACK RANKER - SKIPPED")
            results['feedback_ranker'] = 'skipped'

    # 5. TFLite Conversion
    if not args.skip_convert:
        print("\n[FINAL] TFLITE CONVERSION")
        results['tflite_conversion'] = run_training('convert_tflite.py')
    else:
        print("\n[FINAL] TFLITE CONVERSION - SKIPPED")
        results['tflite_conversion'] = 'skipped'

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for model, status in results.items():
        if status == 'skipped':
            icon = "⏭️"
            status_str = "Skipped"
        elif status:
            icon = "✅"
            status_str = "Success"
        else:
            icon = "❌"
            status_str = "Failed"

        print(f"  {icon} {model}: {status_str}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Return exit code based on results
    failed = [k for k, v in results.items() if v is False]
    if failed:
        print(f"\nFailed models: {', '.join(failed)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
