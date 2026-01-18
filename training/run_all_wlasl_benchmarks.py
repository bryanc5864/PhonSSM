"""Run all WLASL benchmarks sequentially: 300, 1000, 2000."""
import subprocess
import sys
from datetime import datetime

benchmarks = [
    {"subset": 300, "epochs": 100},
    {"subset": 1000, "epochs": 100},
    {"subset": 2000, "epochs": 100},
]

for bench in benchmarks:
    print("\n" + "="*70)
    print(f"STARTING WLASL{bench['subset']} BENCHMARK")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    cmd = [
        sys.executable,
        "training/benchmark_external.py",
        "--dataset", "wlasl",
        "--subset", str(bench["subset"]),
        "--epochs", str(bench["epochs"]),
        "--device", "cuda"
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nWARNING: WLASL{bench['subset']} exited with code {result.returncode}")

    print(f"\nCompleted WLASL{bench['subset']}")

print("\n" + "="*70)
print("ALL BENCHMARKS COMPLETE")
print("="*70)
