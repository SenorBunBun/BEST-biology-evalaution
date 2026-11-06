#!/usr/bin/env python3
"""
Run training for all jan26 dataset configs (402, 403, 404).
"""

import subprocess
import sys
from pathlib import Path

CONFIGS = [
    "dataset_configs/osdr-402-jan26.json",
    "dataset_configs/osdr-403-jan26.json",
    "dataset_configs/osdr-404-jan26.json",
]

def main():
    script_dir = Path(__file__).parent

    for config in CONFIGS:
        print("\n" + "=" * 60)
        print(f"TRAINING: {config}")
        print("=" * 60 + "\n")

        cmd = [
            sys.executable,
            str(script_dir / "driver.py"),
            "--config", config,
            "--stage", "train"
        ]

        result = subprocess.run(cmd, cwd=str(script_dir))

        if result.returncode != 0:
            print(f"\n[WARN] Training failed for {config}, continuing...")
        else:
            print(f"\n[OK] Training completed for {config}")

    print("\n" + "=" * 60)
    print("ALL TRAINING RUNS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
