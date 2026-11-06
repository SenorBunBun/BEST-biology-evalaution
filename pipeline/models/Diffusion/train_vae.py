#!/usr/bin/env python3
"""
VAE Training Script for Diffusion Pipeline

This is a convenience wrapper that calls train.py with --vae-only flag.
For full control, use train.py directly.

Usage:
    python train_vae.py --data path/to/data.h5ad --output-dir path/to/output [other args...]

This is equivalent to:
    python train.py --data path/to/data.h5ad --output-dir path/to/output --vae-only [other args...]
"""

import sys
from pathlib import Path

# Insert --vae-only flag and call train.py
if __name__ == "__main__":
    # Add --vae-only to arguments
    args = sys.argv[1:]
    if '--vae-only' not in args:
        args.append('--vae-only')

    # Import and run the main train function
    sys.argv = [sys.argv[0]] + args

    # Import the main module
    from train import main
    main()
