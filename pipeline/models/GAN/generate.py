#!/usr/bin/env python3
"""
GAN Generation Script for Pipeline Integration

Generates synthetic scRNA-seq data using a trained GAN model,
matching the condition distributions from the original dataset.

Usage:
    python generate.py --data path/to/data.h5ad --checkpoint-dir path/to/checkpoint --output path/to/output.h5ad
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import scanpy as sc
import anndata as ad
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Add directories to path
script_dir = Path(__file__).parent
pipeline_dir = script_dir.parent.parent
cvae_root = pipeline_dir.parent

sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(pipeline_dir))

# Load ConditionProcessorGeneric from GAN's local utils using explicit path
import importlib.util
spec = importlib.util.spec_from_file_location(
    "gan_utils",
    script_dir / "utils" / "condition_processor_generic.py"
)
gan_utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gan_utils_module)
ConditionProcessorGeneric = gan_utils_module.ConditionProcessorGeneric

from models import build_generator


def get_condition_combinations(data: pd.DataFrame, condition_columns: List[str]):
    """Get all unique combinations of conditions and their counts."""
    data_copy = data.copy()

    for col in condition_columns:
        if col not in data_copy.columns:
            raise ValueError(f"Column {col} not found! Available: {list(data_copy.columns)}")
        data_copy[col] = data_copy[col].astype(str)

    condition_data = data_copy[condition_columns]
    combination_counts = condition_data.value_counts().sort_values(ascending=False)

    condition_dicts = []
    counts = []

    for combination, count in combination_counts.items():
        condition_dict = {}
        for i, col in enumerate(condition_columns):
            condition_dict[col] = combination[i] if isinstance(combination, tuple) else combination
        condition_dicts.append(condition_dict)
        counts.append(count)

    return condition_dicts, counts


def save_synthetic_data(
    generated_data_list: List[np.ndarray],
    condition_dicts: List[Dict],
    output_path: str,
    gene_names: Optional[List[str]] = None
) -> ad.AnnData:
    """Save synthetic data as AnnData object."""
    X_combined = np.vstack(generated_data_list)
    n_cells, n_genes = X_combined.shape

    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(n_genes)]

    # Create obs (cell metadata)
    obs_data = []
    for condition_dict, data_array in zip(condition_dicts, generated_data_list):
        n_samples = data_array.shape[0]
        for _ in range(n_samples):
            obs_data.append(condition_dict.copy())

    obs_df = pd.DataFrame(obs_data)
    obs_df.index = [f"Cell_{i}" for i in range(n_cells)]

    # Create var DataFrame
    var_df = pd.DataFrame(index=gene_names)
    var_df['gene_ids'] = gene_names

    # Create AnnData
    adata_synthetic = ad.AnnData(X=X_combined, obs=obs_df, var=var_df)

    # Save
    adata_synthetic.write_h5ad(output_path)

    return adata_synthetic


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate synthetic scRNA-seq data using trained GAN'
    )

    parser.add_argument('--data', required=True,
                        help='Path to reference h5ad file (for condition distributions)')
    parser.add_argument('--checkpoint-dir', required=True,
                        help='Path to checkpoint directory (e.g., run_dir/best_model)')
    parser.add_argument('--output', required=True,
                        help='Output path for synthetic data (.h5ad)')
    parser.add_argument('--condition-cols', required=True,
                        help='Comma-separated condition column names')

    parser.add_argument('--latent-dim', type=int, default=128,
                        help='Latent dimension (must match trained model)')
    parser.add_argument('--gen-layers', default='256,512,1024',
                        help='Generator hidden layers (must match trained model)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for generation (default: 64)')
    parser.add_argument('--scale-factor', type=float, default=1.0,
                        help='Scale factor for number of samples (default: 1.0)')

    return parser.parse_args()


def generate_synthetic_data(args):
    """Main generation function."""
    print("=" * 60)
    print("GAN Synthetic Data Generation")
    print("=" * 60)

    condition_cols = [c.strip() for c in args.condition_cols.split(',')]
    gen_layers = [int(d.strip()) for d in args.gen_layers.split(',')]

    print(f"\nConfiguration:")
    print(f"  Reference data: {args.data}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Output: {args.output}")
    print(f"  Conditions: {condition_cols}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Scale factor: {args.scale_factor}")

    # =========================================================================
    # LOAD REFERENCE DATA
    # =========================================================================
    print("\n" + "-" * 40)
    print("Loading reference data...")

    try:
        adata = sc.read_h5ad(args.data)
        print(f"[OK] Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

        # Verify condition columns
        missing = [c for c in condition_cols if c not in adata.obs.columns]
        if missing:
            print(f"[ERROR] Missing columns: {missing}")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return False

    n_genes = adata.n_vars

    # =========================================================================
    # GET CONDITION COMBINATIONS
    # =========================================================================
    print("\n" + "-" * 40)
    print("Analyzing condition combinations...")

    condition_dicts, counts = get_condition_combinations(adata.obs, condition_cols)

    # Apply scale factor
    if args.scale_factor != 1.0:
        counts = [int(c * args.scale_factor) for c in counts]

    print(f"[OK] Found {len(condition_dicts)} unique combinations")
    for i, (cond, count) in enumerate(zip(condition_dicts[:5], counts[:5])):
        print(f"     {i+1}. {cond} -> {count} samples")
    if len(condition_dicts) > 5:
        print(f"     ... and {len(condition_dicts) - 5} more")

    total_samples = sum(counts)
    print(f"[INFO] Total samples to generate: {total_samples}")

    # =========================================================================
    # LOAD CONDITION PROCESSOR
    # =========================================================================
    print("\n" + "-" * 40)
    print("Loading condition processor...")

    checkpoint_dir = Path(args.checkpoint_dir)

    # Try multiple locations for condition config
    cond_config_paths = [
        checkpoint_dir / "condition_config.json",
        checkpoint_dir.parent / "condition_config.json",
    ]

    condition_processor = ConditionProcessorGeneric(condition_cols)
    loaded_config = False

    for path in cond_config_paths:
        if path.exists():
            print(f"[INFO] Loading from: {path}")
            condition_processor.load_config(str(path))
            loaded_config = True
            break

    if not loaded_config:
        print("[INFO] No saved config found, fitting from reference data")
        condition_processor.fit(adata)

    print(f"[OK] Condition dim: {condition_processor.total_condition_dim}")

    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    print("\n" + "-" * 40)
    print("Loading GAN model...")

    # Load training state for config
    state_path = checkpoint_dir / "training_state.json"
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
        print(f"[INFO] Loaded training state from checkpoint")

    try:
        condition_dim = condition_processor.total_condition_dim

        generator = build_generator(
            latent_dim=args.latent_dim,
            condition_dim=condition_dim,
            output_dim=n_genes,
            hidden_layers=gen_layers
        )

        # Initialize model with dummy forward pass
        dummy_noise = tf.random.normal((1, args.latent_dim))
        dummy_conditions = tf.random.normal((1, condition_dim))
        _ = generator((dummy_noise, dummy_conditions), training=False)

        # Load weights - search in multiple possible locations
        generator_paths = [
            checkpoint_dir / "generator.weights.h5",                      # Direct path
            checkpoint_dir / "best_model" / "generator.weights.h5",       # Best model subdir
            checkpoint_dir / "final_model" / "generator.weights.h5",      # Final model subdir
        ]

        # Also search in checkpoint subdirectories
        checkpoints_dir = checkpoint_dir / "checkpoints"
        if checkpoints_dir.exists():
            checkpoint_subdirs = sorted(checkpoints_dir.glob("checkpoint_step_*"),
                                        key=lambda x: x.stat().st_mtime, reverse=True)
            for subdir in checkpoint_subdirs[:3]:  # Check most recent 3
                generator_paths.append(subdir / "generator.weights.h5")

        generator_path = None
        for path in generator_paths:
            if path.exists():
                generator_path = path
                break

        if generator_path is None:
            print(f"[ERROR] Generator weights not found. Searched:")
            for p in generator_paths[:5]:
                print(f"       - {p}")
            return False

        print(f"[INFO] Loading weights from: {generator_path}")
        generator.load_weights(str(generator_path))
        print(f"[OK] Generator loaded: {generator.count_params():,} parameters")

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # GENERATE SYNTHETIC DATA
    # =========================================================================
    print("\n" + "-" * 40)
    print("Generating synthetic data...")

    generated_data_list = []
    batch_size = args.batch_size

    for i, (condition_dict, count) in enumerate(zip(condition_dicts, counts)):
        if count == 0:
            continue

        print(f"  [{i+1}/{len(condition_dicts)}] {condition_dict} -> {count} samples")

        # Encode condition
        condition_df = pd.DataFrame([condition_dict] * count)
        encoded_conditions = condition_processor.encode_conditions(condition_df)

        # Generate in batches
        condition_samples = []

        for j in range(0, count, batch_size):
            current_batch = min(batch_size, count - j)
            batch_conditions = encoded_conditions[j:j+current_batch]

            noise = tf.random.normal((current_batch, args.latent_dim))
            synthetic_batch = generator((noise, batch_conditions), training=False)
            synthetic_batch = tf.maximum(synthetic_batch, 0).numpy()  # ReLU for non-negative
            condition_samples.append(synthetic_batch)

        condition_data = np.vstack(condition_samples)
        generated_data_list.append(condition_data)

    print(f"[OK] Generated data for {len(generated_data_list)} condition combinations")

    # =========================================================================
    # SAVE OUTPUT
    # =========================================================================
    print("\n" + "-" * 40)
    print("Saving synthetic data...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter out zero-count conditions
    valid_conditions = [(cd, gd) for cd, c, gd in
                        zip(condition_dicts, counts, generated_data_list) if c > 0]
    valid_condition_dicts = [cd for cd, gd in valid_conditions]
    valid_generated = [gd for cd, gd in valid_conditions]

    adata_synthetic = save_synthetic_data(
        generated_data_list=valid_generated,
        condition_dicts=valid_condition_dicts,
        output_path=str(output_path),
        gene_names=adata.var_names.tolist()
    )

    print(f"[OK] Saved: {output_path}")
    print(f"     Shape: {adata_synthetic.shape}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Original: {adata.n_obs} cells")
    print(f"Synthetic: {adata_synthetic.n_obs} cells")
    print(f"Genes: {adata_synthetic.n_vars}")

    return True


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    args = parse_args()
    success = generate_synthetic_data(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
