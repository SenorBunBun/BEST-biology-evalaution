#!/usr/bin/env python3
"""
CVAE Generation Script for Pipeline Integration

Generates synthetic scRNA-seq data using a trained CVAE model,
matching the condition distributions from the original dataset.

Based on cvae_generate_dataset_cancer.py - uses the model's built-in label encoders.

Usage:
    python generate.py --data path/to/data.h5ad --model path/to/model.pt --output path/to/output.h5ad
"""

import argparse
import json
import os
import sys
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import LabelEncoder

# Add parent directories to path for imports
script_dir = Path(__file__).parent
pipeline_dir = script_dir.parent.parent
cvae_root = pipeline_dir.parent

sys.path.insert(0, str(cvae_root))
sys.path.insert(0, str(pipeline_dir))

from src.unbatch_vanilla_cvae_relu_qol_categorical import Enhanced_Vanilla_CVAE_Categorical


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
            condition_dict[col] = combination[i]
        condition_dicts.append(condition_dict)
        counts.append(count)

    return condition_dicts, counts


def save_synthetic_data(
    generated_data_list: List[np.ndarray],
    condition_dicts: List[Dict],
    output_path: str,
    gene_names: Optional[List[str]] = None,
    condition_columns: Optional[List[str]] = None
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
        description='Generate synthetic scRNA-seq data using trained CVAE'
    )

    parser.add_argument('--data', required=True,
                        help='Path to reference h5ad file (for condition distributions)')
    parser.add_argument('--model', required=True,
                        help='Path to trained CVAE model (.pt file)')
    parser.add_argument('--output', required=True,
                        help='Output path for synthetic data (.h5ad)')
    parser.add_argument('--condition-cols', required=True,
                        help='Comma-separated condition column names')
    parser.add_argument('--latent-size', type=int, default=32,
                        help='Latent space dimension (must match model)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for generation (default: 50)')
    parser.add_argument('--scale-factor', type=float, default=1.0,
                        help='Scale factor for number of samples (default: 1.0)')

    return parser.parse_args()


def encode_conditions_with_model_encoders(condition_dicts, label_encoders, condition_cols):
    """
    Encode conditions using the model's trained label encoders.

    Args:
        condition_dicts: List of condition dictionaries
        label_encoders: The model's label_encoders dictionary (from model.label_encoders)
        condition_cols: List of condition column names

    Returns:
        np.ndarray of shape (n_samples, n_labels)
    """
    encoded_conditions = []

    for condition_dict in condition_dicts:
        encoded = []
        for col in condition_cols:
            if col not in condition_dict:
                raise ValueError(f"Column {col} missing from condition dict!")
            if col not in label_encoders:
                raise ValueError(f"Column {col} not in model's label encoders!")

            val = condition_dict[col]
            encoder = label_encoders[col]

            if encoder is not None:
                # Categorical column - use label encoder
                encoded_val = encoder.transform([str(val)])[0]
            else:
                # Continuous/numeric column - use value directly as float
                encoded_val = float(val)

            encoded.append(encoded_val)

        encoded_conditions.append(encoded)

    return np.array(encoded_conditions, dtype=np.float32)


def generate_synthetic_data(args):
    """Main generation function - based on cvae_generate_dataset_cancer.py."""
    print("=" * 60)
    print("CVAE Synthetic Data Generation")
    print("=" * 60)

    condition_cols = [c.strip() for c in args.condition_cols.split(',')]

    print(f"\nConfiguration:")
    print(f"  Reference data: {args.data}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print(f"  Conditions: {condition_cols}")
    print(f"  Latent size: {args.latent_size}")
    print(f"  Scale factor: {args.scale_factor}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

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
            print(f"Available: {list(adata.obs.columns)}")
            sys.exit(1)

    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)

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
    # LOAD MODEL
    # =========================================================================
    print("\n" + "-" * 40)
    print("Loading CVAE model...")

    n_genes = adata.n_vars
    n_labels = len(condition_cols)
    model_path = Path(args.model)

    # Check model file exists
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    try:
        model = Enhanced_Vanilla_CVAE_Categorical(
            n_genes=n_genes,
            n_labels=n_labels,
            latent_size=args.latent_size,
            device=device,
            condition_cols=condition_cols
        )

        model.load_state_dict(torch.load(args.model, map_location=device))
        model.to(device)
        model.eval()
        print(f"[OK] Model loaded successfully")

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # LOAD LABEL ENCODERS
    # =========================================================================
    print("\n" + "-" * 40)
    print("Loading label encoders...")

    # Try multiple locations for label encoders
    encoder_paths = [
        model_path.parent / "label_encoders.pkl",  # Pipeline format
        model_path.with_suffix('').parent / "label_encoders.pkl",  # Alternate
        Path(str(model_path).replace('.pt', '_label_encoders.pkl')),  # Legacy format
    ]

    label_encoders = None
    for enc_path in encoder_paths:
        if enc_path.exists():
            print(f"[INFO] Loading encoders from: {enc_path}")
            try:
                with open(enc_path, 'rb') as f:
                    data = pickle.load(f)
                    # Handle both formats: {'encoders': ...} or raw dict
                    if isinstance(data, dict) and 'encoders' in data:
                        label_encoders = data['encoders']
                    else:
                        label_encoders = data
                print(f"[OK] Loaded label encoders for: {list(label_encoders.keys())}")
                break
            except Exception as e:
                print(f"[WARN] Could not load from {enc_path}: {e}")

    if label_encoders is None:
        print("[ERROR] Could not load label encoders from any location!")
        print("       Make sure label_encoders.pkl exists in the model directory.")
        sys.exit(1)

    # Verify all condition columns have encoders
    missing_encoders = [col for col in condition_cols if col not in label_encoders]
    if missing_encoders:
        print(f"[ERROR] Missing encoders for columns: {missing_encoders}")
        sys.exit(1)

    # =========================================================================
    # GENERATE SYNTHETIC DATA
    # =========================================================================
    print("\n" + "-" * 40)
    print("Generating synthetic data...")

    generated_data_list = []
    valid_condition_dicts = []
    batch_size = args.batch_size

    for i, (condition_dict, count) in enumerate(zip(condition_dicts, counts)):
        if count == 0:
            continue

        print(f"  [{i+1}/{len(condition_dicts)}] {condition_dict} -> {count} samples")

        try:
            # Encode condition using model's label encoders
            encoded_condition = encode_conditions_with_model_encoders(
                [condition_dict], label_encoders, condition_cols
            )[0]
            condition_tensor = torch.tensor(encoded_condition, dtype=torch.float32).to(device)

            # Generate in batches
            condition_samples = []

            for j in range(0, count, batch_size):
                current_batch = min(batch_size, count - j)

                with torch.no_grad():
                    batch_condition = condition_tensor.unsqueeze(0).repeat(current_batch, 1)
                    synthetic_batch = model.generate(batch_condition[0:1], num_samples=current_batch)
                    synthetic_batch = torch.clamp(synthetic_batch.cpu(), min=0).numpy()
                    condition_samples.append(synthetic_batch)

            condition_data = np.vstack(condition_samples)
            generated_data_list.append(condition_data)
            valid_condition_dicts.append(condition_dict)

        except Exception as e:
            print(f"    [WARN] Failed to generate for condition {condition_dict}: {e}")
            continue

    if len(generated_data_list) == 0:
        print("[ERROR] No synthetic data was generated!")
        sys.exit(1)

    print(f"[OK] Generated data for {len(generated_data_list)} condition combinations")

    # =========================================================================
    # SAVE OUTPUT
    # =========================================================================
    print("\n" + "-" * 40)
    print("Saving synthetic data...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata_synthetic = save_synthetic_data(
        generated_data_list=generated_data_list,
        condition_dicts=valid_condition_dicts,
        output_path=str(output_path),
        gene_names=adata.var_names.tolist(),
        condition_columns=condition_cols
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
    args = parse_args()
    success = generate_synthetic_data(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
