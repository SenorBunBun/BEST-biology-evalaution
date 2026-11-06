#!/usr/bin/env python3
"""
Diffusion Model Generation Script for Pipeline Integration

Generates synthetic scRNA-seq data using a trained Diffusion model,
matching the condition distributions from the original dataset.

Usage:
    python generate.py --data path/to/data.h5ad --checkpoint-dir path/to/checkpoints --output path/to/output.h5ad
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

# Add directories to path
script_dir = Path(__file__).parent
pipeline_dir = script_dir.parent.parent
cvae_root = pipeline_dir.parent

sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(cvae_root))
sys.path.insert(0, str(pipeline_dir))

from diffusion_model_cfg import ClassifierFreeDiffusionModel
from vae_diffusion import DiffusionIntegrationVAE
from diffusion_generation_utils import load_model_by_checkpoint, DiffusionModelGenerator, ConditionProcessor


def debug_print_vae_fingerprint(vae_model, label="VAE"):
    """Print VAE weight fingerprint for debugging."""
    print(f"\n[DEBUG] {label} Weight Fingerprint:")
    state_dict = vae_model.state_dict()

    key_layers = ['encoder.0.weight', 'z_mean.weight', 'decoder.0.weight']
    fingerprint_values = []

    for key in key_layers:
        if key in state_dict:
            w = state_dict[key]
            mean_val = w.mean().item()
            std_val = w.std().item()
            sum_val = w.sum().item()
            fingerprint_values.extend([mean_val, std_val, sum_val])
            print(f"  {key}: mean={mean_val:.6f}, std={std_val:.6f}, sum={sum_val:.6f}")

    fingerprint_str = "_".join([f"{v:.6f}" for v in fingerprint_values])
    print(f"  FINGERPRINT: {fingerprint_str}")

    # Check if weights look random (newly initialized)
    if 'encoder.0.weight' in state_dict:
        enc_std = state_dict['encoder.0.weight'].std().item()
        # He initialization for sigmoid typically has std around 0.044 for 2000 input features
        # Trained models usually have different statistics
        if abs(enc_std - 0.044) < 0.005:
            print(f"  [WARNING] Encoder weights appear to be freshly initialized (std={enc_std:.6f})!")

    return fingerprint_str


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
        description='Generate synthetic scRNA-seq data using trained Diffusion model'
    )

    parser.add_argument('--data', required=True,
                        help='Path to reference h5ad file (for condition distributions)')
    parser.add_argument('--checkpoint-dir', required=True,
                        help='Path to checkpoint directory (run folder)')
    parser.add_argument('--output', required=True,
                        help='Output path for synthetic data (.h5ad)')
    parser.add_argument('--condition-cols', required=True,
                        help='Comma-separated condition column names')

    parser.add_argument('--checkpoint-num', type=int, default=None,
                        help='Specific checkpoint number to load (default: latest)')
    parser.add_argument('--vae-latent-size', type=int, default=128,
                        help='VAE latent size (must match model)')
    parser.add_argument('--guidance-scale', type=float, default=2.0,
                        help='Classifier-free guidance scale (default: 2.0)')
    parser.add_argument('--scale-factor', type=float, default=1.0,
                        help='Scale factor for number of samples (default: 1.0)')

    return parser.parse_args()


def generate_synthetic_data(args):
    """Main generation function."""
    print("=" * 60)
    print("Diffusion Model Synthetic Data Generation")
    print("=" * 60)

    condition_cols = [c.strip() for c in args.condition_cols.split(',')]

    print(f"\nConfiguration:")
    print(f"  Reference data: {args.data}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Output: {args.output}")
    print(f"  Conditions: {condition_cols}")
    print(f"  Guidance scale: {args.guidance_scale}")
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
            return False

    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return False

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
    # LOAD MODELS
    # =========================================================================
    print("\n" + "-" * 40)
    print("Loading models...")

    checkpoint_dir = Path(args.checkpoint_dir)
    n_genes = adata.n_vars

    # Find VAE model
    vae_path = checkpoint_dir / "vae_model.pt"
    if not vae_path.exists():
        # Try parent directory
        vae_path = checkpoint_dir.parent / "vae_model.pt"

    # Load VAE config if available
    vae_config_path = checkpoint_dir / "vae_config.json"
    vae_latent_size = args.vae_latent_size
    vae_hidden_dims = [1024, 1024]  # Default

    if vae_config_path.exists():
        with open(vae_config_path, 'r') as f:
            vae_config = json.load(f)
        vae_latent_size = vae_config.get("latent_size", args.vae_latent_size)
        vae_hidden_dims = vae_config.get("hidden_dims", [1024, 1024])
        print(f"[OK] Loaded VAE config: latent={vae_latent_size}, hidden={vae_hidden_dims}")
    else:
        print(f"[WARN] No vae_config.json found, using defaults: latent={vae_latent_size}, hidden={vae_hidden_dims}")

    # Load VAE
    try:
        print(f"[INFO] Loading VAE from: {vae_path}")
        vae_model = DiffusionIntegrationVAE(
            n_genes=n_genes,
            latent_size=vae_latent_size,
            hidden_dims=vae_hidden_dims,
            model=str(vae_path),
            verbose=True,
            auto_load=True
        )
        print("[OK] VAE loaded")

        # Debug: Print weight fingerprint to verify weights loaded correctly (not random)
        # debug_print_vae_fingerprint(vae_model, "Loaded VAE")

    except Exception as e:
        print(f"[ERROR] Failed to load VAE: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load diffusion model
    try:
        # Load condition config to get proper configuration
        condition_config_path = checkpoint_dir / "condition_config.json"
        if condition_config_path.exists():
            with open(condition_config_path, 'r') as f:
                cond_data = json.load(f)
                full_condition_config = cond_data.get('full_condition_config', cond_data.get('condition_config', {}))
        else:
            # Fallback to assuming all categorical
            full_condition_config = {col: 'categorical' for col in condition_cols}
            print(f"[WARN] No condition_config.json found, assuming all categorical")

        # Find checkpoints subdirectory if exists
        checkpoints_subdir = checkpoint_dir / "checkpoints"
        if checkpoints_subdir.exists():
            search_dir = checkpoints_subdir
        else:
            search_dir = checkpoint_dir

        print(f"[INFO] Loading diffusion model from: {search_dir}")

        # First try to load best_model.pt from run directory (copied by train.py)
        best_model_pt = checkpoint_dir / "best_model.pt"
        if best_model_pt.exists():
            print(f"[INFO] Loading from best_model.pt")
            state = torch.load(best_model_pt, map_location=device)

            # Handle different checkpoint formats
            if isinstance(state, dict) and 'model_state_dict' in state:
                model_state = state['model_state_dict']
            else:
                model_state = state

            generator = ClassifierFreeDiffusionModel(
                latent_dim=args.vae_latent_size,
                condition_config=full_condition_config,
                hidden_dims=[512, 512, 256, 128]
            )
            generator.load_state_dict(model_state)

        elif args.checkpoint_num is not None:
            generator = load_model_by_checkpoint(
                directory=str(search_dir),
                checkpoint_num=args.checkpoint_num,
                model_class=ClassifierFreeDiffusionModel
            )
        else:
            # Look for best_model_epoch_*_model.pth files or checkpoint_epoch_*.pth
            best_model_files = list(search_dir.glob("best_model_epoch_*_model.pth"))
            regular_checkpoints = list(search_dir.glob("checkpoint_epoch_*.pth"))
            all_checkpoints = list(search_dir.glob("*.pth")) + list(search_dir.glob("*.pt"))

            if best_model_files:
                latest = max(best_model_files, key=lambda x: x.stat().st_mtime)
            elif regular_checkpoints:
                latest = max(regular_checkpoints, key=lambda x: x.stat().st_mtime)
            elif all_checkpoints:
                latest = max(all_checkpoints, key=lambda x: x.stat().st_mtime)
            else:
                print(f"[ERROR] No checkpoint files found in {search_dir}")
                return False

            print(f"[INFO] Using checkpoint: {latest.name}")

            state = torch.load(latest, map_location=device)

            # Handle different checkpoint formats
            if isinstance(state, dict) and 'model_state_dict' in state:
                model_state = state['model_state_dict']
            else:
                model_state = state

            generator = ClassifierFreeDiffusionModel(
                latent_dim=args.vae_latent_size,
                condition_config=full_condition_config,
                hidden_dims=[512, 512, 256, 128]
            )
            generator.load_state_dict(model_state)

        diffusion_model = generator  # Keep raw model reference
        diffusion_model.to(device)
        diffusion_model.eval()
        print("[OK] Diffusion model loaded")

        # Create condition processor and fit it from reference data
        print("[INFO] Setting up condition processor...")
        condition_processor = ConditionProcessor(full_condition_config)
        condition_processor.fit_conditions(adata)

        # Wrap model in DiffusionModelGenerator for generation
        generator = DiffusionModelGenerator(
            model=diffusion_model,
            condition_processor=condition_processor,
            device=str(device),
            num_timesteps=1000,
            beta_schedule="cosine"
        )
        print("[OK] DiffusionModelGenerator created")

    except Exception as e:
        print(f"[ERROR] Failed to load diffusion model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # GENERATE SYNTHETIC DATA
    # =========================================================================
    print("\n" + "-" * 40)
    print("Generating synthetic data...")

    generated_data_list = []

    for i, (condition_dict, count) in enumerate(zip(condition_dicts, counts)):
        if count == 0:
            continue

        print(f"  [{i+1}/{len(condition_dicts)}] {condition_dict} -> {count} samples")

        try:
            generated_decoded = generator.generate_and_decode(
                conditions=condition_dict,
                vae_model=vae_model,
                num_samples=count,
                guidance_scale=args.guidance_scale
            )
            generated_data_list.append(generated_decoded)

        except Exception as e:
            print(f"[WARN] Generation failed for condition {condition_dict}: {e}")
            # Create zeros as placeholder
            generated_data_list.append(np.zeros((count, n_genes)))

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
    args = parse_args()
    success = generate_synthetic_data(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
