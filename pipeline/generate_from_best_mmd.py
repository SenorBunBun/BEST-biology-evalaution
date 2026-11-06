#!/usr/bin/env python3
"""
Generate Synthetic Datasets from Best MMD Checkpoints

This script:
1. Reads MMD history for each model (checks posthoc_mmd_history.json,
   concurrent_mmd_history.json, then mmd_history.json)
2. Finds the checkpoint with minimum MMD (closest to real data distribution)
3. Generates full synthetic datasets matching the real data's condition distribution
4. Saves them to synthetic_data/ for downstream analysis (DEG, UMAP, etc.)

Usage:
    python generate_from_best_mmd.py --config dataset_configs/CRC_IMM_10X.json
    python generate_from_best_mmd.py --config dataset_configs/CRC_IMM_10X.json --model cvae
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scanpy as sc

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from config_loader import load_config, get_output_paths


def find_best_mmd_checkpoint(run_dir: Path, model_type: str) -> Optional[Dict]:
    """
    Find the checkpoint with minimum MMD from posthoc_mmd_history.json.

    Returns:
        Dict with 'checkpoint', 'epoch'/'step', 'mmd', and 'path'
    """
    # Try posthoc first, then concurrent (training-time with early stopping), then legacy
    for filename in ["posthoc_mmd_history.json", "concurrent_mmd_history.json", "mmd_history.json"]:
        mmd_file = run_dir / filename
        if mmd_file.exists():
            with open(mmd_file) as f:
                data = json.load(f)
            break
    else:
        print(f"[WARN] No MMD history found in {run_dir}")
        return None

    history = data.get("history", [])
    if not history:
        print(f"[WARN] Empty MMD history in {run_dir}")
        return None

    # Concurrent format has best_epoch directly — use it for fast lookup
    best_epoch = data.get("best_epoch")
    if best_epoch is not None:
        best_entries = [e for e in history if e.get("epoch") == best_epoch]
        if best_entries:
            best = best_entries[0]
            print(f"[INFO] Using best_epoch={best_epoch} from {filename}")
        else:
            best = min(history, key=lambda x: x["mmd"])
    else:
        # Filter to numeric epochs/steps only (exclude 'best', 'final', etc.)
        numeric_entries = []
        for entry in history:
            epoch_or_step = entry.get("epoch") if model_type != "gan" else entry.get("step")
            if isinstance(epoch_or_step, (int, float)):
                numeric_entries.append(entry)

        if not numeric_entries:
            print(f"[WARN] No numeric checkpoints in MMD history")
            return None

        # Find minimum MMD
        best = min(numeric_entries, key=lambda x: x["mmd"])

    # Resolve checkpoint path
    checkpoint_name = best.get("checkpoint", "")
    if model_type == "cvae":
        checkpoint_path = run_dir / "checkpoints" / checkpoint_name
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / checkpoint_name
    elif model_type == "diffusion":
        checkpoint_path = run_dir / "checkpoints" / checkpoint_name
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / checkpoint_name
    elif model_type == "gan":
        checkpoint_path = run_dir / "checkpoints" / checkpoint_name
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / checkpoint_name

    return {
        "checkpoint": checkpoint_name,
        "epoch": best.get("epoch"),
        "step": best.get("step"),
        "mmd": best["mmd"],
        "path": checkpoint_path,
        "sigma": data.get("sigma")
    }


def find_latest_run(model_dir: Path) -> Optional[Path]:
    """Find the latest run directory for a model."""
    latest_json = model_dir / "latest_model.json"
    if latest_json.exists():
        with open(latest_json) as f:
            data = json.load(f)
            run_dir_name = data.get("run_dir", "")
            if run_dir_name:
                run_path = model_dir / run_dir_name
                if run_path.exists():
                    return run_path

    # Fallback: find most recent run_* directory
    runs = sorted(model_dir.glob("run_*"), reverse=True)
    return runs[0] if runs else None


def generate_cvae_synthetic(
    checkpoint_path: Path,
    run_dir: Path,
    adata: sc.AnnData,
    condition_cols: List[str],
    output_path: Path
) -> bool:
    """Generate synthetic data from CVAE checkpoint."""
    import torch
    sys.path.insert(0, str(Path(__file__).parent / "models" / "CVAE"))

    from unbatch_vanilla_cvae_relu_qol_categorical import Enhanced_Vanilla_CVAE_Categorical

    # Load model config
    metrics_path = run_dir / "training_metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    model_params = metrics.get("model_params", {})
    n_genes = model_params.get("n_genes", adata.n_vars)
    latent_size = model_params.get("latent_size", 32)
    hidden_dims = model_params.get("hidden_dims", [800, 400])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoders_path = run_dir / "label_encoders.pkl"

    try:
        # Create model
        cvae = Enhanced_Vanilla_CVAE_Categorical(
            n_genes=n_genes,
            n_labels=len(condition_cols),
            latent_size=latent_size,
            hidden_dims=hidden_dims,
            device=device,
            condition_cols=condition_cols,
            verbose=False
        )

        # Load encoders and weights
        if encoders_path.exists():
            cvae.load_label_encoders(str(encoders_path))
        else:
            cvae.fit_label_encoders(adata, condition_cols)

        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cvae.load_state_dict(state_dict)
        cvae.to(device)
        cvae.eval()

        # Generate matching real data's condition distribution
        condition_counts = adata.obs.groupby(condition_cols, observed=True).size()

        generated_list = []
        obs_records = []

        for conditions, count in condition_counts.items():
            if not isinstance(conditions, tuple):
                conditions = (conditions,)

            # Encode conditions
            encoded = []
            for col, val in zip(condition_cols, conditions):
                if cvae.label_encoders.get(col) is not None:
                    enc_val = cvae.label_encoders[col].transform([str(val)])[0]
                else:
                    enc_val = float(val)
                encoded.append(enc_val)

            cond_tensor = torch.tensor([encoded], dtype=torch.float32, device=device)

            with torch.no_grad():
                synthetic = cvae.generate(cond_tensor, count)
                synthetic = torch.clamp(synthetic, min=0).cpu().numpy()

            generated_list.append(synthetic)

            # Create obs records
            for _ in range(count):
                record = {col: val for col, val in zip(condition_cols, conditions)}
                obs_records.append(record)

        # Create AnnData
        import pandas as pd
        X = np.vstack(generated_list)
        obs = pd.DataFrame(obs_records)
        var = adata.var.copy()

        adata_synthetic = sc.AnnData(X=X, obs=obs, var=var)
        adata_synthetic.write_h5ad(output_path)

        print(f"[OK] Generated {adata_synthetic.n_obs} cells, saved to {output_path}")
        return True

    except Exception as e:
        print(f"[ERROR] CVAE generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_diffusion_synthetic(
    checkpoint_path: Path,
    run_dir: Path,
    adata: sc.AnnData,
    condition_cols: List[str],
    output_path: Path
) -> bool:
    """Generate synthetic data from Diffusion checkpoint."""
    import torch
    sys.path.insert(0, str(Path(__file__).parent / "models" / "Diffusion"))

    from diffusion_model_cfg import ClassifierFreeDiffusionModel
    from vae_diffusion import DiffusionIntegrationVAE
    from diffusion_generation_utils import ConditionProcessor, DiffusionModelGenerator

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configs
    vae_config_path = run_dir / "vae_config.json"
    vae_model_path = run_dir / "vae_model.pt"
    condition_config_path = run_dir / "condition_config.json"
    metrics_path = run_dir / "training_metrics.json"

    with open(vae_config_path) as f:
        vae_config = json.load(f)
    with open(condition_config_path) as f:
        condition_info = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)

    full_condition_config = condition_info.get('full_condition_config', {})
    model_params = metrics.get("model_params", {})

    try:
        # Load VAE
        vae_model = DiffusionIntegrationVAE(
            n_genes=vae_config['n_genes'],
            latent_size=vae_config['latent_size'],
            hidden_dims=vae_config['hidden_dims'],
            verbose=False,
            auto_load=False
        )
        vae_model.load_state_dict(torch.load(vae_model_path, map_location=device, weights_only=False))
        vae_model.to(device)
        vae_model.eval()

        # Create condition processor
        processor = ConditionProcessor(full_condition_config)
        processor.fit_conditions(adata)

        # Load diffusion model
        model = ClassifierFreeDiffusionModel(
            latent_dim=model_params.get("latent_size", 128),
            condition_config=full_condition_config,
            hidden_dims=model_params.get("hidden_dims", [512, 512, 256, 128]),
            classifier_free_prob=model_params.get("classifier_free_prob", 0.1),
            dropout=model_params.get("dropout", 0.1)
        )

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        # Create generator
        generator = DiffusionModelGenerator(
            model, processor, str(device),
            num_timesteps=model_params.get("num_timesteps", 1000)
        )

        # Generate matching condition distribution
        condition_counts = adata.obs.groupby(condition_cols, observed=True).size()

        generated_list = []
        obs_records = []

        for conditions, count in condition_counts.items():
            if not isinstance(conditions, tuple):
                conditions = (conditions,)

            cond_dict = {col: val for col, val in zip(condition_cols, conditions)}

            # Generate latent samples
            latent_samples = generator.generate_specific_conditions(
                cond_dict, count, num_inference_steps=50
            )

            # Decode to gene expression
            batch_size = 100
            decoded_samples = []
            for i in range(0, len(latent_samples), batch_size):
                batch = torch.tensor(
                    latent_samples[i:i + batch_size],
                    dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    decoded = vae_model.decode(batch)
                decoded_samples.append(decoded.cpu())

            decoded = torch.cat(decoded_samples, dim=0)
            decoded = torch.clamp(decoded, min=0).numpy()
            generated_list.append(decoded)

            for _ in range(count):
                obs_records.append(cond_dict.copy())

        # Create AnnData
        import pandas as pd
        X = np.vstack(generated_list)
        obs = pd.DataFrame(obs_records)
        var = adata.var.copy()

        adata_synthetic = sc.AnnData(X=X, obs=obs, var=var)
        adata_synthetic.write_h5ad(output_path)

        print(f"[OK] Generated {adata_synthetic.n_obs} cells, saved to {output_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Diffusion generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_gan_synthetic(
    checkpoint_path: Path,
    run_dir: Path,
    adata: sc.AnnData,
    condition_cols: List[str],
    output_path: Path
) -> bool:
    """Generate synthetic data from GAN checkpoint."""
    import tensorflow as tf
    import importlib.util

    gan_dir = Path(__file__).parent / "models" / "GAN"

    # Load condition processor
    spec = importlib.util.spec_from_file_location(
        "gan_utils",
        gan_dir / "utils" / "condition_processor_generic.py"
    )
    gan_utils_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gan_utils_module)
    ConditionProcessorGeneric = gan_utils_module.ConditionProcessorGeneric

    # Load generator
    spec_gen = importlib.util.spec_from_file_location(
        "generator",
        gan_dir / "models" / "generator.py"
    )
    gen_module = importlib.util.module_from_spec(spec_gen)
    spec_gen.loader.exec_module(gen_module)
    build_generator = gen_module.build_generator

    # Load config
    metrics_path = run_dir / "training_metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    model_params = metrics.get("model_params", {})
    n_genes = model_params.get("n_genes", adata.n_vars)
    latent_dim = model_params.get("latent_dim", 128)
    gen_layers = model_params.get("gen_layers", [256, 512, 1024])

    # Load condition processor
    cond_config_path = run_dir / "condition_config.json"
    condition_processor = ConditionProcessorGeneric(condition_cols)
    if cond_config_path.exists():
        condition_processor.load_config(str(cond_config_path))
    else:
        condition_processor.fit(adata)

    try:
        # Build generator
        generator = build_generator(
            latent_dim=latent_dim,
            condition_dim=condition_processor.total_condition_dim,
            output_dim=n_genes,
            hidden_layers=gen_layers
        )

        # Load weights
        gen_weights = checkpoint_path / "generator.weights.h5"
        generator.load_weights(str(gen_weights))

        # Generate matching condition distribution
        conditions = condition_processor.encode_conditions(adata.obs)
        noise = tf.random.normal((len(adata), latent_dim))
        generated = generator((noise, conditions), training=False)
        generated = tf.maximum(generated, 0).numpy()

        # Create AnnData
        import pandas as pd
        obs = adata.obs[condition_cols].copy().reset_index(drop=True)
        var = adata.var.copy()

        adata_synthetic = sc.AnnData(X=generated, obs=obs, var=var)
        adata_synthetic.write_h5ad(output_path)

        print(f"[OK] Generated {adata_synthetic.n_obs} cells, saved to {output_path}")
        return True

    except Exception as e:
        print(f"[ERROR] GAN generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic data from best MMD checkpoints'
    )
    parser.add_argument('--config', required=True, help='Path to dataset config JSON')
    parser.add_argument('--model', choices=['cvae', 'diffusion', 'gan', 'all'],
                        default='all', help='Model to generate from (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show which checkpoints would be used')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    output_paths = get_output_paths(config)
    condition_cols = config["conditions"]["columns"]

    print("\n" + "=" * 60)
    print("GENERATE FROM BEST MMD CHECKPOINTS")
    print("=" * 60)
    print(f"Dataset: {config['dataset_name']}")
    print(f"Conditions: {condition_cols}")

    # Load real data
    adata = sc.read_h5ad(config["paths"]["processed_data"])
    print(f"Real data: {adata.n_obs} cells x {adata.n_vars} genes")

    models = ['cvae', 'diffusion', 'gan'] if args.model == 'all' else [args.model]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {}
    best_info = {}

    for model_type in models:
        print(f"\n{'='*60}")
        print(f"{model_type.upper()}")
        print("=" * 60)

        model_dir = output_paths["trained_models"][model_type]
        if not model_dir.exists():
            print(f"[SKIP] No trained model found")
            continue

        run_dir = find_latest_run(model_dir)
        if not run_dir:
            print(f"[SKIP] No run directory found")
            continue

        # Find best MMD checkpoint
        best = find_best_mmd_checkpoint(run_dir, model_type)
        if not best:
            print(f"[SKIP] No MMD history found")
            continue

        print(f"Best checkpoint: {best['checkpoint']}")
        if model_type == 'gan':
            print(f"  Step: {best.get('step')}, Epoch: {best.get('epoch')}")
        else:
            print(f"  Epoch: {best.get('epoch')}")
        print(f"  MMD: {best['mmd']:.6f}")
        print(f"  Path: {best['path']}")

        if args.dry_run:
            print(f"[DRY-RUN] Would generate from this checkpoint")
            results[model_type] = True
            continue

        # Generate synthetic data
        output_file = output_paths["synthetic_data"] / f"{model_type}_bestmmd_{config['dataset_name']}_{timestamp}.h5ad"
        output_paths["synthetic_data"].mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating synthetic data...")

        if model_type == "cvae":
            success = generate_cvae_synthetic(
                best['path'], run_dir, adata, condition_cols, output_file
            )
        elif model_type == "diffusion":
            success = generate_diffusion_synthetic(
                best['path'], run_dir, adata, condition_cols, output_file
            )
        elif model_type == "gan":
            success = generate_gan_synthetic(
                best['path'], run_dir, adata, condition_cols, output_file
            )

        results[model_type] = success
        if success:
            best_info[model_type] = {
                "epoch": best.get("epoch"),
                "step": best.get("step"),
                "mmd": best["mmd"],
                "checkpoint": best["checkpoint"],
                "sigma": best.get("sigma")
            }

    # Save best MMD info
    if best_info:
        info = {
            "timestamp": timestamp,
            "checkpoints": best_info
        }
        info_path = output_paths["synthetic_data"] / f"best_mmd_info_{timestamp}.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {model.upper()}: [{status}]")

    if best_info:
        print(f"\nBest MMD info saved to: {info_path}")
    print(f"Synthetic data saved to: {output_paths['synthetic_data']}")
    print("Run analysis with:")
    print(f"  python driver.py --config {args.config} --stage analyze")


if __name__ == "__main__":
    main()
