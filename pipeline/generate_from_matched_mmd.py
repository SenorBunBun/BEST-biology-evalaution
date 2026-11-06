#!/usr/bin/env python3
"""
Generate Synthetic Datasets from MMD-Matched Checkpoints

This script:
1. Reads posthoc_mmd_history.json for each model
2. Finds the 3 checkpoints (one per model) with the closest MMD values to each other
3. Generates full synthetic datasets from those matched checkpoints
4. Saves them for fair downstream comparison (DEG, UMAP, etc.)

The goal is to compare models at similar distribution-matching quality levels.

Usage:
    python generate_from_matched_mmd.py --config dataset_configs/CRC_IMM_10X.json
    python generate_from_matched_mmd.py --config dataset_configs/CRC_IMM_10X.json --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scanpy as sc

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from config_loader import load_config, get_output_paths


def load_mmd_history(run_dir: Path, model_type: str) -> List[Dict]:
    """Load MMD history and return list of numeric checkpoints."""
    for filename in ["posthoc_mmd_history.json", "concurrent_mmd_history.json", "mmd_history.json"]:
        mmd_file = run_dir / filename
        if mmd_file.exists():
            with open(mmd_file) as f:
                data = json.load(f)
            break
    else:
        return []

    history = data.get("history", [])
    is_concurrent = (filename == "concurrent_mmd_history.json")

    # Filter to numeric epochs/steps and add metadata
    results = []
    for entry in history:
        epoch = entry.get("epoch")
        step = entry.get("step")
        mmd = entry.get("mmd")
        checkpoint = entry.get("checkpoint", "")

        # Concurrent format always uses epoch (even for GAN)
        if is_concurrent:
            if not isinstance(epoch, (int, float)):
                continue
        elif model_type == "gan":
            if not isinstance(step, (int, float)):
                continue
        else:
            if not isinstance(epoch, (int, float)):
                continue

        # Resolve checkpoint path
        checkpoint_path = run_dir / "checkpoints" / checkpoint
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / checkpoint

        results.append({
            "model": model_type,
            "epoch": epoch,
            "step": step,
            "mmd": mmd,
            "checkpoint": checkpoint,
            "path": checkpoint_path,
            "run_dir": run_dir
        })

    return results


def find_matched_checkpoints(
    cvae_history: List[Dict],
    diffusion_history: List[Dict],
    gan_history: List[Dict]
) -> Tuple[Dict, Dict, Dict, float]:
    """
    Find the triplet of checkpoints (one per model) with minimum MMD variance.

    Returns:
        (cvae_ckpt, diffusion_ckpt, gan_ckpt, mmd_spread)
    """
    best_triplet = None
    best_spread = float('inf')

    # Handle missing models
    histories = []
    if cvae_history:
        histories.append(('cvae', cvae_history))
    if diffusion_history:
        histories.append(('diffusion', diffusion_history))
    if gan_history:
        histories.append(('gan', gan_history))

    if len(histories) < 2:
        print("[WARN] Need at least 2 models with MMD history for matching")
        return None, None, None, float('inf')

    # Generate all combinations and find minimum spread
    if len(histories) == 3:
        for cvae_ckpt, diff_ckpt, gan_ckpt in product(cvae_history, diffusion_history, gan_history):
            mmds = [cvae_ckpt['mmd'], diff_ckpt['mmd'], gan_ckpt['mmd']]
            spread = max(mmds) - min(mmds)

            if spread < best_spread:
                best_spread = spread
                best_triplet = (cvae_ckpt, diff_ckpt, gan_ckpt)

        return best_triplet[0], best_triplet[1], best_triplet[2], best_spread

    elif len(histories) == 2:
        # Only 2 models available
        model1_name, model1_hist = histories[0]
        model2_name, model2_hist = histories[1]

        for ckpt1, ckpt2 in product(model1_hist, model2_hist):
            spread = abs(ckpt1['mmd'] - ckpt2['mmd'])

            if spread < best_spread:
                best_spread = spread
                best_triplet = {model1_name: ckpt1, model2_name: ckpt2}

        result = [None, None, None]
        for name, ckpt in best_triplet.items():
            if name == 'cvae':
                result[0] = ckpt
            elif name == 'diffusion':
                result[1] = ckpt
            elif name == 'gan':
                result[2] = ckpt

        return result[0], result[1], result[2], best_spread

    return None, None, None, float('inf')


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

    runs = sorted(model_dir.glob("run_*"), reverse=True)
    return runs[0] if runs else None


def generate_cvae_synthetic(ckpt: Dict, adata: sc.AnnData, condition_cols: List[str], output_path: Path) -> bool:
    """Generate synthetic data from CVAE checkpoint."""
    import torch
    sys.path.insert(0, str(Path(__file__).parent / "models" / "CVAE"))

    from unbatch_vanilla_cvae_relu_qol_categorical import Enhanced_Vanilla_CVAE_Categorical

    run_dir = ckpt['run_dir']
    checkpoint_path = ckpt['path']

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
        cvae = Enhanced_Vanilla_CVAE_Categorical(
            n_genes=n_genes,
            n_labels=len(condition_cols),
            latent_size=latent_size,
            hidden_dims=hidden_dims,
            device=device,
            condition_cols=condition_cols,
            verbose=False
        )

        if encoders_path.exists():
            cvae.load_label_encoders(str(encoders_path))
        else:
            cvae.fit_label_encoders(adata, condition_cols)

        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cvae.load_state_dict(state_dict)
        cvae.to(device)
        cvae.eval()

        condition_counts = adata.obs.groupby(condition_cols, observed=True).size()

        generated_list = []
        obs_records = []

        for conditions, count in condition_counts.items():
            if not isinstance(conditions, tuple):
                conditions = (conditions,)

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
            for _ in range(count):
                obs_records.append({col: val for col, val in zip(condition_cols, conditions)})

        import pandas as pd
        X = np.vstack(generated_list)
        obs = pd.DataFrame(obs_records)
        var = adata.var.copy()

        adata_synthetic = sc.AnnData(X=X, obs=obs, var=var)
        adata_synthetic.write_h5ad(output_path)

        return True

    except Exception as e:
        print(f"[ERROR] CVAE generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_diffusion_synthetic(ckpt: Dict, adata: sc.AnnData, condition_cols: List[str], output_path: Path) -> bool:
    """Generate synthetic data from Diffusion checkpoint."""
    import torch
    sys.path.insert(0, str(Path(__file__).parent / "models" / "Diffusion"))

    from diffusion_model_cfg import ClassifierFreeDiffusionModel
    from vae_diffusion import DiffusionIntegrationVAE
    from diffusion_generation_utils import ConditionProcessor, DiffusionModelGenerator

    run_dir = ckpt['run_dir']
    checkpoint_path = ckpt['path']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(run_dir / "vae_config.json") as f:
        vae_config = json.load(f)
    with open(run_dir / "condition_config.json") as f:
        condition_info = json.load(f)
    with open(run_dir / "training_metrics.json") as f:
        metrics = json.load(f)

    full_condition_config = condition_info.get('full_condition_config', {})
    model_params = metrics.get("model_params", {})

    try:
        vae_model = DiffusionIntegrationVAE(
            n_genes=vae_config['n_genes'],
            latent_size=vae_config['latent_size'],
            hidden_dims=vae_config['hidden_dims'],
            verbose=False, auto_load=False
        )
        vae_model.load_state_dict(torch.load(run_dir / "vae_model.pt", map_location=device, weights_only=False))
        vae_model.to(device)
        vae_model.eval()

        processor = ConditionProcessor(full_condition_config)
        processor.fit_conditions(adata)

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

        generator = DiffusionModelGenerator(
            model, processor, str(device),
            num_timesteps=model_params.get("num_timesteps", 1000)
        )

        condition_counts = adata.obs.groupby(condition_cols, observed=True).size()

        generated_list = []
        obs_records = []

        for conditions, count in condition_counts.items():
            if not isinstance(conditions, tuple):
                conditions = (conditions,)

            cond_dict = {col: val for col, val in zip(condition_cols, conditions)}
            latent_samples = generator.generate_specific_conditions(cond_dict, count, num_inference_steps=50)

            batch_size = 100
            decoded_samples = []
            for i in range(0, len(latent_samples), batch_size):
                batch = torch.tensor(latent_samples[i:i + batch_size], dtype=torch.float32, device=device)
                with torch.no_grad():
                    decoded = vae_model.decode(batch)
                decoded_samples.append(decoded.cpu())

            decoded = torch.cat(decoded_samples, dim=0)
            decoded = torch.clamp(decoded, min=0).numpy()
            generated_list.append(decoded)

            for _ in range(count):
                obs_records.append(cond_dict.copy())

        import pandas as pd
        X = np.vstack(generated_list)
        obs = pd.DataFrame(obs_records)
        var = adata.var.copy()

        adata_synthetic = sc.AnnData(X=X, obs=obs, var=var)
        adata_synthetic.write_h5ad(output_path)

        return True

    except Exception as e:
        print(f"[ERROR] Diffusion generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_gan_synthetic(ckpt: Dict, adata: sc.AnnData, condition_cols: List[str], output_path: Path) -> bool:
    """Generate synthetic data from GAN checkpoint."""
    import tensorflow as tf
    import importlib.util

    run_dir = ckpt['run_dir']
    checkpoint_path = ckpt['path']
    gan_dir = Path(__file__).parent / "models" / "GAN"

    spec = importlib.util.spec_from_file_location("gan_utils", gan_dir / "utils" / "condition_processor_generic.py")
    gan_utils_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gan_utils_module)
    ConditionProcessorGeneric = gan_utils_module.ConditionProcessorGeneric

    spec_gen = importlib.util.spec_from_file_location("generator", gan_dir / "models" / "generator.py")
    gen_module = importlib.util.module_from_spec(spec_gen)
    spec_gen.loader.exec_module(gen_module)
    build_generator = gen_module.build_generator

    with open(run_dir / "training_metrics.json") as f:
        metrics = json.load(f)

    model_params = metrics.get("model_params", {})
    n_genes = model_params.get("n_genes", adata.n_vars)
    latent_dim = model_params.get("latent_dim", 128)
    gen_layers = model_params.get("gen_layers", [256, 512, 1024])

    cond_config_path = run_dir / "condition_config.json"
    condition_processor = ConditionProcessorGeneric(condition_cols)
    if cond_config_path.exists():
        condition_processor.load_config(str(cond_config_path))
    else:
        condition_processor.fit(adata)

    try:
        generator = build_generator(
            latent_dim=latent_dim,
            condition_dim=condition_processor.total_condition_dim,
            output_dim=n_genes,
            hidden_layers=gen_layers
        )

        gen_weights = checkpoint_path / "generator.weights.h5"
        generator.load_weights(str(gen_weights))

        conditions = condition_processor.encode_conditions(adata.obs)
        noise = tf.random.normal((len(adata), latent_dim))
        generated = generator((noise, conditions), training=False)
        generated = tf.maximum(generated, 0).numpy()

        import pandas as pd
        obs = adata.obs[condition_cols].copy().reset_index(drop=True)
        var = adata.var.copy()

        adata_synthetic = sc.AnnData(X=generated, obs=obs, var=var)
        adata_synthetic.write_h5ad(output_path)

        return True

    except Exception as e:
        print(f"[ERROR] GAN generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate from MMD-matched checkpoints')
    parser.add_argument('--config', required=True, help='Path to dataset config JSON')
    parser.add_argument('--dry-run', action='store_true', help='Only show matched checkpoints')

    args = parser.parse_args()

    config = load_config(args.config)
    output_paths = get_output_paths(config)
    condition_cols = config["conditions"]["columns"]

    print("\n" + "=" * 60)
    print("FIND MMD-MATCHED CHECKPOINTS")
    print("=" * 60)
    print(f"Dataset: {config['dataset_name']}")

    # Load MMD histories for all models
    histories = {}
    for model_type in ['cvae', 'diffusion', 'gan']:
        model_dir = output_paths["trained_models"][model_type]
        if model_dir.exists():
            run_dir = find_latest_run(model_dir)
            if run_dir:
                hist = load_mmd_history(run_dir, model_type)
                if hist:
                    histories[model_type] = hist
                    print(f"[OK] {model_type.upper()}: {len(hist)} checkpoints")
                else:
                    print(f"[SKIP] {model_type.upper()}: No MMD history")
            else:
                print(f"[SKIP] {model_type.upper()}: No run directory")
        else:
            print(f"[SKIP] {model_type.upper()}: Not trained")

    if len(histories) < 2:
        print("\n[ERROR] Need at least 2 models with MMD history")
        return

    # Find matched checkpoints
    cvae_ckpt, diff_ckpt, gan_ckpt, spread = find_matched_checkpoints(
        histories.get('cvae', []),
        histories.get('diffusion', []),
        histories.get('gan', [])
    )

    print("\n" + "=" * 60)
    print("MMD-MATCHED CHECKPOINTS")
    print("=" * 60)
    print(f"MMD Spread: {spread:.6f} (lower = better match)")
    print()

    matched = {}
    if cvae_ckpt:
        print(f"CVAE:      Epoch {cvae_ckpt['epoch']}, MMD = {cvae_ckpt['mmd']:.6f}")
        print(f"           Checkpoint: {cvae_ckpt['checkpoint']}")
        matched['cvae'] = cvae_ckpt
    if diff_ckpt:
        print(f"Diffusion: Epoch {diff_ckpt['epoch']}, MMD = {diff_ckpt['mmd']:.6f}")
        print(f"           Checkpoint: {diff_ckpt['checkpoint']}")
        matched['diffusion'] = diff_ckpt
    if gan_ckpt:
        if gan_ckpt.get('step') is not None:
            print(f"GAN:       Step {gan_ckpt['step']} (Epoch {gan_ckpt['epoch']}), MMD = {gan_ckpt['mmd']:.6f}")
        else:
            print(f"GAN:       Epoch {gan_ckpt['epoch']}, MMD = {gan_ckpt['mmd']:.6f}")
        print(f"           Checkpoint: {gan_ckpt['checkpoint']}")
        matched['gan'] = gan_ckpt

    if args.dry_run:
        print("\n[DRY-RUN] Would generate from these checkpoints")
        return

    # Load real data
    print("\n" + "=" * 60)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 60)

    adata = sc.read_h5ad(config["paths"]["processed_data"])
    print(f"Real data: {adata.n_obs} cells x {adata.n_vars} genes")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_paths["synthetic_data"].mkdir(parents=True, exist_ok=True)

    results = {}

    for model_type, ckpt in matched.items():
        print(f"\n{model_type.upper()}:")
        output_file = output_paths["synthetic_data"] / f"{model_type}_matched_{config['dataset_name']}_{timestamp}.h5ad"

        if model_type == "cvae":
            success = generate_cvae_synthetic(ckpt, adata, condition_cols, output_file)
        elif model_type == "diffusion":
            success = generate_diffusion_synthetic(ckpt, adata, condition_cols, output_file)
        elif model_type == "gan":
            success = generate_gan_synthetic(ckpt, adata, condition_cols, output_file)

        results[model_type] = success
        if success:
            print(f"  [OK] Saved to {output_file}")

    # Save matched info
    matched_info = {
        "timestamp": timestamp,
        "mmd_spread": spread,
        "checkpoints": {
            model: {
                "epoch": ckpt.get("epoch"),
                "step": ckpt.get("step"),
                "mmd": ckpt["mmd"],
                "checkpoint": ckpt["checkpoint"]
            }
            for model, ckpt in matched.items()
        }
    }
    info_path = output_paths["synthetic_data"] / f"matched_mmd_info_{timestamp}.json"
    with open(info_path, 'w') as f:
        json.dump(matched_info, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model, success in results.items():
        print(f"  {model.upper()}: {'OK' if success else 'FAILED'}")

    print(f"\nMatched info saved to: {info_path}")
    print(f"Synthetic data in: {output_paths['synthetic_data']}")
    print("\nRun analysis:")
    print(f"  python driver.py --config {args.config} --stage analyze")


if __name__ == "__main__":
    main()
