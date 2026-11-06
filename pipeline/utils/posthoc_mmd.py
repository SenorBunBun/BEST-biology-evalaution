#!/usr/bin/env python3
"""
Post-hoc MMD Calculation for All Saved Checkpoints

This module provides functionality to calculate MMD (Maximum Mean Discrepancy)
for all saved checkpoints after training is complete. It uses the exact same
MMD evaluation logic as the training scripts.

Two-phase approach for memory efficiency:
  Phase 1: Generate synthetic data from each checkpoint, save to disk
  Phase 2: Load saved data and compute MMD scores

Usage:
    Called from driver.py with --stage posthoc-mmd
"""

import gc
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scanpy as sc


def discover_checkpoints(run_dir: Path, model_type: str) -> List[Dict]:
    """
    Find all checkpoint files for a model.

    Args:
        run_dir: Path to the model's run directory
        model_type: One of 'cvae', 'diffusion', 'gan'

    Returns:
        List of checkpoint info dicts with 'path', 'epoch' (or 'step' for GAN)
    """
    checkpoints = []
    checkpoint_dir = run_dir / "checkpoints"

    if not checkpoint_dir.exists():
        print(f"[WARN] No checkpoints directory found: {checkpoint_dir}")
        return checkpoints

    if model_type == "cvae":
        # CVAE: checkpoints/checkpoint_epoch_*.pt
        for ckpt_file in checkpoint_dir.glob("checkpoint_epoch_*.pt"):
            match = re.search(r'checkpoint_epoch_(\d+)\.pt', ckpt_file.name)
            if match:
                epoch = int(match.group(1))
                checkpoints.append({
                    'path': ckpt_file,
                    'epoch': epoch,
                    'checkpoint': ckpt_file.name
                })

    elif model_type == "diffusion":
        # Diffusion: best_model files have model state only
        # checkpoints/best_model_epoch_*_model.pth
        for ckpt_file in checkpoint_dir.glob("best_model_epoch_*_model.pth"):
            match = re.search(r'best_model_epoch_(\d+)_model\.pth', ckpt_file.name)
            if match:
                epoch = int(match.group(1))
                checkpoints.append({
                    'path': ckpt_file,
                    'epoch': epoch,
                    'checkpoint': ckpt_file.name
                })

        # Also check for regular checkpoint files (checkpoint_epoch_*.pth)
        for ckpt_file in checkpoint_dir.glob("checkpoint_epoch_*.pth"):
            match = re.search(r'checkpoint_epoch_(\d+)\.pth', ckpt_file.name)
            if match:
                epoch = int(match.group(1))
                # Check if we already have this epoch from best_model
                existing_epochs = [c['epoch'] for c in checkpoints]
                if epoch not in existing_epochs:
                    checkpoints.append({
                        'path': ckpt_file,
                        'epoch': epoch,
                        'checkpoint': ckpt_file.name
                    })

    elif model_type == "gan":
        # GAN: checkpoints/checkpoint_step_*/generator.weights.h5
        for ckpt_dir in checkpoint_dir.glob("checkpoint_step_*"):
            if ckpt_dir.is_dir():
                gen_weights = ckpt_dir / "generator.weights.h5"
                if gen_weights.exists():
                    match = re.search(r'checkpoint_step_(\d+)', ckpt_dir.name)
                    if match:
                        step = int(match.group(1))
                        checkpoints.append({
                            'path': ckpt_dir,
                            'step': step,
                            'checkpoint': ckpt_dir.name
                        })

    # Sort by epoch/step
    sort_key = 'epoch' if model_type != 'gan' else 'step'
    checkpoints.sort(key=lambda x: x[sort_key])

    return checkpoints


def cleanup_memory(model_type: str, model=None):
    """
    Clean up memory after model evaluation.

    Args:
        model_type: 'cvae', 'diffusion', or 'gan'
        model: The model object to delete (optional)
    """
    if model is not None:
        del model

    if model_type in ['cvae', 'diffusion']:
        # PyTorch cleanup
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif model_type == 'gan':
        # TensorFlow cleanup
        import tensorflow as tf
        tf.keras.backend.clear_session()

    gc.collect()


# =============================================================================
# PHASE 1: GENERATE SYNTHETIC DATA FROM CHECKPOINTS
# =============================================================================

def generate_from_cvae_checkpoint(
    checkpoint_path: Path,
    run_dir: Path,
    val_adata,
    condition_cols: List[str],
    output_path: Path,
    device
) -> bool:
    """Generate synthetic data from a single CVAE checkpoint and save to disk."""
    import torch
    sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "CVAE"))

    from unbatch_vanilla_cvae_relu_qol_categorical import Enhanced_Vanilla_CVAE_Categorical

    # Load model config from run
    metrics_path = run_dir / "training_metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    model_params = metrics.get("model_params", {})
    n_genes = model_params.get("n_genes", val_adata.n_vars)
    latent_size = model_params.get("latent_size", 32)
    hidden_dims = model_params.get("hidden_dims", [800, 400])
    beta = model_params.get("beta", 0.05)
    dropout = model_params.get("dropout", 0.03)

    encoders_path = run_dir / "label_encoders.pkl"

    try:
        # Create model with exact architecture
        cvae = Enhanced_Vanilla_CVAE_Categorical(
            n_genes=n_genes,
            n_labels=len(condition_cols),
            latent_size=latent_size,
            hidden_dims=hidden_dims,
            beta=beta,
            dropout=dropout,
            device=device,
            condition_cols=condition_cols,
            verbose=False
        )

        # Load label encoders
        if encoders_path.exists():
            cvae.load_label_encoders(str(encoders_path))
        else:
            cvae.fit_label_encoders(val_adata, condition_cols)

        # Load checkpoint weights
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cvae.load_state_dict(state_dict)
        cvae.to(device)
        cvae.eval()

        # Generate synthetic data matching validation conditions
        condition_counts = val_adata.obs.groupby(condition_cols, observed=True).size()

        generated_list = []
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

        generated_data = np.vstack(generated_list)

        # Save to disk
        np.save(output_path, generated_data)

        # Clean up
        del cvae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return True

    except Exception as e:
        print(f"[ERROR] Failed to generate from {checkpoint_path.name}: {e}")
        import traceback
        traceback.print_exc()
        cleanup_memory('cvae')
        return False


def generate_from_diffusion_checkpoint(
    checkpoint_path: Path,
    run_dir: Path,
    val_adata,
    condition_cols: List[str],
    output_path: Path,
    device
) -> bool:
    """Generate synthetic data from a single Diffusion checkpoint and save to disk."""
    import torch
    sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "Diffusion"))

    from diffusion_model_cfg import ClassifierFreeDiffusionModel
    from vae_diffusion import DiffusionIntegrationVAE
    from diffusion_generation_utils import ConditionProcessor, DiffusionModelGenerator

    # Load VAE config and model
    vae_config_path = run_dir / "vae_config.json"
    vae_model_path = run_dir / "vae_model.pt"

    if not vae_config_path.exists() or not vae_model_path.exists():
        print(f"[ERROR] VAE files not found in {run_dir}")
        return False

    with open(vae_config_path) as f:
        vae_config = json.load(f)

    # Load condition config
    condition_config_path = run_dir / "condition_config.json"
    with open(condition_config_path) as f:
        condition_info = json.load(f)

    full_condition_config = condition_info.get('full_condition_config', {})

    # Load training metrics
    metrics_path = run_dir / "training_metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    model_params = metrics.get("model_params", {})
    latent_size = model_params.get("latent_size", 128)
    hidden_dims = model_params.get("hidden_dims", [512, 512, 256, 128])
    classifier_free_prob = model_params.get("classifier_free_prob", 0.1)
    dropout = model_params.get("dropout", 0.1)
    num_timesteps = model_params.get("num_timesteps", 1000)

    try:
        # Load VAE
        vae_model = DiffusionIntegrationVAE(
            n_genes=vae_config['n_genes'],
            latent_size=vae_config['latent_size'],
            hidden_dims=vae_config['hidden_dims'],
            beta=vae_config.get('beta', 0.001),
            dropout=vae_config.get('dropout', 0.1),
            verbose=False,
            auto_load=False
        )
        vae_model.load_state_dict(torch.load(vae_model_path, map_location=device, weights_only=False))
        vae_model.to(device)
        vae_model.eval()

        # Create condition processor
        processor = ConditionProcessor(full_condition_config)
        processor.fit_conditions(val_adata)

        # Create diffusion model
        model = ClassifierFreeDiffusionModel(
            latent_dim=latent_size,
            condition_config=full_condition_config,
            hidden_dims=hidden_dims,
            classifier_free_prob=classifier_free_prob,
            dropout=dropout
        )

        # Load checkpoint weights
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        # Create generator
        generator = DiffusionModelGenerator(
            model, processor, str(device), num_timesteps=num_timesteps
        )

        # Collect conditions from validation data
        condition_list = []
        for idx in range(len(val_adata)):
            sample_conds = {}
            for col in condition_cols:
                sample_conds[col] = val_adata.obs[col].iloc[idx]
            condition_list.append(sample_conds)

        # Group by unique conditions and generate
        from collections import Counter
        cond_tuples = [tuple(sorted(c.items())) for c in condition_list]
        cond_counts = Counter(cond_tuples)

        generated_latent_list = []
        for cond_tuple, count in cond_counts.items():
            cond_dict = dict(cond_tuple)
            latent_samples = generator.generate_specific_conditions(
                cond_dict, count, num_inference_steps=50
            )
            generated_latent_list.append(latent_samples)

        generated_latent = np.vstack(generated_latent_list)

        # Decode latent to gene expression using VAE
        batch_size = 100
        decoded_samples = []

        for i in range(0, len(generated_latent), batch_size):
            batch = torch.tensor(
                generated_latent[i:i + batch_size],
                dtype=torch.float32,
                device=device
            )
            with torch.no_grad():
                decoded_batch = vae_model.decode(batch)
            decoded_samples.append(decoded_batch.cpu())

        generated_gene_expr = torch.cat(decoded_samples, dim=0)
        generated_gene_expr = torch.clamp(generated_gene_expr, min=0).numpy()

        # Save to disk
        np.save(output_path, generated_gene_expr)

        # Clean up
        del model, vae_model, generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return True

    except Exception as e:
        print(f"[ERROR] Failed to generate from {checkpoint_path.name}: {e}")
        import traceback
        traceback.print_exc()
        cleanup_memory('diffusion')
        return False


def generate_from_gan_checkpoint(
    checkpoint_path: Path,
    run_dir: Path,
    val_adata,
    condition_cols: List[str],
    output_path: Path,
    training_metrics: Dict
) -> bool:
    """Generate synthetic data from a single GAN checkpoint and save to disk."""
    import tensorflow as tf
    import importlib.util

    gan_dir = Path(__file__).parent.parent / "models" / "GAN"
    sys.path.insert(0, str(gan_dir))

    # Load GAN's condition processor
    spec = importlib.util.spec_from_file_location(
        "gan_utils",
        gan_dir / "utils" / "condition_processor_generic.py"
    )
    gan_utils_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gan_utils_module)
    ConditionProcessorGeneric = gan_utils_module.ConditionProcessorGeneric

    # Load GAN generator
    spec_gen = importlib.util.spec_from_file_location(
        "generator",
        gan_dir / "models" / "generator.py"
    )
    gen_module = importlib.util.module_from_spec(spec_gen)
    spec_gen.loader.exec_module(gen_module)
    build_generator = gen_module.build_generator

    # Load model params
    model_params = training_metrics.get("model_params", {})
    n_genes = model_params.get("n_genes", val_adata.n_vars)
    latent_dim = model_params.get("latent_dim", 128)
    gen_layers = model_params.get("gen_layers", [256, 512, 1024])

    # Load condition processor
    cond_config_path = run_dir / "condition_config.json"
    condition_processor = ConditionProcessorGeneric(condition_cols)

    if cond_config_path.exists():
        condition_processor.load_config(str(cond_config_path))
    else:
        condition_processor.fit(val_adata)

    condition_dim = condition_processor.total_condition_dim

    try:
        # Build generator
        generator = build_generator(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            output_dim=n_genes,
            hidden_layers=gen_layers
        )

        # Load weights
        gen_weights = checkpoint_path / "generator.weights.h5"
        if not gen_weights.exists():
            print(f"[WARN] Generator weights not found: {gen_weights}")
            return False

        generator.load_weights(str(gen_weights))

        # Encode conditions from validation set
        conditions = condition_processor.encode_conditions(val_adata.obs)

        # Generate synthetic data
        noise = tf.random.normal((len(val_adata), latent_dim))
        generated = generator((noise, conditions), training=False)
        generated = tf.maximum(generated, 0).numpy()

        # Save to disk
        np.save(output_path, generated)

        # Clean up
        del generator
        tf.keras.backend.clear_session()
        gc.collect()

        return True

    except Exception as e:
        print(f"[ERROR] Failed to generate from {checkpoint_path.name}: {e}")
        import traceback
        traceback.print_exc()
        cleanup_memory('gan')
        return False


# =============================================================================
# PHASE 2: COMPUTE MMD FROM SAVED FILES
# =============================================================================

def compute_mmd_from_saved(
    generated_path: Path,
    real_val_data: np.ndarray,
    sigma: float,
    pca_model=None
) -> Optional[float]:
    """Load generated data and compute MMD, optionally in PCA space."""
    from utils.mmd import compute_mmd

    try:
        generated_data = np.load(generated_path)
        if pca_model is not None:
            generated_data = pca_model.transform(generated_data)
        mmd_value = compute_mmd(real_val_data, generated_data, sigma)
        return float(mmd_value)
    except Exception as e:
        print(f"[ERROR] Failed to compute MMD from {generated_path}: {e}")
        return None


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def run_posthoc_mmd_cvae(
    run_dir: Path,
    val_adata,
    condition_cols: List[str],
    sigma: float,
    real_val_data: np.ndarray,
    device,
    pca_model=None
) -> List[Dict]:
    """Run post-hoc MMD evaluation for all CVAE checkpoints."""
    mmd_history = []

    # Create temp directory for generated data
    temp_dir = run_dir / "posthoc_mmd_temp"
    temp_dir.mkdir(exist_ok=True)

    # Discover checkpoints
    checkpoints = discover_checkpoints(run_dir, 'cvae')

    # Add best_model and final_model
    for model_name in ['best_model.pt', 'final_model.pt']:
        model_path = run_dir / model_name
        if model_path.exists():
            checkpoints.append({
                'path': model_path,
                'epoch': model_name.replace('.pt', '').replace('_model', ''),
                'checkpoint': model_name
            })

    print(f"[INFO] Found {len(checkpoints)} CVAE checkpoints")

    # Phase 1: Generate from each checkpoint
    print("\n[Phase 1] Generating synthetic data from checkpoints...")
    generated_files = {}

    for ckpt in checkpoints:
        epoch_label = ckpt['epoch']
        output_path = temp_dir / f"generated_epoch_{epoch_label}.npy"

        if output_path.exists():
            print(f"  Epoch {epoch_label}: Using cached generation")
            generated_files[epoch_label] = output_path
            continue

        print(f"  Epoch {epoch_label}: Generating...", end=" ", flush=True)
        success = generate_from_cvae_checkpoint(
            ckpt['path'], run_dir, val_adata, condition_cols, output_path, device
        )

        if success:
            generated_files[epoch_label] = output_path
            print("OK")
        else:
            print("FAILED")

    # Phase 2: Compute MMD from saved files
    print("\n[Phase 2] Computing MMD scores...")
    for ckpt in checkpoints:
        epoch_label = ckpt['epoch']
        if epoch_label not in generated_files:
            continue

        print(f"  Epoch {epoch_label}: Computing MMD...", end=" ", flush=True)
        mmd = compute_mmd_from_saved(generated_files[epoch_label], real_val_data, sigma, pca_model)

        if mmd is not None:
            mmd_history.append({
                'epoch': epoch_label,
                'mmd': mmd,
                'checkpoint': ckpt['checkpoint']
            })
            print(f"MMD = {mmd:.6f}")
        else:
            print("FAILED")

    return mmd_history


def run_posthoc_mmd_diffusion(
    run_dir: Path,
    val_adata,
    condition_cols: List[str],
    sigma: float,
    real_val_data: np.ndarray,
    device,
    pca_model=None
) -> List[Dict]:
    """Run post-hoc MMD evaluation for all Diffusion checkpoints."""
    mmd_history = []

    # Create temp directory for generated data
    temp_dir = run_dir / "posthoc_mmd_temp"
    temp_dir.mkdir(exist_ok=True)

    # Discover checkpoints
    checkpoints = discover_checkpoints(run_dir, 'diffusion')

    # Add best_model
    for model_name in ['best_model.pt']:
        model_path = run_dir / model_name
        if model_path.exists():
            checkpoints.append({
                'path': model_path,
                'epoch': 'best',
                'checkpoint': model_name
            })

    print(f"[INFO] Found {len(checkpoints)} Diffusion checkpoints")

    # Phase 1: Generate from each checkpoint
    print("\n[Phase 1] Generating synthetic data from checkpoints...")
    generated_files = {}

    for ckpt in checkpoints:
        epoch_label = ckpt['epoch']
        output_path = temp_dir / f"generated_epoch_{epoch_label}.npy"

        if output_path.exists():
            print(f"  Epoch {epoch_label}: Using cached generation")
            generated_files[epoch_label] = output_path
            continue

        print(f"  Epoch {epoch_label}: Generating...", end=" ", flush=True)
        success = generate_from_diffusion_checkpoint(
            ckpt['path'], run_dir, val_adata, condition_cols, output_path, device
        )

        if success:
            generated_files[epoch_label] = output_path
            print("OK")
        else:
            print("FAILED")

    # Phase 2: Compute MMD from saved files
    print("\n[Phase 2] Computing MMD scores...")
    for ckpt in checkpoints:
        epoch_label = ckpt['epoch']
        if epoch_label not in generated_files:
            continue

        print(f"  Epoch {epoch_label}: Computing MMD...", end=" ", flush=True)
        mmd = compute_mmd_from_saved(generated_files[epoch_label], real_val_data, sigma, pca_model)

        if mmd is not None:
            mmd_history.append({
                'epoch': epoch_label,
                'mmd': mmd,
                'checkpoint': ckpt['checkpoint']
            })
            print(f"MMD = {mmd:.6f}")
        else:
            print("FAILED")

    return mmd_history


def run_posthoc_mmd_gan(
    run_dir: Path,
    val_adata,
    condition_cols: List[str],
    sigma: float,
    real_val_data: np.ndarray,
    training_metrics: Dict,
    pca_model=None
) -> Tuple[List[Dict], float]:
    """Run post-hoc MMD evaluation for all GAN checkpoints."""
    mmd_history = []

    # Create temp directory for generated data
    temp_dir = run_dir / "posthoc_mmd_temp"
    temp_dir.mkdir(exist_ok=True)

    # Calculate steps_per_epoch for conversion
    total_steps = training_metrics.get("total_steps", 1)
    total_epochs = training_metrics.get("total_epochs", 1)
    steps_per_epoch = total_steps / total_epochs if total_epochs > 0 else 1

    # Discover checkpoints
    checkpoints = discover_checkpoints(run_dir, 'gan')

    # Add best_model and final_model
    for model_name in ['best_model', 'final_model']:
        model_dir = run_dir / model_name
        if model_dir.exists() and (model_dir / "generator.weights.h5").exists():
            checkpoints.append({
                'path': model_dir,
                'step': model_name,
                'checkpoint': model_name
            })

    print(f"[INFO] Found {len(checkpoints)} GAN checkpoints")

    # Phase 1: Generate from each checkpoint
    print("\n[Phase 1] Generating synthetic data from checkpoints...")
    generated_files = {}

    for ckpt in checkpoints:
        step = ckpt['step']
        step_label = step if isinstance(step, str) else f"step_{step}"
        output_path = temp_dir / f"generated_{step_label}.npy"

        if output_path.exists():
            print(f"  {step_label}: Using cached generation")
            generated_files[step] = output_path
            continue

        if isinstance(step, int):
            epoch = step / steps_per_epoch
            print(f"  Step {step} (epoch {epoch:.1f}): Generating...", end=" ", flush=True)
        else:
            print(f"  {step}: Generating...", end=" ", flush=True)

        success = generate_from_gan_checkpoint(
            ckpt['path'], run_dir, val_adata, condition_cols, output_path, training_metrics
        )

        if success:
            generated_files[step] = output_path
            print("OK")
        else:
            print("FAILED")

    # Phase 2: Compute MMD from saved files
    print("\n[Phase 2] Computing MMD scores...")
    for ckpt in checkpoints:
        step = ckpt['step']
        if step not in generated_files:
            continue

        if isinstance(step, int):
            epoch = step / steps_per_epoch
            print(f"  Step {step} (epoch {epoch:.1f}): Computing MMD...", end=" ", flush=True)
        else:
            print(f"  {step}: Computing MMD...", end=" ", flush=True)

        mmd = compute_mmd_from_saved(generated_files[step], real_val_data, sigma, pca_model)

        if mmd is not None:
            if isinstance(step, int):
                mmd_history.append({
                    'step': step,
                    'epoch': round(step / steps_per_epoch, 2),
                    'mmd': mmd,
                    'checkpoint': ckpt['checkpoint']
                })
            else:
                mmd_history.append({
                    'step': step,
                    'epoch': step,
                    'mmd': mmd,
                    'checkpoint': ckpt['checkpoint']
                })
            print(f"MMD = {mmd:.6f}")
        else:
            print("FAILED")

    return mmd_history, steps_per_epoch


def find_latest_run(model_dir: Path) -> Optional[Path]:
    """Find the latest run directory for a model."""
    latest_json = model_dir / "latest_model.json"
    if latest_json.exists():
        with open(latest_json) as f:
            data = json.load(f)
            run_dir_name = data.get("run_dir", "")
            if run_dir_name:
                run_path = Path(run_dir_name)
                if not run_path.is_absolute():
                    run_path = model_dir / run_path
                if run_path.exists():
                    return run_path

    # Fallback: find most recent run_* directory
    runs = sorted(model_dir.glob("run_*"), reverse=True)
    if runs:
        return runs[0]
    return None


def run_posthoc_mmd(
    config: Dict,
    output_paths: Dict,
    models: List[str],
    dry_run: bool = False
) -> Dict[str, bool]:
    """
    Main orchestration function for post-hoc MMD calculation.

    Args:
        config: Pipeline configuration
        output_paths: Output paths dictionary
        models: List of models to evaluate ('cvae', 'diffusion', 'gan')
        dry_run: If True, only print what would be done

    Returns:
        Dict mapping model name to success status
    """
    import torch
    from utils.mmd import compute_sigma_from_real_data
    from utils.data_splitter import load_split_indices

    results = {}
    condition_cols = config["conditions"]["columns"]
    processed_data_path = config["paths"]["processed_data"]

    print("\n" + "=" * 60)
    print("POST-HOC MMD CALCULATION")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    adata = sc.read_h5ad(processed_data_path)
    print(f"[OK] Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    # Load split indices
    split_path = output_paths["base"] / "data_split_indices.npz"
    if not split_path.exists():
        print(f"[ERROR] Split indices not found: {split_path}")
        print("Run training first to generate data splits.")
        return {m: False for m in models}

    train_indices, val_indices = load_split_indices(str(split_path))
    val_adata = adata[val_indices].copy()
    print(f"[OK] Loaded validation split: {val_adata.n_obs} cells")

    # Get real validation data
    if hasattr(val_adata.X, 'toarray'):
        real_val_data = val_adata.X.toarray()
    else:
        real_val_data = np.array(val_adata.X)

    # Check for PCA-MMD mode
    pca_model = None
    pca_path = output_paths["base"] / "pca_mmd_transform.pkl"
    sigma_pca_path = output_paths["base"] / "sigma_heuristic_pca.npz"

    if pca_path.exists() and sigma_pca_path.exists():
        import joblib
        from utils.mmd_early_stopping import load_sigma_heuristic

        pca_model = joblib.load(pca_path)
        sigma = load_sigma_heuristic(str(sigma_pca_path))
        n_components = pca_model.n_components_
        print(f"[OK] PCA-MMD mode: {n_components} components")
        print(f"[OK] Loaded PCA transform from {pca_path}")
        print(f"[OK] Loaded PCA sigma from {sigma_pca_path}: {sigma:.6f}")

        # Transform real validation data into PCA space (once, upfront)
        real_val_data = pca_model.transform(real_val_data)
        print(f"[OK] Transformed validation data to PCA space: {real_val_data.shape}")
    else:
        # Original full-space MMD
        sigma_path = output_paths["base"] / "sigma_heuristic.npz"
        if sigma_path.exists():
            from utils.mmd_early_stopping import load_sigma_heuristic
            sigma = load_sigma_heuristic(str(sigma_path))
            print(f"[OK] Loaded sigma from {sigma_path}: {sigma:.6f}")
        else:
            print("[WARN] sigma_heuristic.npz not found, computing from validation data (legacy)")
            sigma = compute_sigma_from_real_data(real_val_data)
            print(f"[OK] Sigma: {sigma:.6f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[OK] Device: {device}")

    # Process each model
    for model_type in models:
        if model_type not in ['cvae', 'diffusion', 'gan']:
            continue

        print(f"\n{'='*60}")
        print(f"Processing {model_type.upper()}")
        print("=" * 60)

        model_dir = output_paths["trained_models"][model_type]
        if not model_dir.exists():
            print(f"[SKIP] Model directory not found: {model_dir}")
            results[model_type] = False
            continue

        run_dir = find_latest_run(model_dir)
        if run_dir is None:
            print(f"[SKIP] No run directory found for {model_type}")
            results[model_type] = False
            continue

        print(f"[INFO] Run directory: {run_dir}")

        if dry_run:
            print(f"[DRY-RUN] Would evaluate checkpoints in {run_dir}")
            results[model_type] = True
            continue

        try:
            # Common PCA metadata for output
            pca_metadata = {}
            if pca_model is not None:
                pca_metadata = {
                    'pca_enabled': True,
                    'pca_n_components': int(pca_model.n_components_),
                }

            if model_type == "cvae":
                mmd_history = run_posthoc_mmd_cvae(
                    run_dir, val_adata, condition_cols, sigma, real_val_data, device,
                    pca_model=pca_model
                )
                output_data = {
                    'sigma': sigma,
                    'computed_at': datetime.now().isoformat(),
                    **pca_metadata,
                    'history': mmd_history
                }

            elif model_type == "diffusion":
                mmd_history = run_posthoc_mmd_diffusion(
                    run_dir, val_adata, condition_cols, sigma, real_val_data, device,
                    pca_model=pca_model
                )
                output_data = {
                    'sigma': sigma,
                    'computed_at': datetime.now().isoformat(),
                    **pca_metadata,
                    'history': mmd_history
                }

            elif model_type == "gan":
                # Load training metrics for step-to-epoch conversion
                metrics_path = run_dir / "training_metrics.json"
                if not metrics_path.exists():
                    print(f"[ERROR] Training metrics not found: {metrics_path}")
                    results[model_type] = False
                    continue

                with open(metrics_path) as f:
                    training_metrics = json.load(f)

                mmd_history, steps_per_epoch = run_posthoc_mmd_gan(
                    run_dir, val_adata, condition_cols, sigma, real_val_data, training_metrics,
                    pca_model=pca_model
                )
                output_data = {
                    'sigma': sigma,
                    'steps_per_epoch': steps_per_epoch,
                    'computed_at': datetime.now().isoformat(),
                    **pca_metadata,
                    'history': mmd_history
                }

            # Save results
            output_file = run_dir / "posthoc_mmd_history.json"
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"\n[OK] Saved {len(mmd_history)} MMD measurements to: {output_file}")
            results[model_type] = True

        except Exception as e:
            print(f"[ERROR] Failed to process {model_type}: {e}")
            import traceback
            traceback.print_exc()
            results[model_type] = False

    # Generate combined plot if any succeeded
    successful = [m for m, success in results.items() if success]
    if successful and not dry_run:
        print("\n" + "=" * 60)
        print("GENERATING COMPARISON PLOT")
        print("=" * 60)
        try:
            from analysis.plot_mmd_history import plot_mmd_history
            plot_mmd_history(output_paths["base"], dpi=300, show_plots=False)
            print(f"[OK] MMD history plot saved to: {output_paths['base']}/analysis/metrics/mmd_history_comparison.png")
        except Exception as e:
            print(f"[WARN] Could not generate plot: {e}")

    return results
