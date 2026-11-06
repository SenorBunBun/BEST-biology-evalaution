#!/usr/bin/env python3
"""
driver.py - Main orchestrator for the scRNA-seq synthetic data generation pipeline.

Usage:
    python driver.py --config dataset_configs/cancer.json
    python driver.py --config dataset_configs/cancer.json --stage train --model cvae
    python driver.py --config dataset_configs/cancer.json --stage generate
    python driver.py --config dataset_configs/cancer.json --stage analyze
    python driver.py --config dataset_configs/cancer.json --stage posthoc-mmd
    python driver.py --config dataset_configs/cancer.json --stage posthoc-mmd --model cvae
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
import scanpy as sc

from config_loader import load_config, create_output_dirs, get_output_paths
from utils.model_tracker import ModelTracker, create_generation_metadata


def get_dataset_sample_count(data_path: str) -> int:
    """Load dataset and return the number of samples (cells)."""
    adata = sc.read_h5ad(data_path)
    return adata.n_obs


def calculate_gan_max_steps(n_epochs: int, n_samples: int, train_split: float,
                            batch_size: int, critic_iters: int) -> int:
    """
    Calculate max_steps for GAN training from epochs.

    Formula: steps_per_epoch = (n_samples * train_split / batch_size) / critic_iters
             max_steps = n_epochs * steps_per_epoch

    Args:
        n_epochs: Number of training epochs
        n_samples: Total samples in dataset
        train_split: Fraction of data for training (e.g., 0.85)
        batch_size: Batch size for training
        critic_iters: Number of critic iterations per generator step

    Returns:
        max_steps for GAN training
    """
    train_samples = int(n_samples * train_split)
    batches_per_epoch = train_samples // batch_size
    steps_per_epoch = batches_per_epoch // critic_iters
    max_steps = n_epochs * steps_per_epoch
    return max_steps


def ensure_data_split(config: Dict, output_paths: Dict, dry_run: bool = False) -> Path:
    """
    Create stratified train/val split if not exists.

    The split file is shared by all models to ensure they train on the same data.
    Stratification ensures proportional representation of all condition combinations.

    Args:
        config: Pipeline configuration
        output_paths: Output paths dict
        dry_run: If True, only print what would be done

    Returns:
        Path to the split indices file
    """
    split_path = output_paths["base"] / "data_split_indices.npz"
    processed_data_path = config["paths"]["processed_data"]

    if split_path.exists():
        print(f"[OK] Using existing data split: {split_path}")
        return split_path

    if dry_run:
        print(f"[DRY-RUN] Would create stratified data split: {split_path}")
        return split_path

    # Create new stratified split
    from utils.data_splitter import stratified_split, save_split_indices, print_split_summary

    print("\n" + "=" * 60)
    print("CREATING STRATIFIED DATA SPLIT")
    print("=" * 60)

    adata = sc.read_h5ad(processed_data_path)

    split_config = config.get("data_split", {})
    val_ratio = split_config.get("val_ratio", 0.2)
    seed = split_config.get("seed", 42)
    stratify_cols = config["conditions"]["columns"]

    print(f"Data: {processed_data_path}")
    print(f"Stratifying by: {stratify_cols}")
    print(f"Val ratio: {val_ratio}")
    print(f"Seed: {seed}")

    train_idx, val_idx = stratified_split(adata, stratify_cols, val_ratio, seed)
    save_split_indices(split_path, train_idx, val_idx)

    print_split_summary(adata, train_idx, val_idx, stratify_cols)
    print(f"[OK] Saved split indices: {split_path}")

    return split_path


def ensure_sigma_heuristic(config: Dict, output_paths: Dict, split_path: Path,
                           dry_run: bool = False) -> Path:
    """
    Pre-compute sigma heuristic from training data with stratified sampling.

    The sigma file is shared by all models to ensure consistent MMD evaluation.

    Args:
        config: Pipeline configuration
        output_paths: Output paths dict
        split_path: Path to the data split indices file
        dry_run: If True, only print what would be done

    Returns:
        Path to the sigma_heuristic.npz file
    """
    sigma_path = output_paths["base"] / "sigma_heuristic.npz"
    processed_data_path = config["paths"]["processed_data"]

    if sigma_path.exists():
        print(f"[OK] Using existing sigma heuristic: {sigma_path}")
        return sigma_path

    if dry_run:
        print(f"[DRY-RUN] Would compute sigma heuristic: {sigma_path}")
        return sigma_path

    from utils.mmd import compute_sigma_stratified
    from utils.mmd_early_stopping import save_sigma_heuristic
    from utils.data_splitter import load_split_indices

    print("\n" + "=" * 60)
    print("COMPUTING SIGMA HEURISTIC")
    print("=" * 60)

    adata = sc.read_h5ad(processed_data_path)
    train_indices, _ = load_split_indices(str(split_path))
    train_adata = adata[train_indices]

    # Get training data as dense array
    if hasattr(train_adata.X, 'toarray'):
        train_data = train_adata.X.toarray()
    else:
        train_data = np.array(train_adata.X)

    # Create combined condition labels for stratification
    condition_cols = config["conditions"]["columns"]
    condition_labels = train_adata.obs[condition_cols].astype(str).agg('_'.join, axis=1).values

    mmd_config = config.get("mmd_eval", {})
    max_sigma_samples = mmd_config.get("max_sigma_samples", 1000)
    seed = config.get("data_split", {}).get("seed", 42)

    print(f"Data: {processed_data_path}")
    print(f"Training samples: {len(train_data)}")
    print(f"Max sigma samples: {max_sigma_samples}")
    print(f"Stratifying by: {condition_cols}")

    sigma, sampled_indices = compute_sigma_stratified(
        train_data, condition_labels, max_total_samples=max_sigma_samples, seed=seed
    )

    # Gather sampled data and conditions for saving
    sampled_data = train_data[sampled_indices]
    sampled_conditions = condition_labels[sampled_indices]

    save_sigma_heuristic(
        str(sigma_path), sigma, sampled_indices, sampled_data,
        sampled_conditions, seed, max_sigma_samples
    )

    print(f"[OK] Sigma: {sigma:.6f}")
    print(f"[OK] Saved sigma heuristic: {sigma_path}")

    return sigma_path


def ensure_pca_model(config: Dict, output_paths: Dict, split_path: Path,
                     dry_run: bool = False) -> Optional[Path]:
    """
    Pre-compute PCA transformation from the full real dataset for PCA-space MMD.

    The PCA is fit on the entire processed dataset (all cells), then the
    PCA-space sigma is computed from stratified training data. Both the fitted
    PCA transform and PCA sigma are saved persistently and reused by posthoc-mmd.

    Args:
        config: Pipeline configuration
        output_paths: Output paths dict
        split_path: Path to the data split indices file
        dry_run: If True, only print what would be done

    Returns:
        Path to the pca_mmd_transform.pkl file, or None if PCA-MMD is not enabled
    """
    mmd_config = config.get("mmd_eval", {})
    pca_config = mmd_config.get("pca_mmd", {})
    if not pca_config.get("enabled", False):
        return None

    n_components = pca_config.get("n_components", 50)
    pca_path = output_paths["base"] / "pca_mmd_transform.pkl"
    sigma_pca_path = output_paths["base"] / "sigma_heuristic_pca.npz"
    processed_data_path = config["paths"]["processed_data"]

    if pca_path.exists() and sigma_pca_path.exists():
        import joblib
        existing_pca = joblib.load(pca_path)
        if existing_pca.n_components_ == n_components:
            print(f"[OK] Using existing PCA transform ({n_components} components): {pca_path}")
            print(f"[OK] Using existing PCA sigma: {sigma_pca_path}")
            return pca_path
        else:
            print(f"[INFO] PCA n_components changed ({existing_pca.n_components_} -> {n_components}), recomputing...")
            del existing_pca

    if dry_run:
        print(f"[DRY-RUN] Would compute PCA transform ({n_components} components): {pca_path}")
        return pca_path

    import joblib
    from sklearn.decomposition import PCA
    from utils.mmd import compute_sigma_stratified
    from utils.mmd_early_stopping import save_sigma_heuristic
    from utils.data_splitter import load_split_indices

    print("\n" + "=" * 60)
    print("COMPUTING PCA TRANSFORM FOR MMD")
    print("=" * 60)

    # Load full dataset and fit PCA on ALL real data
    adata = sc.read_h5ad(processed_data_path)
    if hasattr(adata.X, 'toarray'):
        full_data = adata.X.toarray()
    else:
        full_data = np.array(adata.X)

    print(f"Data: {processed_data_path}")
    print(f"Total cells: {len(full_data)}")
    print(f"Original dimensions: {full_data.shape[1]}")
    print(f"PCA components: {n_components}")

    pca = PCA(n_components=n_components)
    pca.fit(full_data)

    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"[OK] PCA fit complete - explained variance: {explained_var:.1f}%")

    # Save fitted PCA transform
    joblib.dump(pca, pca_path)
    print(f"[OK] Saved PCA transform: {pca_path}")

    # Compute sigma in PCA space from stratified training data
    train_indices, _ = load_split_indices(str(split_path))
    train_data = full_data[train_indices]
    train_data_pca = pca.transform(train_data)

    condition_cols = config["conditions"]["columns"]
    train_adata = adata[train_indices]
    condition_labels = train_adata.obs[condition_cols].astype(str).agg('_'.join, axis=1).values

    max_sigma_samples = mmd_config.get("max_sigma_samples", 1000)
    seed = config.get("data_split", {}).get("seed", 42)

    print(f"Computing PCA-space sigma (stratified, {len(train_data_pca)} training samples)...")

    sigma_pca, sampled_indices = compute_sigma_stratified(
        train_data_pca, condition_labels, max_total_samples=max_sigma_samples, seed=seed
    )

    sampled_data = train_data_pca[sampled_indices]
    sampled_conditions = condition_labels[sampled_indices]

    save_sigma_heuristic(
        str(sigma_pca_path), sigma_pca, sampled_indices, sampled_data,
        sampled_conditions, seed, max_sigma_samples
    )

    print(f"[OK] PCA-space sigma: {sigma_pca:.6f}")
    print(f"[OK] Saved PCA sigma: {sigma_pca_path}")

    return pca_path


# Valid options
STAGES = ['all', 'preprocess', 'train', 'generate', 'analyze', 'posthoc-mmd']
MODELS = ['all', 'cvae', 'vae', 'diffusion', 'gan']


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_step(text: str, status: str = "...") -> None:
    """Print a step indicator."""
    print(f"  [{status}] {text}")


def run_subprocess(cmd: List[str], description: str, dry_run: bool = False) -> bool:
    """
    Run a subprocess command.

    Args:
        cmd: Command and arguments as list
        description: Description for logging
        dry_run: If True, only print command without executing

    Returns:
        True if successful, False otherwise
    """
    print_step(description, "RUN")
    print(f"      Command: {' '.join(cmd)}")

    if dry_run:
        print(f"      [DRY RUN] Would execute above command")
        return True

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print_step(description, "OK")
            return True
        else:
            print_step(description, "FAIL")
            return False
    except Exception as e:
        print_step(f"{description} - Error: {e}", "FAIL")
        return False


# ============================================================================
# STAGE: PREPROCESSING
# ============================================================================

def run_preprocessing(config: Dict, output_paths: Dict, dry_run: bool = False) -> bool:
    """Run data preprocessing stage."""
    print_header("STAGE: Preprocessing")

    preprocessing_enabled = config.get("preprocessing", {}).get("enabled", True)
    preprocess_config = config.get("preprocessing", {})
    raw_data = config["paths"].get("raw_data")
    processed_data = config["paths"]["processed_data"]

    if not preprocessing_enabled:
        print_step("Preprocessing disabled in config", "SKIP")
        return True

    # Preprocessing is enabled - raw_data is required
    if not raw_data:
        print_step("Preprocessing enabled but no raw_data path specified", "FAIL")
        print_step("Either set paths.raw_data or disable preprocessing", "INFO")
        return False

    # Check if raw data exists
    if not Path(raw_data).exists():
        print_step(f"Raw data file not found: {raw_data}", "FAIL")
        return False

    # Check if processed data already exists
    if Path(processed_data).exists():
        print_step(f"Processed data already exists: {processed_data}", "INFO")
        print_step("To re-preprocess, delete the existing file or change output path", "INFO")
        return True

    print_step(f"Preprocessing: {raw_data}", "...")

    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "preprocessing" / "preprocess.py"),
        "--input", raw_data,
        "--output", processed_data,
        "--min-cells", str(preprocess_config.get("min_cells", 3)),
        "--sparsity-threshold", str(preprocess_config.get("sparsity_threshold", 0.997)),
        "--n-hvg", str(preprocess_config.get("n_hvg", 2000)),
        "--target-sum", str(preprocess_config.get("target_sum", 1e4)),
    ]

    # Handle normalization option
    if not preprocess_config.get("normalize", True):
        cmd.append("--no-normalize")

    return run_subprocess(cmd, "Data preprocessing", dry_run)


# ============================================================================
# STAGE: TRAINING
# ============================================================================

def run_cvae_training(config: Dict, output_paths: Dict, dry_run: bool = False,
                      split_path: Optional[Path] = None,
                      sigma_path: Optional[Path] = None,
                      pca_path: Optional[Path] = None) -> bool:
    """Run CVAE model training."""
    cvae_config = config["models"]["cvae"]

    if not cvae_config.get("enabled", True):
        print_step("CVAE training disabled in config", "SKIP")
        return True

    print_step("Training CVAE model", "...")

    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "models" / "CVAE" / "train.py"),
        "--data", config["paths"]["processed_data"],
        "--output-dir", str(output_paths["trained_models"]["cvae"]),
        "--condition-cols", ",".join(config["conditions"]["columns"]),
        "--latent-size", str(cvae_config.get("latent_size", 32)),
        "--hidden-dims", ",".join(map(str, cvae_config.get("hidden_dims", [800, 400]))),
        "--beta", str(cvae_config.get("beta", 0.05)),
        "--dropout", str(cvae_config.get("dropout", 0.03)),
        "--lr", str(cvae_config.get("lr", 0.01)),
        "--batch-size", str(cvae_config.get("batch_size", 128)),
        "--n-epochs", str(cvae_config.get("n_epochs", 50)),
        "--patience", str(cvae_config.get("patience", 15)),
        "--checkpoint-freq", str(cvae_config.get("checkpoint_freq", 10)),
    ]

    # Add split indices if provided
    if split_path:
        cmd.extend(["--split-indices", str(split_path)])

    # Add MMD evaluation args if enabled
    mmd_config = config.get("mmd_eval", {})
    if mmd_config.get("enabled", False):
        cmd.extend(["--mmd-eval"])
        cmd.extend(["--mmd-eval-freq", str(mmd_config.get("cvae_freq", 5))])
        if sigma_path:
            cmd.extend(["--sigma-path", str(sigma_path)])
        cmd.extend(["--mmd-patience", str(mmd_config.get("patience_mmd", 3))])
        cmd.extend(["--mmd-min-delta", str(mmd_config.get("min_delta", 0.001))])
        if pca_path:
            cmd.extend(["--pca-path", str(pca_path)])

    return run_subprocess(cmd, "CVAE training", dry_run)


def run_diffusion_training(config: Dict, output_paths: Dict, dry_run: bool = False,
                           vae_only: bool = False, vae_model_path: Optional[Path] = None,
                           split_path: Optional[Path] = None,
                           sigma_path: Optional[Path] = None,
                           pca_path: Optional[Path] = None) -> bool:
    """Run Diffusion model training (VAE first, then Diffusion).

    Args:
        config: Pipeline config
        output_paths: Output directory paths
        dry_run: If True, only print commands
        vae_only: If True, only train VAE (Phase 1), skip diffusion training
        vae_model_path: Path to pre-trained VAE model (skip VAE training if provided)
    """
    diff_config = config["models"]["diffusion"]

    if not diff_config.get("enabled", True):
        print_step("Diffusion training disabled in config", "SKIP")
        return True

    if vae_only:
        print_step("Training VAE only (Phase 1 for Diffusion)", "...")
    elif vae_model_path:
        print_step(f"Training Diffusion model (using pre-trained VAE: {vae_model_path.name})", "...")
    else:
        print_step("Training Diffusion model (Phase 1: VAE, Phase 2: Diffusion)", "...")

    # Get condition columns and types
    condition_cols = config["conditions"]["columns"]
    condition_types = config["conditions"].get("types", {})

    # Build condition types string in same order as columns
    condition_types_list = [condition_types.get(col, "categorical") for col in condition_cols]

    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "models" / "Diffusion" / "train.py"),
        "--data", config["paths"]["processed_data"],
        "--output-dir", str(output_paths["trained_models"]["diffusion"]),
        "--condition-cols", ",".join(condition_cols),
        "--condition-types", ",".join(condition_types_list),
        # VAE architecture parameters
        "--vae-latent-size", str(diff_config.get("vae_latent_size", 128)),
        "--vae-hidden-dims", ",".join(map(str, diff_config.get("vae_hidden_dims", [1024, 1024]))),
        # VAE training parameters
        "--vae-beta", str(diff_config.get("vae_beta", 0.001)),
        "--vae-dropout", str(diff_config.get("vae_dropout", 0.1)),
        "--vae-lr", str(diff_config.get("vae_lr", 0.001)),
        "--vae-weight-decay", str(diff_config.get("vae_weight_decay", 0.001)),
        "--vae-n-epochs", str(diff_config.get("vae_n_epochs", 200)),
        "--vae-patience", str(diff_config.get("vae_patience", 20)),
        "--vae-overfitting-patience", str(diff_config.get("vae_overfitting_patience", 10)),
        "--vae-min-delta", str(diff_config.get("vae_min_delta", 1e-4)),
        # Diffusion architecture parameters
        "--latent-size", str(diff_config.get("latent_size", 128)),
        "--hidden-dims", ",".join(map(str, diff_config.get("hidden_dims", [512, 512, 256, 128]))),
        "--num-timesteps", str(diff_config.get("num_timesteps", 1000)),
        "--beta-schedule", diff_config.get("beta_schedule", "cosine"),
        "--classifier-free-prob", str(diff_config.get("classifier_free_prob", 0.1)),
        "--dropout", str(diff_config.get("dropout", 0.1)),
        # Diffusion training parameters
        "--lr", str(diff_config.get("lr", 1e-4)),
        "--weight-decay", str(diff_config.get("weight_decay", 1e-5)),
        "--batch-size", str(diff_config.get("batch_size", 128)),
        "--n-epochs", str(diff_config.get("n_epochs", 50)),
        "--patience", str(diff_config.get("patience", 15)),
        "--train-split", str(diff_config.get("train_split", 0.8)),
        "--val-split", str(diff_config.get("val_split", 0.1)),
        "--checkpoint-freq", str(diff_config.get("checkpoint_freq", 5)),
    ]

    # Add --vae-only flag if requested
    if vae_only:
        cmd.append("--vae-only")

    # Add --vae-model-path if a pre-trained VAE is provided
    if vae_model_path and not vae_only:
        cmd.extend(["--vae-model-path", str(vae_model_path)])

    # Add split indices if provided
    if split_path:
        cmd.extend(["--split-indices", str(split_path)])

    # Add MMD evaluation args if enabled
    mmd_config = config.get("mmd_eval", {})
    if mmd_config.get("enabled", False):
        cmd.extend(["--mmd-eval"])
        cmd.extend(["--mmd-eval-freq", str(mmd_config.get("diffusion_freq", 5))])
        if sigma_path:
            cmd.extend(["--sigma-path", str(sigma_path)])
        cmd.extend(["--mmd-patience", str(mmd_config.get("patience_mmd", 3))])
        cmd.extend(["--mmd-min-delta", str(mmd_config.get("min_delta", 0.001))])
        if pca_path:
            cmd.extend(["--pca-path", str(pca_path)])

    description = "VAE training (Phase 1)" if vae_only else "Diffusion training"
    return run_subprocess(cmd, description, dry_run)


def run_gan_training(config: Dict, output_paths: Dict, dry_run: bool = False,
                     split_path: Optional[Path] = None,
                     sigma_path: Optional[Path] = None,
                     pca_path: Optional[Path] = None) -> bool:
    """Run GAN model training."""
    gan_config = config["models"]["gan"]

    if not gan_config.get("enabled", True):
        print_step("GAN training disabled in config", "SKIP")
        return True

    print_step("Training GAN model", "...")

    # Determine max_steps: either from config directly or calculated from epochs
    training_unit = gan_config.get("training_unit", "steps")
    batch_size = gan_config.get("batch_size", 32)
    critic_iters = gan_config.get("critic_iters", 4)
    train_split = gan_config.get("train_split", 0.85)

    if training_unit == "epochs":
        n_epochs = gan_config.get("n_epochs", 100)
        # Get dataset sample count
        n_samples = get_dataset_sample_count(config["paths"]["processed_data"])
        max_steps = calculate_gan_max_steps(n_epochs, n_samples, train_split, batch_size, critic_iters)

        # Calculate steps_per_epoch for display
        train_samples = int(n_samples * train_split)
        batches_per_epoch = train_samples // batch_size
        steps_per_epoch = batches_per_epoch // critic_iters

        print_step(f"GAN: {n_epochs} epochs -> {max_steps} steps "
                   f"({n_samples} samples, {steps_per_epoch} steps/epoch)", "INFO")
    else:
        max_steps = gan_config.get("max_steps", 300000)
        print_step(f"GAN: {max_steps} steps (direct)", "INFO")

    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "models" / "GAN" / "train.py"),
        "--data", config["paths"]["processed_data"],
        "--output-dir", str(output_paths["trained_models"]["gan"]),
        "--condition-cols", ",".join(config["conditions"]["columns"]),
        "--latent-dim", str(gan_config.get("latent_dim", 128)),
        "--gen-layers", ",".join(map(str, gan_config.get("gen_layers", [256, 512, 1024]))),
        "--critic-layers", ",".join(map(str, gan_config.get("critic_layers", [1024, 512, 256]))),
        "--lambda-gp", str(gan_config.get("lambda_gp", 15.0)),
        "--lr-g", str(gan_config.get("lr_g", 0.00015)),
        "--lr-c", str(gan_config.get("lr_c", 0.00015)),
        "--beta1", str(gan_config.get("beta1", 0.0)),
        "--beta2", str(gan_config.get("beta2", 0.9)),
        "--batch-size", str(batch_size),
        "--max-steps", str(max_steps),
        "--critic-iters", str(critic_iters),
        "--train-split", str(train_split),
        "--save-freq", str(gan_config.get("save_freq", 500 if training_unit != "epochs" else 5)),
        "--save-freq-unit", training_unit if training_unit == "epochs" else "steps",
        "--log-freq", str(gan_config.get("log_freq", 100)),
        "--validation-freq", str(gan_config.get("validation_freq", 1)),
        "--gp-target", str(gan_config.get("gp_target", 0.6)),
    ]

    # Add optional gradient clipping
    if gan_config.get("clip_gradients", True):
        cmd.append("--clip-gradients")
        cmd.extend(["--gradient-clip-norm", str(gan_config.get("gradient_clip_norm", 1.0))])

    # Add split indices if provided
    if split_path:
        cmd.extend(["--split-indices", str(split_path)])

    # Add MMD evaluation flag if enabled
    mmd_config = config.get("mmd_eval", {})
    if mmd_config.get("enabled", False):
        cmd.extend(["--mmd-eval"])
        # gan_freq replaces save_freq when MMD is active
        save_freq_idx = cmd.index("--save-freq")
        cmd[save_freq_idx + 1] = str(mmd_config.get("gan_freq", 1))
        if sigma_path:
            cmd.extend(["--sigma-path", str(sigma_path)])
        cmd.extend(["--mmd-patience", str(mmd_config.get("patience_mmd", 3))])
        cmd.extend(["--mmd-min-delta", str(mmd_config.get("min_delta", 0.001))])
        if training_unit == "epochs":
            cmd.extend(["--mmd-n-epochs", str(n_epochs)])
        if pca_path:
            cmd.extend(["--pca-path", str(pca_path)])

    return run_subprocess(cmd, "GAN training", dry_run)


def find_latest_vae_model(output_paths: Dict) -> Optional[Path]:
    """Find the most recently trained VAE model from diffusion training.

    First checks latest_model.json for vae_model_path, then falls back to
    scanning run directories.
    """
    diffusion_dir = output_paths["trained_models"]["diffusion"]
    if not diffusion_dir.exists():
        return None

    # First, try to get VAE path from latest_model.json
    latest_model_json = diffusion_dir / "latest_model.json"
    if latest_model_json.exists():
        try:
            import json
            with open(latest_model_json, 'r') as f:
                latest_info = json.load(f)
            if "vae_model_path" in latest_info:
                vae_path = diffusion_dir / latest_info["vae_model_path"]
                if vae_path.exists():
                    return vae_path
        except Exception:
            pass  # Fall back to directory scanning

    # Fallback: Look for vae_model.pt in run directories
    run_dirs = sorted(diffusion_dir.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        vae_path = run_dir / "vae_model.pt"
        if vae_path.exists():
            return vae_path

    return None


def run_training(config: Dict, output_paths: Dict, models: List[str],
                 dry_run: bool = False) -> Dict[str, bool]:
    """
    Run training stage for specified models.

    Args:
        config: Pipeline config
        output_paths: Output directory paths
        models: List of models to train ('all' or specific models)
        dry_run: If True, only print commands

    Returns:
        Dict mapping model name to success status
    """
    print_header("STAGE: Training")

    # Ensure stratified data split exists (shared by all models)
    split_path = ensure_data_split(config, output_paths, dry_run)

    # Pre-compute sigma heuristic if MMD early stopping is enabled
    sigma_path = None
    pca_path = None
    mmd_config = config.get("mmd_eval", {})
    if mmd_config.get("enabled", False):
        pca_config = mmd_config.get("pca_mmd", {})
        if pca_config.get("enabled", False):
            # PCA-MMD: compute PCA transform and PCA-space sigma (no regular sigma needed)
            pca_path = ensure_pca_model(config, output_paths, split_path, dry_run)
            sigma_path = output_paths["base"] / "sigma_heuristic_pca.npz"
        else:
            # Regular full-space sigma
            sigma_path = ensure_sigma_heuristic(config, output_paths, split_path, dry_run)

    results = {}
    train_models = MODELS[1:] if 'all' in models else models  # Exclude 'all' from list

    if 'cvae' in train_models:
        results['cvae'] = run_cvae_training(config, output_paths, dry_run, split_path, sigma_path, pca_path)

    # VAE-only training (for diffusion model's VAE component)
    # Only train VAE if explicitly requested and diffusion is NOT being trained
    # (to avoid duplicate VAE training)
    if 'vae' in train_models and 'diffusion' not in train_models:
        results['vae'] = run_diffusion_training(config, output_paths, dry_run, vae_only=True,
                                                 split_path=split_path, sigma_path=sigma_path,
                                                 pca_path=pca_path)

    # Full diffusion training (VAE + Diffusion)
    if 'diffusion' in train_models:
        # Check if VAE was already trained (from a previous vae-only run)
        vae_path = find_latest_vae_model(output_paths)
        if vae_path and 'vae' in train_models:
            print_step(f"Using pre-trained VAE from: {vae_path}", "INFO")
        results['diffusion'] = run_diffusion_training(
            config, output_paths, dry_run, vae_only=False, vae_model_path=vae_path,
            split_path=split_path, sigma_path=sigma_path, pca_path=pca_path
        )

    if 'gan' in train_models:
        results['gan'] = run_gan_training(config, output_paths, dry_run, split_path, sigma_path, pca_path)

    return results


# ============================================================================
# STAGE: GENERATION
# ============================================================================

def get_model_checkpoint(config: Dict, output_paths: Dict, model_type: str,
                         checkpoint_override: Optional[str] = None) -> Optional[Path]:
    """
    Get the model checkpoint path for generation.

    Args:
        config: Pipeline config
        output_paths: Output directory paths
        model_type: Model type (cvae, diffusion, gan)
        checkpoint_override: CLI override for checkpoint path

    Returns:
        Path to model checkpoint, or None if not found
    """
    model_config = config["models"][model_type]
    checkpoint_spec = checkpoint_override or model_config.get("use_checkpoint", "latest")

    tracker = ModelTracker(output_paths["trained_models"][model_type], model_type)
    return tracker.get_model_path(checkpoint_spec)


def run_cvae_generation(config: Dict, output_paths: Dict,
                        checkpoint_override: Optional[str] = None,
                        dry_run: bool = False) -> Optional[str]:
    """Run CVAE synthetic data generation."""
    cvae_config = config["models"]["cvae"]

    if not cvae_config.get("enabled", True):
        print_step("CVAE generation disabled in config", "SKIP")
        return None

    # Get model checkpoint
    model_path = get_model_checkpoint(config, output_paths, "cvae", checkpoint_override)
    if model_path is None:
        print_step("No CVAE model found - train first or specify checkpoint", "FAIL")
        return None

    print_step(f"Using CVAE model: {model_path}", "INFO")

    # Output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_paths["synthetic_data"] / f"cvae_{config['dataset_name']}_{timestamp}.h5ad"
    meta_file = output_paths["synthetic_data"] / f"cvae_{config['dataset_name']}_{timestamp}_meta.json"

    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "models" / "CVAE" / "generate.py"),
        "--data", config["paths"]["processed_data"],
        "--model", str(model_path),
        "--output", str(output_file),
        "--condition-cols", ",".join(config["conditions"]["columns"]),
        "--latent-size", str(cvae_config.get("latent_size", 32)),
    ]

    success = run_subprocess(cmd, "CVAE generation", dry_run)

    if success and not dry_run:
        # Save metadata
        create_generation_metadata(
            model_path=str(model_path),
            source_data=config["paths"]["processed_data"],
            model_params=cvae_config,
            output_path=meta_file
        )
        return str(output_file)

    return str(output_file) if dry_run else None


def run_diffusion_generation(config: Dict, output_paths: Dict,
                             checkpoint_override: Optional[str] = None,
                             dry_run: bool = False) -> Optional[str]:
    """Run Diffusion synthetic data generation."""
    diff_config = config["models"]["diffusion"]

    if not diff_config.get("enabled", True):
        print_step("Diffusion generation disabled in config", "SKIP")
        return None

    # Get model checkpoint
    model_path = get_model_checkpoint(config, output_paths, "diffusion", checkpoint_override)
    if model_path is None:
        print_step("No Diffusion model found - train first or specify checkpoint", "FAIL")
        return None

    print_step(f"Using Diffusion model: {model_path}", "INFO")

    # Output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_paths["synthetic_data"] / f"diffusion_{config['dataset_name']}_{timestamp}.h5ad"
    meta_file = output_paths["synthetic_data"] / f"diffusion_{config['dataset_name']}_{timestamp}_meta.json"

    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "models" / "Diffusion" / "generate.py"),
        "--data", config["paths"]["processed_data"],
        "--checkpoint-dir", str(model_path.parent),
        "--output", str(output_file),
        "--condition-cols", ",".join(config["conditions"]["columns"]),
    ]

    success = run_subprocess(cmd, "Diffusion generation", dry_run)

    if success and not dry_run:
        create_generation_metadata(
            model_path=str(model_path),
            source_data=config["paths"]["processed_data"],
            model_params=diff_config,
            output_path=meta_file
        )
        return str(output_file)

    return str(output_file) if dry_run else None


def run_gan_generation(config: Dict, output_paths: Dict,
                       checkpoint_override: Optional[str] = None,
                       dry_run: bool = False) -> Optional[str]:
    """Run GAN synthetic data generation."""
    gan_config = config["models"]["gan"]

    if not gan_config.get("enabled", True):
        print_step("GAN generation disabled in config", "SKIP")
        return None

    # Get model checkpoint - for GAN, we need the run directory
    tracker = ModelTracker(output_paths["trained_models"]["gan"], "gan")

    if checkpoint_override:
        model_path = Path(checkpoint_override)
    else:
        latest_info = tracker.get_latest_model_info()
        if latest_info is None:
            print_step("No GAN model found - train first or specify checkpoint", "FAIL")
            return None

        # For GAN, use the run_dir since it's a directory-based model
        model_path = output_paths["trained_models"]["gan"] / latest_info["run_dir"]

    if not model_path.exists():
        print_step(f"GAN model not found at: {model_path}", "FAIL")
        return None

    print_step(f"Using GAN model: {model_path}", "INFO")

    # Output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_paths["synthetic_data"] / f"gan_{config['dataset_name']}_{timestamp}.h5ad"
    meta_file = output_paths["synthetic_data"] / f"gan_{config['dataset_name']}_{timestamp}_meta.json"

    # Build command
    # GAN checkpoint is a directory (e.g., best_model/) - find parent run directory
    checkpoint_dir = model_path if model_path.is_dir() else model_path.parent
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "models" / "GAN" / "generate.py"),
        "--data", config["paths"]["processed_data"],
        "--checkpoint-dir", str(checkpoint_dir),
        "--output", str(output_file),
        "--condition-cols", ",".join(config["conditions"]["columns"]),
        "--latent-dim", str(gan_config.get("latent_dim", 128)),
    ]

    success = run_subprocess(cmd, "GAN generation", dry_run)

    if success and not dry_run:
        create_generation_metadata(
            model_path=str(model_path),
            source_data=config["paths"]["processed_data"],
            model_params=gan_config,
            output_path=meta_file
        )
        return str(output_file)

    return str(output_file) if dry_run else None


def run_generation(config: Dict, output_paths: Dict, models: List[str],
                   checkpoint_override: Optional[str] = None,
                   dry_run: bool = False) -> Dict[str, Optional[str]]:
    """
    Run generation stage for specified models.

    Returns:
        Dict mapping model name to generated file path (or None if failed)
    """
    print_header("STAGE: Generation")

    results = {}
    gen_models = MODELS[1:] if 'all' in models else models

    if 'cvae' in gen_models:
        results['cvae'] = run_cvae_generation(config, output_paths, checkpoint_override, dry_run)

    if 'diffusion' in gen_models:
        results['diffusion'] = run_diffusion_generation(config, output_paths, checkpoint_override, dry_run)

    if 'gan' in gen_models:
        results['gan'] = run_gan_generation(config, output_paths, checkpoint_override, dry_run)

    return results


# ============================================================================
# STAGE: ANALYSIS
# ============================================================================

def find_latest_synthetic_data(output_paths: Dict, model_type: str) -> Optional[Path]:
    """Find the most recently generated synthetic data file for a model."""
    pattern = f"{model_type}_*.h5ad"
    files = list(output_paths["synthetic_data"].glob(pattern))
    # Exclude metadata files
    files = [f for f in files if not f.name.endswith("_meta.json")]

    if not files:
        return None

    # Sort by modification time, newest first
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0]


def run_analysis(config: Dict, output_paths: Dict,
                 synthetic_paths: Optional[Dict[str, str]] = None,
                 dry_run: bool = False,
                 show_plots: bool = False) -> Dict[str, bool]:
    """
    Run analysis stage.

    Args:
        config: Pipeline config
        output_paths: Output directory paths
        synthetic_paths: Dict of model_type -> synthetic data path (auto-detected if None)
        dry_run: If True, only print commands
        show_plots: If True, show plots interactively

    Returns:
        Dict mapping analysis type to success status
    """
    print_header("STAGE: Analysis")

    # Find synthetic data paths if not provided
    if synthetic_paths is None:
        synthetic_paths = {}
        for model in ['cvae', 'diffusion', 'gan']:
            path = find_latest_synthetic_data(output_paths, model)
            if path:
                synthetic_paths[model] = str(path)
                print_step(f"Found {model} synthetic data: {path.name}", "INFO")
            else:
                print_step(f"No synthetic data found for {model}", "WARN")

    results = {}
    analysis_config = config.get("analysis", {})
    conditions = config["conditions"]
    real_data = config["paths"]["processed_data"]

    # Get display names for groups (fallback to actual values if not provided)
    primary = conditions["primary"]
    group1_display = primary.get("group1_display_name", str(primary["group1"]))
    group2_display = primary.get("group2_display_name", str(primary["group2"]))

    # Run DEG analysis: compute real once, each synthetic once, then compare
    if analysis_config.get("deg", {}).get("enabled", True):
        deg_config = analysis_config.get("deg", {})
        deg_dir = output_paths["analysis"]["deg"]

        # 1. Compute real DEGs once
        real_deg_dir = deg_dir / "real"
        real_deg_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "analysis" / "compute_deg.py"),
            "--data", real_data,
            "--output-dir", str(real_deg_dir),
            "--name", "Real",
            "--column-name", primary["column"],
            "--group1", str(primary["group1"]),
            "--group2", str(primary["group2"]),
            "--method", deg_config.get("method", "wilcoxon"),
            "--n-genes", str(deg_config.get("n_genes", 50)),
        ]
        results['deg_real'] = run_subprocess(cmd, "DEG analysis (Real)", dry_run)

        # 2. Compute synthetic DEGs for each model
        for model, synth_path in synthetic_paths.items():
            output_dir = deg_dir / model
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "analysis" / "compute_deg.py"),
                "--data", synth_path,
                "--output-dir", str(output_dir),
                "--name", f"Synthetic_{model.upper()}",
                "--column-name", primary["column"],
                "--group1", str(primary["group1"]),
                "--group2", str(primary["group2"]),
                "--method", deg_config.get("method", "wilcoxon"),
                "--n-genes", str(deg_config.get("n_genes", 50)),
            ]
            results[f'deg_{model}'] = run_subprocess(
                cmd, f"DEG analysis ({model})", dry_run
            )

        # 3. Run comparison + combined plot
        core_output_dir = output_paths["analysis"]["base"]
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "analysis" / "shared_deg_comparison.py"),
            "--deg-dir", str(deg_dir),
            "--output-dir", str(deg_dir),
            "--core-output-dir", str(core_output_dir),
            "--real-results", str(real_deg_dir / "deg_results.csv"),
            "--dataset-name", config["dataset_name"],
            "--group1-display", group1_display,
            "--group2-display", group2_display,
            "--n-genes", str(deg_config.get("n_genes", 50)),
        ]
        if show_plots:
            cmd.append("--show-plots")

        results['deg_comparison'] = run_subprocess(
            cmd, "DEG comparison + combined plot", dry_run
        )

    # Run RF classifier analysis: compute real once, each synthetic once, then compare
    if analysis_config.get("classifier", {}).get("enabled", True):
        clf_config = analysis_config.get("classifier", {})
        clf_dir = output_paths["analysis"]["classifier"]

        # 1. Compute real RF classification once
        real_clf_dir = clf_dir / "real"
        real_clf_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "analysis" / "compute_classifier.py"),
            "--data", real_data,
            "--output-dir", str(real_clf_dir),
            "--name", "Real",
            "--column-name", primary["column"],
            "--group1", str(primary["group1"]),
            "--group2", str(primary["group2"]),
            "--n-genes", str(clf_config.get("n_genes", 50)),
            "--n-estimators", str(clf_config.get("n_estimators", 100)),
            "--n-folds", str(clf_config.get("n_folds", 5)),
            "--n-repeats", str(clf_config.get("n_repeats", 3)),
        ]
        results['classifier_real'] = run_subprocess(cmd, "RF classifier (Real)", dry_run)

        # 2. Compute synthetic RF classification for each model
        for model, synth_path in synthetic_paths.items():
            output_dir = clf_dir / model
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "analysis" / "compute_classifier.py"),
                "--data", synth_path,
                "--output-dir", str(output_dir),
                "--name", f"Synthetic_{model.upper()}",
                "--column-name", primary["column"],
                "--group1", str(primary["group1"]),
                "--group2", str(primary["group2"]),
                "--n-genes", str(clf_config.get("n_genes", 50)),
                "--n-estimators", str(clf_config.get("n_estimators", 100)),
                "--n-folds", str(clf_config.get("n_folds", 5)),
                "--n-repeats", str(clf_config.get("n_repeats", 3)),
            ]
            results[f'classifier_{model}'] = run_subprocess(
                cmd, f"RF classifier ({model})", dry_run
            )

        # 3. Run comparison + combined plot
        core_output_dir = output_paths["analysis"]["base"]
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "analysis" / "shared_classifier_comparison.py"),
            "--clf-dir", str(clf_dir),
            "--real-results", str(real_clf_dir / "rf_results.csv"),
            "--output-dir", str(clf_dir),
            "--core-output-dir", str(core_output_dir),
            "--dataset-name", config["dataset_name"],
            "--group1-display", group1_display,
            "--group2-display", group2_display,
            "--n-genes", str(clf_config.get("n_genes", 50)),
        ]
        if show_plots:
            cmd.append("--show-plots")

        results['classifier_comparison'] = run_subprocess(
            cmd, "RF classifier comparison + combined plot", dry_run
        )

    # UMAP analysis
    if analysis_config.get("umap", {}).get("enabled", True):
        umap_output = output_paths["analysis"]["umap"]
        umap_output.mkdir(parents=True, exist_ok=True)
        core_output_dir = output_paths["analysis"]["base"]

        cmd = [
            sys.executable,
            str(Path(__file__).parent / "analysis" / "run_umap_analysis.py"),
            "--real", real_data,
            "--output-dir", str(umap_output),
            "--core-output-dir", str(core_output_dir),
            "--dataset-name", config["dataset_name"],
        ]

        # Add synthetic data paths for each available model
        for model, synth_path in synthetic_paths.items():
            cmd.extend([f"--{model}", synth_path])

        if show_plots:
            cmd.append("--show-plots")

        results['umap'] = run_subprocess(cmd, "UMAP analysis", dry_run)

    # Scientific metrics analysis
    if analysis_config.get("metrics", {}).get("enabled", True):
        metrics_output = output_paths["analysis"]["metrics"]
        metrics_output.mkdir(parents=True, exist_ok=True)
        core_output_dir = output_paths["analysis"]["base"]

        cmd = [
            sys.executable,
            str(Path(__file__).parent / "analysis" / "run_metrics_analysis.py"),
            "--real", real_data,
            "--output-dir", str(metrics_output),
            "--core-output-dir", str(core_output_dir),
            "--dataset-name", config["dataset_name"],
        ]

        # Add synthetic data paths for each available model
        for model, synth_path in synthetic_paths.items():
            cmd.extend([f"--{model}", synth_path])

        if show_plots:
            cmd.append("--show-plots")

        results['metrics'] = run_subprocess(cmd, "Scientific metrics analysis", dry_run)

    return results


# ============================================================================
# STAGE: POST-HOC MMD
# ============================================================================

def run_posthoc_mmd_stage(config: Dict, output_paths: Dict, models: List[str],
                          dry_run: bool = False) -> Dict[str, bool]:
    """
    Run post-hoc MMD calculation for all checkpoints.

    Args:
        config: Pipeline config
        output_paths: Output directory paths
        models: List of models to evaluate ('cvae', 'diffusion', 'gan')
        dry_run: If True, only print what would be done

    Returns:
        Dict mapping model name to success status
    """
    print_header("STAGE: Post-hoc MMD Calculation")

    from utils.posthoc_mmd import run_posthoc_mmd

    # Filter to valid model types for MMD
    mmd_models = [m for m in models if m in ['cvae', 'diffusion', 'gan']]

    if not mmd_models:
        print_step("No valid models specified for MMD calculation", "SKIP")
        return {}

    # Ensure PCA transform exists if PCA-MMD is enabled
    split_path = output_paths["base"] / "data_split_indices.npz"
    if split_path.exists():
        ensure_pca_model(config, output_paths, split_path, dry_run)

    return run_posthoc_mmd(config, output_paths, mmd_models, dry_run)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='scRNA-seq Synthetic Data Generation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python driver.py --config dataset_configs/cancer.json

  # Run specific stages
  python driver.py --config dataset_configs/cancer.json --stage train
  python driver.py --config dataset_configs/cancer.json --stage train --model cvae
  python driver.py --config dataset_configs/cancer.json --stage generate
  python driver.py --config dataset_configs/cancer.json --stage analyze

  # Post-hoc MMD calculation for all checkpoints
  python driver.py --config dataset_configs/cancer.json --stage posthoc-mmd
  python driver.py --config dataset_configs/cancer.json --stage posthoc-mmd --model cvae

  # Override checkpoint for generation
  python driver.py --config dataset_configs/cancer.json --stage generate --model cvae --checkpoint path/to/model.pt

  # Dry run (show commands without executing)
  python driver.py --config dataset_configs/cancer.json --dry-run

  # Fail fast on any error (stop pipeline immediately)
  python driver.py --config dataset_configs/cancer.json --fail-fast
        """
    )

    parser.add_argument('--config', required=True,
                        help='Path to dataset config JSON file')
    parser.add_argument('--stage', default='all', choices=STAGES,
                        help='Pipeline stage to run (default: all)')
    parser.add_argument('--model', default='all', choices=MODELS,
                        help='Model to train/generate (default: all)')
    parser.add_argument('--checkpoint', default=None,
                        help='Override checkpoint path for generation')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--fail-fast', action='store_true',
                        help='Stop pipeline immediately on any error')
    parser.add_argument('--show-plots', action='store_true',
                        help='Show plots interactively during analysis')

    args = parser.parse_args()

    # Load config
    print_header(f"Loading config: {args.config}")
    try:
        config = load_config(args.config)
        print_step(f"Dataset: {config['dataset_name']}", "OK")
        print_step(f"Output dir: {config['paths']['output_dir']}", "OK")
    except Exception as e:
        print_step(f"Failed to load config: {e}", "FAIL")
        sys.exit(1)

    # Create output directories
    output_paths = create_output_dirs(config)
    print_step("Output directories created", "OK")

    # Determine which stages to run
    stages = ['preprocess', 'train', 'generate', 'analyze'] if args.stage == 'all' else [args.stage]
    models = [args.model] if args.model != 'all' else MODELS[1:]  # Exclude 'all' placeholder

    print_header(f"Pipeline: {config['dataset_name']}")
    print(f"  Stages: {', '.join(stages)}")
    print(f"  Models: {', '.join(models)}")
    if args.dry_run:
        print(f"  Mode: DRY RUN")

    synthetic_paths = {}
    pipeline_success = True

    # Run stages
    if 'preprocess' in stages:
        success = run_preprocessing(config, output_paths, args.dry_run)
        if not success:
            pipeline_success = False
            if args.fail_fast:
                print_step("Preprocessing failed - stopping pipeline (--fail-fast)", "FAIL")
                sys.exit(1)

    if 'train' in stages:
        train_results = run_training(config, output_paths, models, args.dry_run)
        print(f"\n  Training results: {train_results}")
        if any(not v for v in train_results.values()):
            pipeline_success = False
            if args.fail_fast:
                failed = [k for k, v in train_results.items() if not v]
                print_step(f"Training failed for: {failed} - stopping pipeline (--fail-fast)", "FAIL")
                sys.exit(1)

    if 'generate' in stages:
        gen_results = run_generation(config, output_paths, models, args.checkpoint, args.dry_run)
        synthetic_paths = {k: v for k, v in gen_results.items() if v is not None}
        print(f"\n  Generation results: {gen_results}")
        if any(v is None for v in gen_results.values()):
            pipeline_success = False
            if args.fail_fast:
                failed = [k for k, v in gen_results.items() if v is None]
                print_step(f"Generation failed for: {failed} - stopping pipeline (--fail-fast)", "FAIL")
                sys.exit(1)

    if 'analyze' in stages:
        analysis_results = run_analysis(config, output_paths, synthetic_paths or None, args.dry_run, args.show_plots)
        print(f"\n  Analysis results: {analysis_results}")
        if any(not v for v in analysis_results.values()):
            pipeline_success = False

    if 'posthoc-mmd' in stages:
        mmd_results = run_posthoc_mmd_stage(config, output_paths, models, args.dry_run)
        print(f"\n  Post-hoc MMD results: {mmd_results}")
        if any(not v for v in mmd_results.values()):
            pipeline_success = False
            if args.fail_fast:
                failed = [k for k, v in mmd_results.items() if not v]
                print_step(f"Post-hoc MMD failed for: {failed} - stopping pipeline (--fail-fast)", "FAIL")
                sys.exit(1)

    print_header("Pipeline Complete")
    print(f"  Results directory: {output_paths['base']}")
    if not pipeline_success:
        print_step("Some steps failed - check logs above", "WARN")
        sys.exit(1 if args.fail_fast else 0)  # Exit with error if fail-fast mode


if __name__ == "__main__":
    main()
