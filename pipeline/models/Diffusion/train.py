#!/usr/bin/env python3
"""
Diffusion Model Training Script for Pipeline Integration

This script wraps the existing Diffusion training functionality with:
- Command-line argument parsing
- Integration with ModelTracker for checkpoint management
- Standardized output format

Usage:
    python train.py --data path/to/data.h5ad --output-dir path/to/output --condition-cols col1,col2
"""

import sys

# Set matplotlib backend FIRST before any other imports
if '--show-plots' not in sys.argv:
    import matplotlib
    matplotlib.use('Agg')

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import scanpy as sc

# Add directories to path
script_dir = Path(__file__).parent
pipeline_dir = script_dir.parent.parent
cvae_root = pipeline_dir.parent

sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(cvae_root))
sys.path.insert(0, str(pipeline_dir))

from diffusion_model_cfg import ClassifierFreeDiffusionModel
from diffusion_training_utils import DiffusionTrainer, create_data_loaders
from vae_diffusion import DiffusionIntegrationVAE
from utils.model_tracker import ModelTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Diffusion model for scRNA-seq synthetic data generation'
    )

    # Required arguments
    parser.add_argument('--data', required=True,
                        help='Path to input h5ad file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for trained models')
    parser.add_argument('--condition-cols', required=True,
                        help='Comma-separated condition column names')
    parser.add_argument('--condition-types', default=None,
                        help='Comma-separated condition types (categorical/continuous) matching condition-cols order. If not provided, types are inferred from data.')

    # VAE architecture parameters
    parser.add_argument('--vae-latent-size', type=int, default=128,
                        help='VAE latent space dimension (default: 128)')
    parser.add_argument('--vae-hidden-dims', default='1024,1024',
                        help='VAE hidden layer dimensions (default: 1024,1024)')
    parser.add_argument('--vae-model-path', default=None,
                        help='Path to pre-trained VAE model (skip VAE training if provided)')

    # VAE training parameters
    parser.add_argument('--vae-beta', type=float, default=0.001,
                        help='VAE KL divergence weight (default: 0.001)')
    parser.add_argument('--vae-dropout', type=float, default=0.1,
                        help='VAE dropout rate (default: 0.1)')
    parser.add_argument('--vae-lr', type=float, default=0.001,
                        help='VAE learning rate (default: 0.001)')
    parser.add_argument('--vae-weight-decay', type=float, default=0.001,
                        help='VAE weight decay (default: 0.001)')
    parser.add_argument('--vae-n-epochs', type=int, default=200,
                        help='VAE maximum epochs (default: 200)')
    parser.add_argument('--vae-patience', type=int, default=20,
                        help='VAE early stopping patience (default: 20)')
    parser.add_argument('--vae-overfitting-patience', type=int, default=10,
                        help='VAE overfitting detection patience (default: 10)')
    parser.add_argument('--vae-min-delta', type=float, default=1e-4,
                        help='VAE minimum improvement threshold (default: 1e-4)')

    # Diffusion model parameters
    parser.add_argument('--latent-size', type=int, default=128,
                        help='Diffusion latent dimension (default: 128)')
    parser.add_argument('--hidden-dims', default='512,512,256,128',
                        help='Diffusion hidden layer dimensions (default: 512,512,256,128)')
    parser.add_argument('--num-timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps (default: 1000)')
    parser.add_argument('--beta-schedule', default='cosine',
                        help='Beta schedule (default: cosine)')
    parser.add_argument('--classifier-free-prob', type=float, default=0.1,
                        help='Classifier-free guidance probability (default: 0.1)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')

    # Diffusion training parameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Diffusion learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Diffusion weight decay (default: 1e-5)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--n-epochs', type=int, default=50,
                        help='Diffusion maximum epochs (default: 50)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Diffusion early stopping patience (default: 15)')
    parser.add_argument('--checkpoint-freq', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Training split ratio (default: 0.8)')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    parser.add_argument('--split-indices', type=str, default=None,
                        help='Path to pre-computed data_split_indices.npz (stratified split)')

    # Training mode options
    parser.add_argument('--vae-only', action='store_true',
                        help='Only train VAE (Phase 1), skip diffusion training')

    # MMD evaluation
    parser.add_argument('--mmd-eval', action='store_true',
                        help='Enable MMD evaluation during training')
    parser.add_argument('--mmd-eval-freq', type=int, default=5,
                        help='MMD evaluation frequency in epochs (default: 5)')
    parser.add_argument('--sigma-path', type=str, default=None,
                        help='Path to pre-computed sigma_heuristic.npz')
    parser.add_argument('--mmd-patience', type=int, default=3,
                        help='MMD early stopping patience (default: 3)')
    parser.add_argument('--mmd-min-delta', type=float, default=0.001,
                        help='Minimum MMD improvement threshold (default: 0.001)')
    parser.add_argument('--pca-path', type=str, default=None,
                        help='Path to fitted PCA transform (pca_mmd_transform.pkl) for PCA-space MMD')

    # Other
    parser.add_argument('--show-plots', action='store_true',
                        help='Display plots interactively (default: save only)')

    return parser.parse_args()


def parse_condition_types(condition_cols, condition_types_str):
    """
    Parse condition types from command line argument.

    Args:
        condition_cols: List of condition column names
        condition_types_str: Comma-separated types string (e.g., "categorical,categorical,continuous")

    Returns dict like: {'col1': 'categorical', 'col2': 'continuous'}
    """
    types = [t.strip() for t in condition_types_str.split(',')]
    if len(types) != len(condition_cols):
        raise ValueError(f"Number of condition types ({len(types)}) must match number of columns ({len(condition_cols)})")

    condition_config = {}
    for col, ctype in zip(condition_cols, types):
        if ctype not in ['categorical', 'continuous']:
            raise ValueError(f"Invalid condition type '{ctype}' for column '{col}'. Must be 'categorical' or 'continuous'.")
        condition_config[col] = ctype

    return condition_config


def infer_condition_types(adata, condition_cols):
    """
    Infer condition types (categorical/continuous) from data.

    Returns dict like: {'col1': 'categorical', 'col2': 'continuous'}
    """
    condition_config = {}

    for col in condition_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in data")

        values = adata.obs[col]

        # Check if numeric and has many unique values -> continuous
        if values.dtype in ['float64', 'float32', 'int64', 'int32']:
            unique_vals = values.nunique()
            if unique_vals > 10:  # Likely continuous
                condition_config[col] = 'continuous'
            else:  # Few unique values -> categorical
                condition_config[col] = 'categorical'
        else:
            condition_config[col] = 'categorical'

    return condition_config


def train_diffusion(args):
    """Main training function."""
    print("=" * 60)
    print("Diffusion Model Training Pipeline")
    print("=" * 60)

    # Parse arguments
    condition_cols = [c.strip() for c in args.condition_cols.split(',')]
    vae_hidden_dims = [int(d.strip()) for d in args.vae_hidden_dims.split(',')]
    diff_hidden_dims = [int(d.strip()) for d in args.hidden_dims.split(',')]

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model tracker and create run directory
    tracker = ModelTracker(output_dir, 'diffusion')
    run_dir = tracker.create_run_dir()

    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print(f"  Output: {run_dir}")
    print(f"  Conditions: {condition_cols}")
    print(f"  VAE latent: {args.vae_latent_size}")
    print(f"  Diff latent: {args.latent_size}")
    print(f"  Epochs: {args.n_epochs}")
    print(f"  Batch size: {args.batch_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    print("\n" + "-" * 40)
    print("Loading data...")

    try:
        adata = sc.read_h5ad(args.data)
        print(f"[OK] Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

        n_genes = adata.n_vars

        # Verify condition columns
        missing = [c for c in condition_cols if c not in adata.obs.columns]
        if missing:
            print(f"[ERROR] Missing columns: {missing}")
            return False

        # Get condition types from args or infer from data
        if args.condition_types:
            condition_config = parse_condition_types(condition_cols, args.condition_types)
            print(f"[OK] Condition config (from args): {condition_config}")
        else:
            condition_config = infer_condition_types(adata, condition_cols)
            print(f"[OK] Condition config (inferred): {condition_config}")

    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # VAE SETUP AND TRAINING
    # =========================================================================
    print("\n" + "-" * 40)
    print("Setting up VAE...")

    vae_path = run_dir / "vae_model.pt"
    vae_metrics_path = run_dir / "vae_training_metrics.json"

    try:
        vae_model = DiffusionIntegrationVAE(
            n_genes=n_genes,
            latent_size=args.vae_latent_size,
            hidden_dims=vae_hidden_dims,
            beta=args.vae_beta,
            dropout=args.vae_dropout,
            lr=args.vae_lr,
            wd=args.vae_weight_decay,
            model=str(vae_path),
            metrics_path=str(vae_metrics_path),
            verbose=True,
            auto_load=False
        )

        # Print VAE architecture verification
        print("\n" + "-" * 40)
        print("VAE Architecture Verification:")
        print(f"  Input genes: {n_genes}")
        print(f"  Hidden dims: {vae_hidden_dims}")
        print(f"  Latent size: {args.vae_latent_size}")
        print(f"  Beta (KL weight): {args.vae_beta}")
        print(f"  Dropout: {args.vae_dropout}")
        print(f"  Learning rate: {args.vae_lr}")
        print(f"  Weight decay: {args.vae_weight_decay}")
        print(f"  Max epochs: {args.vae_n_epochs}")
        print(f"  Patience: {args.vae_patience}")
        print(f"  Overfitting patience: {args.vae_overfitting_patience}")

        # Load pre-trained VAE if provided, otherwise train from scratch
        if args.vae_model_path and os.path.exists(args.vae_model_path):
            print(f"\n[INFO] Loading pre-trained VAE from: {args.vae_model_path}")
            vae_model.load_state_dict(torch.load(args.vae_model_path, map_location=device))
            print(f"[OK] Pre-trained VAE loaded successfully")
        else:
            # Train VAE from scratch
            print("\n" + "=" * 60)
            print("PHASE 1: VAE TRAINING")
            print("=" * 60)
            print("[INFO] Training VAE from scratch...")

            # Import VAE dataset creation utility
            from vae_diffusion import create_pytorch_datasets

            # Create VAE-specific data loaders (without conditions)
            vae_train_loader, vae_val_loader, vae_test_loader = create_pytorch_datasets(
                adata.X,
                batch_size=args.batch_size,
                validation_split=args.val_split,
                test_split=1.0 - args.train_split - args.val_split,
                num_workers=0,
                split_indices=args.split_indices
            )

            print(f"[OK] VAE data loaders created")
            print(f"     Train batches: {len(vae_train_loader)}")
            print(f"     Val batches: {len(vae_val_loader)}")
            print(f"     Test batches: {len(vae_test_loader)}")

            # Train VAE
            vae_start_time = time.time()
            vae_history = vae_model.train_net(
                train_loader=vae_train_loader,
                validation_loader=vae_val_loader,
                n_epochs=args.vae_n_epochs,
                patience=args.vae_patience,
                overfitting_patience=args.vae_overfitting_patience,
                min_delta=args.vae_min_delta,
                save_model=True
            )
            vae_training_time = time.time() - vae_start_time

            # Evaluate VAE on test set
            print("\n" + "-" * 40)
            print("Evaluating VAE on test set...")
            vae_test_results = vae_model.test_model(vae_test_loader)
            print(f"[OK] VAE Test Results:")
            print(f"     Total Loss: {vae_test_results['total_loss']:.6f}")
            print(f"     Reconstruction Loss: {vae_test_results['recon_loss']:.6f}")
            print(f"     KL Loss: {vae_test_results['kl_loss']:.6f}")

            # Save VAE training plot
            vae_plot_path = run_dir / "vae_training_curves.png"
            vae_model.plot_comprehensive_training_history(
                vae_history,
                save_path=str(vae_plot_path)
            )
            import matplotlib.pyplot as plt
            plt.close('all')  # Prevent display (Agg backend set at top)
            print(f"[OK] VAE training curves saved: {vae_plot_path}")

            print(f"\n[OK] VAE training completed in {vae_training_time/60:.2f} minutes")
            print(f"     Best epoch: {vae_history.get('best_epoch', 'N/A')}")
            print(f"     Best val loss: {vae_history.get('best_val_loss', 'N/A'):.6f}")

        print(f"[OK] VAE initialized and ready")

    except Exception as e:
        print(f"[ERROR] VAE setup/training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # VAE-ONLY MODE: Save and exit after VAE training
    # =========================================================================
    if args.vae_only:
        print("\n" + "=" * 60)
        print("VAE-ONLY MODE: Saving VAE and exiting")
        print("=" * 60)

        # Save VAE model
        torch.save(vae_model.state_dict(), vae_path)
        print(f"[OK] VAE saved: {vae_path}")

        # Save VAE config for generate.py to use
        vae_config = {
            "n_genes": n_genes,
            "latent_size": args.vae_latent_size,
            "hidden_dims": vae_hidden_dims,
            "beta": args.vae_beta,
            "dropout": args.vae_dropout
        }
        vae_config_path = run_dir / "vae_config.json"
        with open(vae_config_path, 'w') as f:
            json.dump(vae_config, f, indent=2)
        print(f"[OK] VAE config saved: {vae_config_path}")

        # Get VAE training info
        vae_best_epoch = vae_history.get('best_epoch', args.vae_n_epochs) if 'vae_history' in dir() else args.vae_n_epochs
        vae_best_val_loss = vae_history.get('best_val_loss', 0.0) if 'vae_history' in dir() else 0.0

        # Save VAE-only training metrics
        vae_metrics = {
            "mode": "vae_only",
            "vae_best_epoch": vae_best_epoch,
            "vae_best_val_loss": vae_best_val_loss,
            "vae_model_path": str(vae_path),
            "vae_params": {
                "latent_size": args.vae_latent_size,
                "hidden_dims": vae_hidden_dims,
                "beta": args.vae_beta,
                "dropout": args.vae_dropout,
                "lr": args.vae_lr,
                "n_epochs": args.vae_n_epochs
            }
        }
        tracker.save_training_metrics(run_dir, vae_metrics)

        # Update latest_model.json with VAE info
        tracker.update_latest_model(
            run_dir, vae_best_epoch, vae_best_val_loss,
            config={
                "diffusion": {
                    "mode": "vae_only",
                    "vae_params": vae_metrics["vae_params"]
                }
            },
            additional_paths={
                "vae_model_path": f"{run_dir.name}/vae_model.pt"
            }
        )

        print(f"\n[OK] VAE-only training complete")
        print(f"Run directory: {run_dir}")
        print(f"VAE model: {vae_path}")
        print(f"\nTo continue with diffusion training, run again without --vae-only")
        print(f"and use --vae-model-path {vae_path}")

        return True

    # =========================================================================
    # PHASE 2: DIFFUSION MODEL TRAINING
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: DIFFUSION MODEL TRAINING")
    print("=" * 60)

    # =========================================================================
    # CREATE DATA LOADERS (with VAE encoding)
    # =========================================================================
    print("\n" + "-" * 40)
    print("Creating diffusion data loaders (encoding to VAE latent space)...")

    try:
        train_loader, val_loader, test_loader, processor, full_condition_config = create_data_loaders(
            adata,
            condition_config,
            vae_model=vae_model,
            batch_size=args.batch_size,
            train_split=args.train_split,
            val_split=args.val_split,
            split_indices=args.split_indices
        )

        print(f"[OK] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # Debug: Print 20th validation sample from loader for cross-model verification
        # if args.split_indices:
        #     from utils.data_splitter import load_split_indices
        #     _train_idx, _val_idx = load_split_indices(args.split_indices)
        #     print(f"[DEBUG] Diffusion val_indices[19]: {_val_idx[19]}")
        # sample_count = 0
        # for batch_data, batch_conds in val_loader:
        #     for i in range(batch_data.shape[0]):
        #         if sample_count == 19:
        #             val_sample = batch_data[i].numpy()
        #             print(f"[DEBUG] Diffusion val_loader[19] first 5 values: {val_sample[:5].tolist()}")
        #             print(f"[DEBUG] Diffusion val_loader[19] sum: {val_sample.sum():.6f} (latent space)")
        #             break
        #         sample_count += 1
        #     if sample_count >= 19:
        #         break

        # Save condition processor info
        processor_info = {
            'condition_config': condition_config,
            'full_condition_config': full_condition_config
        }
        with open(run_dir / "condition_config.json", 'w') as f:
            json.dump(processor_info, f, indent=2, default=str)

    except Exception as e:
        print(f"[ERROR] Data loader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # DIFFUSION MODEL SETUP
    # =========================================================================
    print("\n" + "-" * 40)
    print("Creating diffusion model...")

    try:
        model = ClassifierFreeDiffusionModel(
            latent_dim=args.latent_size,
            condition_config=full_condition_config,
            hidden_dims=diff_hidden_dims,
            classifier_free_prob=args.classifier_free_prob,
            dropout=args.dropout
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"[OK] Diffusion model: {total_params:,} parameters")

        # Print model architecture verification
        print("\n" + "-" * 40)
        print("Model Architecture Verification:")
        print(f"  VAE latent size: {args.vae_latent_size}")
        print(f"  VAE hidden dims: {vae_hidden_dims}")
        print(f"  Diffusion latent size: {args.latent_size}")
        print(f"  Diffusion hidden dims: {diff_hidden_dims}")
        print(f"  Num timesteps: {args.num_timesteps}")
        print(f"  Beta schedule: {args.beta_schedule}")
        print(f"  Classifier-free prob: {args.classifier_free_prob}")
        print(f"  Dropout: {args.dropout}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Condition config: {condition_config}")

    except Exception as e:
        print(f"[ERROR] Diffusion model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # TRAINING
    # =========================================================================
    print("\n" + "-" * 40)
    print("Setting up trainer...")

    try:
        trainer = DiffusionTrainer(
            model=model,
            condition_processor=processor,
            vae_model=vae_model,
            num_timesteps=args.num_timesteps,
            beta_schedule=args.beta_schedule
        )

        # Custom checkpoint directory within run_dir
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        print(f"[OK] Trainer initialized")

    except Exception as e:
        print(f"[ERROR] Trainer setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "-" * 40)
    print("Starting training...")

    # Prepare real validation gene expression data for MMD evaluation
    real_val_gene_data = None
    if args.mmd_eval:
        # Get validation indices
        if args.split_indices:
            from utils.data_splitter import load_split_indices
            _, val_idx = load_split_indices(args.split_indices)
            val_adata = adata[val_idx]
        else:
            # Random split - need to recreate
            n_samples = adata.n_obs
            n_train = int(n_samples * args.train_split)
            n_val = int(n_samples * args.val_split)
            torch.manual_seed(42)
            indices = torch.randperm(n_samples).numpy()
            val_idx = indices[n_train:n_train + n_val]
            val_adata = adata[val_idx]

        # Get real gene expression from validation set
        if hasattr(val_adata.X, 'toarray'):
            real_val_gene_data = val_adata.X.toarray()
        else:
            real_val_gene_data = np.array(val_adata.X)
        print(f"[MMD] Prepared validation gene expression data: {real_val_gene_data.shape}")

    # Load PCA transform if provided
    pca_model = None
    if args.mmd_eval and args.pca_path:
        import joblib
        pca_model = joblib.load(args.pca_path)
        real_val_gene_data = pca_model.transform(real_val_gene_data)
        print(f"[MMD] PCA-MMD mode: {pca_model.n_components_} components")
        print(f"[MMD] Transformed validation data to PCA space: {real_val_gene_data.shape}")

    # Create MMD tracker if sigma path provided
    mmd_tracker = None
    if args.mmd_eval and args.sigma_path:
        from utils.mmd_early_stopping import load_sigma_heuristic, MMDEarlyStoppingTracker
        mmd_sigma_loaded = load_sigma_heuristic(args.sigma_path)
        print(f"[MMD] Loaded sigma from {args.sigma_path}: {mmd_sigma_loaded:.6f}")

        mmd_tracker = MMDEarlyStoppingTracker(
            model_type="diffusion",
            dataset_name=Path(args.data).stem,
            sigma=mmd_sigma_loaded,
            eval_epoch_interval=args.mmd_eval_freq,
            patience_mmd=args.mmd_patience,
            min_delta=args.mmd_min_delta,
            save_dir=str(run_dir)
        )

    start_time = time.time()

    try:
        # Train the model
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.n_epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            save_every=args.checkpoint_freq,
            save_dir=str(checkpoint_dir),
            mmd_eval=args.mmd_eval,
            mmd_eval_freq=args.mmd_eval_freq,
            real_val_gene_data=real_val_gene_data,
            mmd_tracker=mmd_tracker,
            pca_model=pca_model
        )

        training_time = time.time() - start_time
        print(f"\n[OK] Training completed in {training_time/60:.2f} minutes")

        # Copy MMD history to run_dir if it exists
        if args.mmd_eval:
            mmd_history_src = checkpoint_dir / "mmd_history.json"
            if mmd_history_src.exists():
                import shutil
                shutil.copy2(mmd_history_src, run_dir / "mmd_history.json")
                print(f"[OK] MMD history copied to: {run_dir / 'mmd_history.json'}")

    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    print("\n" + "-" * 40)
    print("Saving outputs...")

    # Note: VAE is already saved by train_net() during VAE training phase
    # Only save here if we loaded a pre-trained VAE (to copy it to this run directory)
    if args.vae_model_path and os.path.exists(args.vae_model_path):
        import shutil
        shutil.copy2(args.vae_model_path, vae_path)
        print(f"[OK] Pre-trained VAE copied to: {vae_path}")
    elif vae_path.exists():
        print(f"[OK] VAE already saved at: {vae_path}")

    # Save VAE config for generate.py to use
    vae_config = {
        "n_genes": n_genes,
        "latent_size": args.vae_latent_size,
        "hidden_dims": vae_hidden_dims,
        "beta": args.vae_beta,
        "dropout": args.vae_dropout
    }
    vae_config_path = run_dir / "vae_config.json"
    with open(vae_config_path, 'w') as f:
        json.dump(vae_config, f, indent=2)
    print(f"[OK] VAE config saved: {vae_config_path}")

    # Find best checkpoint (trainer should have saved this)
    # The DiffusionTrainer saves:
    # - Best models: best_model_epoch_X_model.pth (with config.json and info.json)
    # - Regular checkpoints: checkpoint_epoch_X.pth

    # Look for best model .pth files first
    best_model_files = list(checkpoint_dir.glob("best_model_epoch_*_model.pth"))
    regular_checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))

    best_model_path = None
    best_epoch = args.n_epochs

    if best_model_files:
        # Find the highest epoch best model
        best_checkpoint = max(best_model_files, key=lambda x: x.stat().st_mtime)
        best_model_path = run_dir / "best_model.pt"

        # Copy to best_model.pt
        import shutil
        shutil.copy2(best_checkpoint, best_model_path)
        print(f"[OK] Best model saved: {best_model_path}")

        # Also copy associated config files
        checkpoint_stem = best_checkpoint.stem.replace('_model', '')
        for suffix in ['_config.json', '_info.json']:
            src = checkpoint_dir / f"{checkpoint_stem}{suffix}"
            if src.exists():
                dst = run_dir / f"best_model{suffix}"
                shutil.copy2(src, dst)

        # Extract epoch number
        try:
            epoch_str = checkpoint_stem.split('_')[-1]
            best_epoch = int(epoch_str)
        except:
            pass

    elif regular_checkpoints:
        # Fall back to regular checkpoints
        best_checkpoint = max(regular_checkpoints, key=lambda x: x.stat().st_mtime)
        best_model_path = run_dir / "best_model.pt"

        import shutil
        shutil.copy2(best_checkpoint, best_model_path)
        print(f"[OK] Best model saved (from regular checkpoint): {best_model_path}")

        try:
            epoch_str = best_checkpoint.stem.split('_')[-1]
            best_epoch = int(epoch_str)
        except:
            pass
    else:
        print(f"[WARN] No checkpoint files found in {checkpoint_dir}")

    # Create epoch losses list (simplified - trainer doesn't return detailed history)
    epoch_losses = [
        {"epoch": i + 1, "train_loss": 0.0, "val_loss": 0.0, "checkpoint": None}
        for i in range(best_epoch)
    ]

    # Get VAE training info if available
    vae_best_epoch = vae_history.get('best_epoch', args.vae_n_epochs) if 'vae_history' in dir() else args.vae_n_epochs
    vae_best_val_loss = vae_history.get('best_val_loss', 0.0) if 'vae_history' in dir() else 0.0

    # Save training metrics
    metrics = {
        "mode": "full",  # Full diffusion training (VAE + Diffusion)
        "best_epoch": best_epoch,
        "best_val_loss": 0.0,  # Would need to track this from trainer
        "total_epochs": args.n_epochs,
        "early_stopped": best_epoch < args.n_epochs,
        "best_checkpoint": str(best_model_path) if best_model_path else None,
        "epoch_losses": epoch_losses,
        "training_time_seconds": training_time,
        # VAE info
        "vae_model_path": str(vae_path),
        "vae_best_epoch": vae_best_epoch,
        "vae_best_val_loss": vae_best_val_loss,
        "vae_params": {
            "latent_size": args.vae_latent_size,
            "hidden_dims": vae_hidden_dims,
            "beta": args.vae_beta,
            "dropout": args.vae_dropout,
            "lr": args.vae_lr,
            "n_epochs": args.vae_n_epochs
        },
        # Diffusion model params
        "model_params": {
            "latent_size": args.latent_size,
            "hidden_dims": diff_hidden_dims,
            "num_timesteps": args.num_timesteps,
            "beta_schedule": args.beta_schedule,
            "classifier_free_prob": args.classifier_free_prob,
            "dropout": args.dropout,
            "lr": args.lr,
            "batch_size": args.batch_size
        }
    }
    tracker.save_training_metrics(run_dir, metrics)
    print(f"[OK] Training metrics saved")

    # Save run config
    run_config = {
        "data_path": args.data,
        "condition_cols": condition_cols,
        "condition_config": condition_config,
        "vae_model_path": str(vae_path),
        "vae_params": metrics["vae_params"],
        "model_params": metrics["model_params"]
    }
    tracker.save_run_config(run_dir, {"diffusion": run_config})
    print(f"[OK] Run config saved")

    # Update latest_model.json with both VAE and diffusion info
    tracker.update_latest_model(
        run_dir, best_epoch, 0.0,
        config={"diffusion": run_config},
        additional_paths={
            "vae_model_path": f"{run_dir.name}/vae_model.pt"
        }
    )
    print(f"[OK] Updated latest_model.json")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"VAE model: {vae_path}")
    print(f"Best diffusion model: {best_model_path}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Training time: {training_time/60:.2f} minutes")

    return True


def main():
    args = parse_args()
    success = train_diffusion(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
