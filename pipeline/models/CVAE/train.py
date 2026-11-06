#!/usr/bin/env python3
"""
CVAE Training Script for Pipeline Integration

Based on train_cvae_cancer.py - uses the model's built-in train_net() method.

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
from scipy import sparse

# Add parent directories to path for imports
script_dir = Path(__file__).parent
pipeline_dir = script_dir.parent.parent
cvae_root = pipeline_dir.parent

sys.path.insert(0, str(script_dir))  # Add models/CVAE/ for local imports
sys.path.insert(0, str(pipeline_dir))

from unbatch_vanilla_cvae_relu_qol_categorical import (
    Enhanced_Vanilla_CVAE_Categorical,
    create_dataloader_with_encoders
)
from utils.model_tracker import ModelTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train CVAE model for scRNA-seq synthetic data generation'
    )

    # Required arguments
    parser.add_argument('--data', required=True,
                        help='Path to input h5ad file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for trained models')
    parser.add_argument('--condition-cols', required=True,
                        help='Comma-separated condition column names')

    # Model architecture
    parser.add_argument('--latent-size', type=int, default=32,
                        help='Latent space dimension (default: 32)')
    parser.add_argument('--hidden-dims', default='800,400',
                        help='Comma-separated hidden layer dimensions (default: 800,400)')

    # Training parameters
    parser.add_argument('--beta', type=float, default=0.05,
                        help='KL divergence weight (default: 0.05)')
    parser.add_argument('--dropout', type=float, default=0.03,
                        help='Dropout rate (default: 0.03)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='Weight decay (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--n-epochs', type=int, default=50,
                        help='Maximum epochs (default: 50)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--overfitting-patience', type=int, default=10,
                        help='Overfitting detection patience (default: 10)')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                        help='Minimum delta for early stopping (default: 1e-4)')
    parser.add_argument('--val-split', type=float, default=0.3,
                        help='Validation split ratio (default: 0.3)')
    parser.add_argument('--checkpoint-freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')

    # Other
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--show-plots', action='store_true',
                        help='Display plots with plt.show() (default: False)')
    parser.add_argument('--split-indices', type=str, default=None,
                        help='Path to pre-computed data_split_indices.npz for stratified splits')

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

    return parser.parse_args()


def evaluate_mmd_cvae(model, val_adata, condition_cols, device, sigma, real_data, pca_model=None):
    """
    Generate synthetic data matching validation conditions and compute MMD.

    Args:
        model: Trained CVAE model
        val_adata: Validation AnnData object
        condition_cols: List of condition column names
        device: torch device
        sigma: Pre-computed kernel bandwidth
        real_data: Real validation data as numpy array (already PCA-transformed if PCA enabled)
        pca_model: Fitted PCA model for transforming synthetic data (optional)

    Returns:
        mmd_value: float
    """
    from utils.mmd import compute_mmd

    # Get condition counts from validation set
    condition_counts = val_adata.obs.groupby(condition_cols, observed=True).size()

    generated_list = []
    for conditions, count in condition_counts.items():
        if not isinstance(conditions, tuple):
            conditions = (conditions,)

        # Encode conditions using model's label encoders
        encoded = []
        for col, val in zip(condition_cols, conditions):
            if model.label_encoders.get(col) is not None:
                enc_val = model.label_encoders[col].transform([str(val)])[0]
            else:
                enc_val = float(val)
            encoded.append(enc_val)

        cond_tensor = torch.tensor([encoded], dtype=torch.float32, device=device)

        # Generate synthetic samples
        model.eval()
        with torch.no_grad():
            synthetic = model.generate(cond_tensor, count)
            synthetic = torch.clamp(synthetic, min=0).cpu().numpy()
        generated_list.append(synthetic)

    generated_data = np.vstack(generated_list)
    if pca_model is not None:
        generated_data = pca_model.transform(generated_data)
    mmd_value = compute_mmd(real_data, generated_data, sigma)
    return mmd_value


def train_cvae(args):
    """Main training function - based on train_cvae_cancer.py"""
    print("CVAE Training")
    print("=" * 40)

    # Parse condition columns
    condition_cols = [c.strip() for c in args.condition_cols.split(',')]
    hidden_dims = [int(d.strip()) for d in args.hidden_dims.split(',')]

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model tracker and create run directory
    tracker = ModelTracker(output_dir, 'cvae')
    run_dir = tracker.create_run_dir()

    print(f"Output: {run_dir}")
    print(f"Config: latent={args.latent_size}, hidden={hidden_dims}, beta={args.beta}")

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    try:
        adata = sc.read_h5ad(args.data)
        print(f"Data: {adata.n_obs} cells x {adata.n_vars} genes")

        # Convert CSC to CSR for efficient row access
        if sparse.issparse(adata.X) and adata.X.format == 'csc':
            adata.X = adata.X.tocsr()

        n_genes = adata.n_vars

        # Verify condition columns exist
        missing = [c for c in condition_cols if c not in adata.obs.columns]
        if missing:
            print(f"[ERROR] Missing columns: {missing}")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return False

    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if torch.cuda.is_available() else ""))

    model_path = run_dir / "best_model.pt"
    metrics_path = run_dir / "training_metrics.json"

    try:
        cvae = Enhanced_Vanilla_CVAE_Categorical(
            n_genes=n_genes,
            n_labels=len(condition_cols),
            latent_size=args.latent_size,
            beta=args.beta,
            dropout=args.dropout,
            lr=args.lr,
            wd=args.weight_decay,
            device=device,
            model=str(model_path),
            metrics_path=str(metrics_path),
            verbose=args.verbose,
            condition_cols=condition_cols
        )

        cvae.fit_label_encoders(adata, condition_cols)
        total_params = sum(p.numel() for p in cvae.parameters())
        print(f"Model: {total_params:,} parameters")

    except Exception as e:
        print(f"[ERROR] Model initialization failed: {e}")
        return False

    # =========================================================================
    # DATA SPLITTING AND DATALOADER CREATION
    # =========================================================================
    try:
        if not args.split_indices:
            print("[ERROR] --split-indices is required. Run via driver.py to auto-generate splits.")
            return False

        # Use pre-computed stratified split indices
        from utils.data_splitter import load_split_indices
        train_indices, val_indices = load_split_indices(args.split_indices)
        train_adata = adata[train_indices].copy()
        val_adata = adata[val_indices].copy()
        print(f"[OK] Loaded stratified split from: {args.split_indices}")
        print(f"Split: {train_adata.n_obs} train / {val_adata.n_obs} val")

        train_loader = create_dataloader_with_encoders(
            train_adata, condition_cols, cvae.label_encoders,
            batch_size=args.batch_size, shuffle=True
        )
        val_loader = create_dataloader_with_encoders(
            val_adata, condition_cols, cvae.label_encoders,
            batch_size=args.batch_size, shuffle=False
        )

        # Debug: Print 20th validation sample from loader for cross-model verification
        # if args.split_indices:
        #     print(f"[DEBUG] CVAE val_indices[19]: {val_indices[19]}")
        # sample_count = 0
        # for batch_data, batch_labels in val_loader:
        #     for i in range(batch_data.shape[0]):
        #         if sample_count == 19:
        #             val_sample = batch_data[i].numpy()
        #             print(f"[DEBUG] CVAE val_loader[19] first 5 values: {val_sample[:5]}")
        #             print(f"[DEBUG] CVAE val_loader[19] sum: {val_sample.sum():.6f}")
        #             break
        #         sample_count += 1
        #     if sample_count >= 19:
        #         break

    except Exception as e:
        print(f"[ERROR] Data splitting failed: {e}")
        return False

    # =========================================================================
    # MMD EVALUATION SETUP
    # =========================================================================
    mmd_history = []
    mmd_sigma = None
    real_val_data = None
    mmd_tracker = None

    if args.mmd_eval:
        from utils.mmd import compute_sigma_from_real_data, compute_mmd

        # Get real validation data for MMD evaluation
        if hasattr(val_adata.X, 'toarray'):
            real_val_data = val_adata.X.toarray()
        else:
            real_val_data = np.array(val_adata.X)

        # Load PCA transform if provided
        pca_model = None
        if args.pca_path:
            import joblib
            pca_model = joblib.load(args.pca_path)
            real_val_data = pca_model.transform(real_val_data)
            print(f"[MMD] PCA-MMD mode: {pca_model.n_components_} components")
            print(f"[MMD] Transformed validation data to PCA space: {real_val_data.shape}")

        # Use pre-computed sigma if available, otherwise compute from val data
        if args.sigma_path:
            from utils.mmd_early_stopping import load_sigma_heuristic, MMDEarlyStoppingTracker
            mmd_sigma = load_sigma_heuristic(args.sigma_path)
            print(f"[MMD] Loaded sigma from {args.sigma_path}: {mmd_sigma:.6f}")

            # Create tracker for early stopping
            mmd_tracker = MMDEarlyStoppingTracker(
                model_type="cvae",
                dataset_name=Path(args.data).stem,
                sigma=mmd_sigma,
                eval_epoch_interval=args.mmd_eval_freq,
                patience_mmd=args.mmd_patience,
                min_delta=args.mmd_min_delta,
                save_dir=str(run_dir)
            )

            def mmd_epoch_callback(epoch, model, history):
                """Callback with MMD early stopping."""
                if mmd_tracker.should_evaluate(epoch - 1):  # epoch is 1-indexed in callback
                    # Save checkpoint
                    ckpt_filename = f"checkpoint_epoch_{epoch}.pt"
                    ckpt_path = checkpoint_dir / ckpt_filename
                    torch.save(model.state_dict(), ckpt_path)

                    # Evaluate MMD
                    mmd_value = evaluate_mmd_cvae(
                        model, val_adata, condition_cols, device, mmd_sigma, real_val_data, pca_model
                    )

                    # Get current loss
                    current_loss = history['val_loss'][-1] if history['val_loss'] else None

                    is_best, should_stop = mmd_tracker.record(
                        epoch - 1, mmd_value, current_loss, ckpt_filename
                    )

                    if is_best:
                        torch.save(model.state_dict(), run_dir / "best_model.pt")
                        print(f"  [MMD] Saved best model at epoch {epoch}")

                    if should_stop:
                        mmd_tracker.save_history()
                        return "stop"

            epoch_callback = mmd_epoch_callback

            # Disable loss-based early stopping; MMD callback handles stopping
            args.patience = args.n_epochs
            # Disable periodic checkpoints; MMD callback handles them
            args.checkpoint_freq = 0
        else:
            mmd_sigma = compute_sigma_from_real_data(real_val_data)
            print(f"[MMD] Computed sigma from validation data: {mmd_sigma:.6f}")

            def mmd_epoch_callback(epoch, model, history):
                """Callback to evaluate MMD at specified intervals (passive, no early stopping)."""
                if epoch % args.mmd_eval_freq == 0:
                    mmd_value = evaluate_mmd_cvae(
                        model, val_adata, condition_cols, device, mmd_sigma, real_val_data, pca_model
                    )
                    mmd_history.append({'epoch': epoch, 'mmd': mmd_value})
                    print(f"[MMD] Epoch {epoch}: {mmd_value:.6f}")

            epoch_callback = mmd_epoch_callback
    else:
        epoch_callback = None

    # =========================================================================
    # TRAINING
    # =========================================================================
    print(f"\nTraining (max {args.n_epochs} epochs, patience={args.patience})...")
    start_time = time.time()

    # Create checkpoint directory
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    try:
        training_history = cvae.train_net(
            train_loader=train_loader,
            test_loader=val_loader,
            n_epochs=args.n_epochs,
            patience=args.patience,
            overfitting_patience=args.overfitting_patience,
            min_delta=args.min_delta,
            save_model=True,
            epoch_callback=epoch_callback,
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_dir=str(checkpoint_dir)
        )
        training_time = time.time() - start_time

    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # MMD EVALUATION ON BEST MODEL
    # =========================================================================
    if args.mmd_eval:
        if mmd_tracker is not None:
            # MMD early stopping was active - save history if not already saved
            if mmd_tracker.stop_reason is None:
                mmd_tracker.save_history(stop_reason_override='max_epochs_reached')
        else:
            # Legacy passive MMD mode - evaluate best model
            cvae.load_state_dict(torch.load(model_path))
            final_mmd = evaluate_mmd_cvae(
                cvae, val_adata, condition_cols, device, mmd_sigma, real_val_data, pca_model
            )
            mmd_history.append({'epoch': 'best', 'mmd': final_mmd})
            print(f"[MMD] Best model: {final_mmd:.6f}")

            # Save MMD history
            mmd_output = {'sigma': mmd_sigma, 'history': mmd_history}
            with open(run_dir / "mmd_history.json", 'w') as f:
                json.dump(mmd_output, f, indent=2)
            print(f"[OK] MMD history saved to: {run_dir / 'mmd_history.json'}")

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    best_epoch = training_history.get('best_epoch', len(training_history['train_loss']))
    best_val_loss = training_history.get('best_val_loss', training_history['val_loss'][-1] if training_history['val_loss'] else 0)

    # Save label encoders
    encoders_path = run_dir / "label_encoders.pkl"
    cvae.save_label_encoders(str(encoders_path))

    # Convert training history to epoch_losses format for tracker
    epoch_losses = [
        {"epoch": i + 1, "train_loss": train_l, "val_loss": val_l}
        for i, (train_l, val_l) in enumerate(zip(training_history['train_loss'], training_history['val_loss']))
    ]

    # Save training metrics
    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs": len(epoch_losses),
        "early_stopped": training_history.get('stopped_early', False),
        "stop_reason": training_history.get('stop_reason', None),
        "epoch_losses": epoch_losses,
        "training_time_seconds": training_time,
        "model_params": {
            "n_genes": n_genes,
            "latent_size": args.latent_size,
            "hidden_dims": hidden_dims,
            "beta": args.beta,
            "dropout": args.dropout,
            "lr": args.lr,
            "batch_size": args.batch_size
        }
    }
    tracker.save_training_metrics(run_dir, metrics)
    tracker.save_loss_curves(run_dir, epoch_losses)

    # Save run config
    run_config = {
        "data_path": args.data,
        "condition_cols": condition_cols,
        "model_params": metrics["model_params"],
        "training_params": {
            "n_epochs": args.n_epochs,
            "patience": args.patience,
            "overfitting_patience": args.overfitting_patience,
            "min_delta": args.min_delta,
            "val_split": args.val_split,
            "weight_decay": args.weight_decay,
            "random_state": args.random_state
        }
    }
    tracker.save_run_config(run_dir, {"cvae": run_config})
    tracker.update_latest_model(run_dir, best_epoch, best_val_loss, {"cvae": run_config})

    # Create training plot
    try:
        import matplotlib.pyplot as plt
        plot_path = run_dir / "training_curves.png"
        cvae.plot_enhanced_training_history(history=training_history, save_path=str(plot_path))
        plt.close('all')  # Close figures (Agg backend set at top if not --show-plots)
    except Exception as e:
        print(f"[WARN] Could not save training plot: {e}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 40)
    print(f"Training complete in {training_time/60:.1f} min")
    print(f"Best: epoch {best_epoch}, val_loss={best_val_loss:.6f}")
    print(f"Model: {model_path}")

    return True


def main():
    args = parse_args()
    success = train_cvae(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
