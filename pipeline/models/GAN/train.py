#!/usr/bin/env python3
"""
GAN Training Script for Pipeline Integration

This script wraps the Multi-Conditional scGAN training with:
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
import numpy as np
import pandas as pd
import tensorflow as tf
import scanpy as sc
from datetime import datetime
from pathlib import Path

# Add directories to path
script_dir = Path(__file__).parent
pipeline_dir = script_dir.parent.parent
cvae_root = pipeline_dir.parent

sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(pipeline_dir))

# Local GAN imports - use explicit submodule imports to avoid utils conflict
import importlib.util

# Load ConditionProcessorGeneric from GAN's local utils
spec = importlib.util.spec_from_file_location(
    "gan_utils",
    script_dir / "utils" / "condition_processor_generic.py"
)
gan_utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gan_utils_module)
ConditionProcessorGeneric = gan_utils_module.ConditionProcessorGeneric

# Load models from local models.py
from models import build_generator, build_critic, gradient_penalty

# Load ModelTracker from pipeline utils
spec_tracker = importlib.util.spec_from_file_location(
    "model_tracker",
    pipeline_dir / "utils" / "model_tracker.py"
)
tracker_module = importlib.util.module_from_spec(spec_tracker)
spec_tracker.loader.exec_module(tracker_module)
ModelTracker = tracker_module.ModelTracker


def evaluate_mmd_gan(generator, valid_data, condition_processor, latent_dim, sigma, real_data,
                     pca_model=None):
    """
    Generate synthetic data matching validation conditions and compute MMD.

    Args:
        generator: Trained generator model
        valid_data: Validation AnnData object
        condition_processor: ConditionProcessorGeneric instance
        latent_dim: Latent dimension for noise
        sigma: Pre-computed kernel bandwidth
        real_data: Real validation data as numpy array (already PCA-transformed if PCA enabled)
        pca_model: Fitted PCA model for transforming synthetic data (optional)

    Returns:
        mmd_value: float
    """
    from utils.mmd import compute_mmd

    # Encode conditions from validation set
    conditions = condition_processor.encode_conditions(valid_data.obs)

    # Generate synthetic data matching validation conditions
    noise = tf.random.normal((len(valid_data), latent_dim))
    generated = generator((noise, conditions), training=False)
    generated = tf.maximum(generated, 0).numpy()  # Clamp to non-negative

    if pca_model is not None:
        generated = pca_model.transform(generated)
    mmd_value = compute_mmd(real_data, generated, sigma)
    return mmd_value


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train GAN model for scRNA-seq synthetic data generation'
    )

    # Required arguments
    parser.add_argument('--data', required=True,
                        help='Path to input h5ad file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for trained models')
    parser.add_argument('--condition-cols', required=True,
                        help='Comma-separated condition column names')

    # Model architecture
    parser.add_argument('--latent-dim', type=int, default=128,
                        help='Latent dimension (default: 128)')
    parser.add_argument('--gen-layers', default='256,512,1024',
                        help='Generator hidden layers (default: 256,512,1024)')
    parser.add_argument('--critic-layers', default='1024,512,256',
                        help='Critic hidden layers (default: 1024,512,256)')
    parser.add_argument('--lambda-gp', type=float, default=10.0,
                        help='Gradient penalty lambda (default: 10.0)')

    # Training parameters
    parser.add_argument('--lr-g', type=float, default=1e-4,
                        help='Generator learning rate (default: 1e-4)')
    parser.add_argument('--lr-c', type=float, default=1e-4,
                        help='Critic learning rate (default: 1e-4)')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Adam beta1 (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='Adam beta2 (default: 0.9)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--max-steps', type=int, default=50000,
                        help='Maximum training steps (default: 50000)')
    parser.add_argument('--critic-iters', type=int, default=5,
                        help='Critic iterations per generator iteration (default: 5)')
    parser.add_argument('--patience', type=int, default=10,
                        help='(Unused) Kept for backward compatibility. GAN trains to max_steps.')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Training split ratio (default: 0.8)')
    parser.add_argument('--save-freq', type=int, default=500,
                        help='Save checkpoint every N steps (default: 500)')
    parser.add_argument('--log-freq', type=int, default=100,
                        help='Log every N steps (default: 100)')
    parser.add_argument('--validation-freq', type=int, default=1,
                        help='Run validation every N checkpoints (default: 1)')
    parser.add_argument('--gp-target', type=float, default=0.6,
                        help='Target gradient penalty value for adaptive GP (default: 0.6)')
    parser.add_argument('--clip-gradients', action='store_true',
                        help='Enable gradient clipping')
    parser.add_argument('--gradient-clip-norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (default: 1.0)')
    parser.add_argument('--split-indices', type=str, default=None,
                        help='Path to pre-computed data_split_indices.npz (stratified split)')

    # MMD evaluation
    parser.add_argument('--mmd-eval', action='store_true',
                        help='Enable MMD evaluation at checkpoints')
    parser.add_argument('--mmd-eval-freq', type=int, default=1,
                        help='MMD evaluation frequency (every N checkpoints, default: 1)')
    parser.add_argument('--sigma-path', type=str, default=None,
                        help='Path to pre-computed sigma_heuristic.npz')
    parser.add_argument('--mmd-patience', type=int, default=3,
                        help='MMD early stopping patience (default: 3)')
    parser.add_argument('--mmd-min-delta', type=float, default=0.001,
                        help='Minimum MMD improvement threshold (default: 0.001)')
    parser.add_argument('--mmd-n-epochs', type=int, default=0,
                        help='Total epochs for epoch-based MMD tracking (0=use steps)')
    parser.add_argument('--save-freq-unit', type=str, default='steps',
                        choices=['steps', 'epochs'],
                        help='Unit for save-freq: steps or epochs (default: steps)')
    parser.add_argument('--pca-path', type=str, default=None,
                        help='Path to fitted PCA transform (pca_mmd_transform.pkl) for PCA-space MMD')

    return parser.parse_args()


def train_gan(args):
    """Main training function."""
    print("=" * 60)
    print("GAN Training Pipeline")
    print("=" * 60)

    # Parse arguments
    condition_cols = [c.strip() for c in args.condition_cols.split(',')]
    gen_layers = [int(d.strip()) for d in args.gen_layers.split(',')]
    critic_layers = [int(d.strip()) for d in args.critic_layers.split(',')]

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model tracker and create run directory
    tracker = ModelTracker(output_dir, 'gan')
    run_dir = tracker.create_run_dir()

    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print(f"  Output: {run_dir}")
    print(f"  Conditions: {condition_cols}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Batch size: {args.batch_size}")

    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  GPU: {len(gpus)} device(s) available")
    else:
        print("  GPU: Not available, using CPU")

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    print("\n" + "-" * 40)
    print("Loading data...")

    try:
        adata = sc.read_h5ad(args.data)
        print(f"[OK] Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

        n_genes = adata.n_vars

        # Verify condition columns exist
        missing = [c for c in condition_cols if c not in adata.obs.columns]
        if missing:
            print(f"[ERROR] Missing columns: {missing}")
            print(f"Available: {list(adata.obs.columns)}")
            return False

        # Initialize condition processor
        condition_processor = ConditionProcessorGeneric(condition_cols)
        condition_processor.fit(adata)

        # Save condition processor config
        cond_config_path = run_dir / "condition_config.json"
        condition_processor.save_config(str(cond_config_path))

    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # DATA SPLITTING AND LOADERS
    # =========================================================================
    print("\n" + "-" * 40)
    print("Creating data splits...")

    n_total = adata.n_obs

    if args.split_indices:
        # Load pre-computed stratified split
        from utils.data_splitter import load_split_indices
        train_indices, valid_indices = load_split_indices(args.split_indices)
        print(f"[OK] Loaded stratified split from: {args.split_indices}")
    else:
        # Fallback to random split
        n_train = int(n_total * args.train_split)
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        valid_indices = indices[n_train:]
        print("[INFO] Using random split (no --split-indices provided)")

    # Save indices for reproducibility (copy to run directory)
    np.savez(run_dir / "data_split_indices.npz",
             train_indices=train_indices, valid_indices=valid_indices)

    train_data = adata[train_indices].copy()
    valid_data = adata[valid_indices].copy()

    print(f"[OK] Train: {len(train_indices)} cells, Val: {len(valid_indices)} cells")

    # Create TensorFlow datasets
    def create_tf_dataset(adata_subset, shuffle=True):
        if hasattr(adata_subset.X, 'toarray'):
            expression = adata_subset.X.toarray().astype(np.float32)
        else:
            expression = adata_subset.X.astype(np.float32)

        conditions = condition_processor.encode_conditions(adata_subset.obs)

        dataset = tf.data.Dataset.from_tensor_slices((expression, conditions))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(args.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train_dataset = create_tf_dataset(train_data, shuffle=True)
    valid_dataset = create_tf_dataset(valid_data, shuffle=False)

    print(f"[OK] Train batches: {len(list(train_dataset))}")

    # Debug: Print 20th validation sample from loader for cross-model verification
    # if args.split_indices:
    #     print(f"[DEBUG] GAN val_indices[19]: {valid_indices[19]}")
    # sample_count = 0
    # debug_done = False
    # for batch_data, batch_conds in valid_dataset:
    #     for i in range(batch_data.shape[0]):
    #         if sample_count == 19:
    #             val_sample = batch_data[i].numpy()
    #             print(f"[DEBUG] GAN val_loader[19] first 5 values: {val_sample[:5]}")
    #             print(f"[DEBUG] GAN val_loader[19] sum: {val_sample.sum():.6f}")
    #             debug_done = True
    #             break
    #         sample_count += 1
    #     if debug_done:
    #         break

    # =========================================================================
    # MODEL BUILDING
    # =========================================================================
    print("\n" + "-" * 40)
    print("Building models...")

    condition_dim = condition_processor.total_condition_dim

    generator = build_generator(
        latent_dim=args.latent_dim,
        condition_dim=condition_dim,
        output_dim=n_genes,
        hidden_layers=gen_layers
    )

    critic = build_critic(
        input_dim=n_genes,
        condition_dim=condition_dim,
        hidden_layers=critic_layers
    )

    g_optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.lr_g,
        beta_1=args.beta1,
        beta_2=args.beta2
    )

    c_optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.lr_c,
        beta_1=args.beta1,
        beta_2=args.beta2
    )

    print(f"[OK] Generator: {generator.count_params():,} parameters")
    print(f"[OK] Critic: {critic.count_params():,} parameters")

    # Print model architecture verification
    print("\n" + "-" * 40)
    print("Model Architecture Verification:")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Generator layers: {gen_layers}")
    print(f"  Critic layers: {critic_layers}")
    print(f"  Lambda GP: {args.lambda_gp}")
    print(f"  GP Target: {args.gp_target}")
    print(f"  Learning rate (G): {args.lr_g}")
    print(f"  Learning rate (C): {args.lr_c}")
    print(f"  Beta1: {args.beta1}, Beta2: {args.beta2}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Critic iterations: {args.critic_iters}")
    print(f"  Condition dim: {condition_dim}")
    print(f"  Output dim (genes): {n_genes}")

    # =========================================================================
    # TRAINING FUNCTIONS
    # =========================================================================
    @tf.function
    def train_critic_step(real_expression, conditions):
        batch_size = tf.shape(real_expression)[0]
        noise = tf.random.normal((batch_size, args.latent_dim))

        with tf.GradientTape() as tape:
            fake_expression = generator((noise, conditions), training=True)
            real_score = critic((real_expression, conditions), training=True)
            fake_score = critic((fake_expression, conditions), training=True)

            real_score_mean = tf.reduce_mean(real_score)
            fake_score_mean = tf.reduce_mean(fake_score)

            c_loss = fake_score_mean - real_score_mean
            gp = gradient_penalty(critic, real_expression, fake_expression,
                                  conditions, lambda_gp=args.lambda_gp)
            total_c_loss = c_loss + gp

        c_gradients = tape.gradient(total_c_loss, critic.trainable_variables)
        c_optimizer.apply_gradients(zip(c_gradients, critic.trainable_variables))
        return c_loss, gp, real_score_mean, fake_score_mean

    @tf.function
    def train_generator_step(conditions):
        batch_size = tf.shape(conditions)[0]
        noise = tf.random.normal((batch_size, args.latent_dim))

        with tf.GradientTape() as tape:
            fake_expression = generator((noise, conditions), training=True)
            fake_score = critic((fake_expression, conditions), training=True)
            g_loss = -tf.reduce_mean(fake_score)

        g_gradients = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
        return g_loss

    # =========================================================================
    # MMD EVALUATION SETUP
    # =========================================================================
    mmd_history = []
    mmd_sigma = None
    real_val_data = None
    mmd_tracker = None

    if args.mmd_eval:
        # Get real validation data for MMD evaluation
        if hasattr(valid_data.X, 'toarray'):
            real_val_data = valid_data.X.toarray()
        else:
            real_val_data = np.array(valid_data.X)

        # Load PCA transform if provided
        pca_model = None
        if args.pca_path:
            import joblib
            pca_model = joblib.load(args.pca_path)
            real_val_data = pca_model.transform(real_val_data)
            print(f"[MMD] PCA-MMD mode: {pca_model.n_components_} components")
            print(f"[MMD] Transformed validation data to PCA space: {real_val_data.shape}")

        if args.sigma_path:
            from utils.mmd_early_stopping import load_sigma_heuristic, MMDEarlyStoppingTracker
            mmd_sigma = load_sigma_heuristic(args.sigma_path)
            print(f"[MMD] Loaded sigma from {args.sigma_path}: {mmd_sigma:.6f}")

            mmd_tracker = MMDEarlyStoppingTracker(
                model_type="gan",
                dataset_name=Path(args.data).stem,
                sigma=mmd_sigma,
                eval_epoch_interval=args.save_freq,  # save_freq IS the MMD freq
                patience_mmd=args.mmd_patience,
                min_delta=args.mmd_min_delta,
                save_dir=str(run_dir)
            )
        else:
            from utils.mmd import compute_sigma_from_real_data
            mmd_sigma = compute_sigma_from_real_data(real_val_data)
            print(f"[MMD] Computed sigma from validation data: {mmd_sigma:.6f}")

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print("\n" + "-" * 40)
    print("Starting training...")

    start_time = time.time()
    step = 0
    current_epoch = 0
    epoch_losses = []
    last_save_epoch = -1
    last_validated_epoch = -1
    mmd_should_stop = False

    # Checkpoint directory
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    train_iter = iter(train_dataset)

    c_losses = []
    g_losses = []
    gp_values = []
    real_scores = []
    fake_scores = []

    while step < args.max_steps and not mmd_should_stop:
        # Train critic
        for _ in range(args.critic_iters):
            try:
                real_expression, conditions = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataset)
                real_expression, conditions = next(train_iter)
                current_epoch += 1

            c_loss, gp, real_score, fake_score = train_critic_step(real_expression, conditions)
            c_losses.append(c_loss.numpy())
            gp_values.append(gp.numpy())
            real_scores.append(real_score.numpy())
            fake_scores.append(fake_score.numpy())

        # Train generator
        g_loss = train_generator_step(conditions)
        g_losses.append(g_loss.numpy())

        # Logging
        if step % args.log_freq == 0:
            avg_c_loss = np.mean(c_losses[-args.log_freq:]) if c_losses else 0
            avg_g_loss = np.mean(g_losses[-args.log_freq:]) if g_losses else 0
            avg_gp = np.mean(gp_values[-args.log_freq:]) if gp_values else 0
            avg_real_score = np.mean(real_scores[-args.log_freq:]) if real_scores else 0
            avg_fake_score = np.mean(fake_scores[-args.log_freq:]) if fake_scores else 0
            wasserstein_dist = avg_real_score - avg_fake_score

            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0

            print(f"Step {step}/{args.max_steps} | Epoch {current_epoch} | "
                  f"C: {avg_c_loss:.4f} | G: {avg_g_loss:.4f} | GP: {avg_gp:.4f} | "
                  f"W-dist: {wasserstein_dist:.4f} | {steps_per_sec:.2f} steps/s")

        # Save checkpoint - supports both step-based and epoch-based
        should_save = False
        if args.save_freq_unit == "epochs":
            if current_epoch > last_save_epoch and current_epoch > 0 and current_epoch % args.save_freq == 0:
                last_save_epoch = current_epoch
                should_save = True
        else:
            should_save = step > 0 and step % args.save_freq == 0

        if should_save:
            if args.save_freq_unit == "epochs":
                ckpt_path = checkpoint_dir / f"checkpoint_epoch_{current_epoch}"
            else:
                ckpt_path = checkpoint_dir / f"checkpoint_step_{step}"
            ckpt_path.mkdir(exist_ok=True)
            generator.save_weights(str(ckpt_path / "generator.weights.h5"))
            critic.save_weights(str(ckpt_path / "critic.weights.h5"))

            # Save training state with all metrics
            state = {
                'step': int(step),
                'epoch': int(current_epoch),
                'c_loss': float(avg_c_loss),
                'g_loss': float(avg_g_loss),
                'gradient_penalty': float(avg_gp),
                'real_score': float(avg_real_score),
                'fake_score': float(avg_fake_score),
                'wasserstein_distance': float(wasserstein_dist)
            }
            with open(ckpt_path / "training_state.json", 'w') as f:
                json.dump(state, f, indent=2)

            epoch_losses.append({
                "step": step,
                "epoch": current_epoch,
                "c_loss": float(avg_c_loss),
                "g_loss": float(avg_g_loss),
                "gradient_penalty": float(avg_gp),
                "real_score": float(avg_real_score),
                "fake_score": float(avg_fake_score),
                "wasserstein_distance": float(wasserstein_dist),
                "checkpoint": str(ckpt_path)
            })

            print(f"[CHECKPOINT] Saved at step {step} (epoch {current_epoch})")

            # MMD evaluation at every checkpoint when tracker is active
            if mmd_tracker is not None:
                mmd_value = evaluate_mmd_gan(
                    generator, valid_data, condition_processor,
                    args.latent_dim, mmd_sigma, real_val_data, pca_model
                )
                ckpt_filename = ckpt_path.name
                is_best, should_stop = mmd_tracker.record(
                    current_epoch, float(mmd_value),
                    loss_value=float(avg_c_loss),
                    checkpoint_filename=ckpt_filename
                )
                if is_best:
                    best_path = run_dir / "best_model"
                    best_path.mkdir(exist_ok=True)
                    generator.save_weights(str(best_path / "generator.weights.h5"))
                    critic.save_weights(str(best_path / "critic.weights.h5"))
                    print(f"  [MMD] Saved best model at epoch {current_epoch}")
                if should_stop:
                    mmd_tracker.save_history()
                    mmd_should_stop = True
            elif args.mmd_eval:
                # Legacy passive MMD (no tracker, no sigma_path)
                mmd_value = evaluate_mmd_gan(
                    generator, valid_data, condition_processor,
                    args.latent_dim, mmd_sigma, real_val_data, pca_model
                )
                mmd_history.append({'step': step, 'mmd': float(mmd_value)})
                print(f"[MMD] Step {step}: {mmd_value:.6f}")

        # Validation
        should_validate = False
        if args.save_freq_unit == "epochs":
            # In epoch mode, validate every validation_freq epochs
            if current_epoch > last_validated_epoch and current_epoch > 0 and current_epoch % args.validation_freq == 0:
                last_validated_epoch = current_epoch
                should_validate = True
        else:
            if step > 0 and step % (args.save_freq * args.validation_freq) == 0:
                should_validate = True
        if should_validate:
            val_c_losses = []
            val_g_losses = []
            val_real_scores = []
            val_fake_scores = []

            for real_expression, conditions in valid_dataset.take(10):
                batch_size = tf.shape(real_expression)[0]
                noise = tf.random.normal((batch_size, args.latent_dim))
                fake_expression = generator((noise, conditions), training=False)

                real_score = critic((real_expression, conditions), training=False)
                fake_score = critic((fake_expression, conditions), training=False)

                real_score_mean = tf.reduce_mean(real_score).numpy()
                fake_score_mean = tf.reduce_mean(fake_score).numpy()

                val_c_loss = fake_score_mean - real_score_mean
                val_g_loss = -fake_score_mean

                val_c_losses.append(val_c_loss)
                val_g_losses.append(val_g_loss)
                val_real_scores.append(real_score_mean)
                val_fake_scores.append(fake_score_mean)

            avg_val_c = np.mean(val_c_losses)
            avg_val_g = np.mean(val_g_losses)
            avg_val_real = np.mean(val_real_scores)
            avg_val_fake = np.mean(val_fake_scores)
            val_wasserstein = avg_val_real - avg_val_fake

            print(f"[VALIDATION] C: {avg_val_c:.4f}, G: {avg_val_g:.4f}, "
                  f"W-dist: {val_wasserstein:.4f}")

        step += 1

    training_time = time.time() - start_time
    print(f"\n[OK] Training completed in {training_time/60:.2f} minutes")

    # =========================================================================
    # MMD EVALUATION ON FINAL MODEL
    # =========================================================================
    if args.mmd_eval:
        if mmd_tracker is not None:
            # MMD early stopping was active - save history if not already saved
            if mmd_tracker.stop_reason is None:
                mmd_tracker.save_history(stop_reason_override='max_epochs_reached')
        else:
            # Legacy passive MMD mode
            print("\n" + "-" * 40)
            print("Evaluating MMD on final model...")

            final_mmd = evaluate_mmd_gan(
                generator, valid_data, condition_processor,
                args.latent_dim, mmd_sigma, real_val_data, pca_model
            )
            mmd_history.append({'step': 'final', 'mmd': float(final_mmd)})
            print(f"[MMD] Final model: {final_mmd:.6f}")

            # Save MMD history
            mmd_output = {'sigma': float(mmd_sigma), 'history': mmd_history}
            with open(run_dir / "mmd_history.json", 'w') as f:
                json.dump(mmd_output, f, indent=2)
            print(f"[OK] MMD history saved to: {run_dir / 'mmd_history.json'}")

    # =========================================================================
    # SAVE FINAL MODEL
    # =========================================================================
    print("\n" + "-" * 40)
    print("Saving outputs...")

    # Save final model
    final_path = run_dir / "final_model"
    final_path.mkdir(exist_ok=True)
    generator.save_weights(str(final_path / "generator.weights.h5"))
    critic.save_weights(str(final_path / "critic.weights.h5"))

    with open(final_path / "training_state.json", 'w') as f:
        json.dump({
            'step': int(step),
            'epoch': int(current_epoch),
            'final': True
        }, f, indent=2)

    print(f"[OK] Final model saved: {final_path}")

    # Save training metrics
    metrics = {
        "total_steps": step,
        "total_epochs": current_epoch,
        "final_checkpoint": str(final_path),
        "epoch_losses": epoch_losses,
        "training_time_seconds": training_time,
        "model_params": {
            "n_genes": n_genes,
            "latent_dim": args.latent_dim,
            "gen_layers": gen_layers,
            "critic_layers": critic_layers,
            "lambda_gp": args.lambda_gp,
            "lr_g": args.lr_g,
            "lr_c": args.lr_c,
            "batch_size": args.batch_size
        }
    }
    tracker.save_training_metrics(run_dir, metrics)
    print(f"[OK] Training metrics saved")

    # Save loss curves with additional metrics
    tracker.save_loss_curves(run_dir, epoch_losses, x_key='step',
                             y_keys=['c_loss', 'g_loss', 'gradient_penalty', 'wasserstein_distance'])

    # Save run config
    run_config = {
        "data_path": args.data,
        "condition_cols": condition_cols,
        "model_params": metrics["model_params"]
    }
    tracker.save_run_config(run_dir, {"gan": run_config})
    print(f"[OK] Run config saved")

    # Update latest_model.json — always points to final_model
    best_model_relative = Path(run_dir.name) / "final_model"

    tracker.update_latest_model(
        run_dir, step,
        0.0,  # No meaningful combined val_loss for GANs
        {"gan": run_config},
        additional_paths={"best_model_dir": str(best_model_relative)}
    )
    print(f"[OK] Updated latest_model.json")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"Final model: {final_path}")
    print(f"Total steps: {step}")
    print(f"Training time: {training_time/60:.2f} minutes")

    return True


def main():
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    args = parse_args()
    success = train_gan(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
