#!/usr/bin/env python
"""
Training script for Multi-Conditional scGAN
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import scanpy as sc
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_cond.utils import ConditionProcessor, ConditionProcessorCancer
from multi_cond.models import build_generator, build_critic, gradient_penalty


class MultiCondGANTrainer:
    """Trainer for Multi-Conditional scGAN"""

    def __init__(self, config_path="multi_cond/configs/parameters.json", run_dir=None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Set custom run directory if provided
        self.custom_run_dir = run_dir

        # Set up paths
        self.setup_paths()

        # Initialize components
        self.condition_processor = ConditionProcessorCancer()
        self.generator = None
        self.critic = None
        self.g_optimizer = None
        self.c_optimizer = None

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_checkpoint_path = None

    def setup_paths(self):
        """Create necessary directories"""
        for key in ['checkpoint_dir', 'log_dir', 'generated_dir']:
            path = self.config['output'][key]
            Path(path).mkdir(parents=True, exist_ok=True)

        # Use custom run directory if provided, otherwise create timestamped one
        if self.custom_run_dir:
            self.run_dir = self.custom_run_dir
            print(f"Using custom run directory: {self.run_dir}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(self.config['output']['checkpoint_dir'], f"run_{timestamp}")

        # Get absolute path for clarity
        self.run_dir = os.path.abspath(self.run_dir)
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)
        print(f"📂 Absolute run directory: {self.run_dir}")
        print(f"   Directory exists: {os.path.exists(self.run_dir)}")
        print(f"   Directory is writable: {os.access(self.run_dir, os.W_OK)}")

        # TensorBoard log directory - align with run directory
        run_name = os.path.basename(self.run_dir)
        self.tb_dir = os.path.join(self.config['output']['log_dir'], run_name)
        self.summary_writer = tf.summary.create_file_writer(self.tb_dir)

    def load_data(self, resume_indices=None):
        """Load and prepare training data

        Args:
            resume_indices: Tuple of (train_indices, valid_indices) when resuming from checkpoint
        """
        print("\nLoading data...")
        data_path = self.config['data']['input_file']

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load h5ad file
        self.adata = sc.read_h5ad(data_path)
        print(f"Loaded data shape: {self.adata.shape}")

        # Fit condition processor
        self.condition_processor.fit(self.adata)

        # Save condition processor config
        cond_config_path = os.path.join(self.run_dir, "condition_config.json")
        self.condition_processor.save_config(cond_config_path)

        # Split data into train/validation
        n_total = self.adata.n_obs

        if resume_indices is not None:
            # Use saved indices when resuming
            train_indices, valid_indices = resume_indices
            print("📂 Using saved train/validation split from checkpoint")
        else:
            # Create new split
            n_train = int(n_total * self.config['data']['train_split'])

            # Shuffle indices
            indices = np.random.permutation(n_total)
            train_indices = indices[:n_train]
            valid_indices = indices[n_train:]

            # Save indices for future resumption
            indices_path = os.path.join(self.run_dir, "data_split_indices.npz")
            np.savez(indices_path, train_indices=train_indices, valid_indices=valid_indices)
            print(f"💾 Saved data split indices to {indices_path}")

        # Create train/validation splits
        self.train_data = self.adata[train_indices].copy()
        self.valid_data = self.adata[valid_indices].copy()

        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(valid_indices)}")

        # Prepare TensorFlow datasets
        self.train_dataset = self.create_tf_dataset(self.train_data, shuffle=True)
        self.valid_dataset = self.create_tf_dataset(self.valid_data, shuffle=False)

    def create_tf_dataset(self, adata, shuffle=True):
        """Create TensorFlow dataset from AnnData"""
        # Extract expression data
        if hasattr(adata.X, 'toarray'):
            expression = adata.X.toarray().astype(np.float32)
        else:
            expression = adata.X.astype(np.float32)

        # Encode conditions
        conditions = self.condition_processor.encode_conditions(adata.obs)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((expression, conditions))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.batch(self.config['data']['batch_size'])
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def build_models(self):
        """Build generator and critic models"""
        print("\nBuilding models...")

        # Get condition dimension from processor
        condition_dim = self.condition_processor.total_condition_dim

        # Build generator
        self.generator = build_generator(
            latent_dim=self.config['model']['latent_dim'],
            condition_dim=condition_dim,
            output_dim=self.config['data']['gene_dim'],
            hidden_layers=self.config['model']['gen_layers']
        )

        # Build critic
        self.critic = build_critic(
            input_dim=self.config['data']['gene_dim'],
            condition_dim=condition_dim,
            hidden_layers=self.config['model']['critic_layers']
        )

        # Create optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['training']['learning_rate_g'],
            beta_1=self.config['training']['beta1'],
            beta_2=self.config['training']['beta2']
        )

        self.c_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['training']['learning_rate_c'],
            beta_1=self.config['training']['beta1'],
            beta_2=self.config['training']['beta2']
        )

        print(f"Generator parameters: {self.generator.count_params():,}")
        print(f"Critic parameters: {self.critic.count_params():,}")

    def load_checkpoint_state(self, checkpoint_path):
        """Load checkpoint state and config (before building models)

        Returns:
            Tuple of (train_indices, valid_indices) or None
        """
        checkpoint_path = os.path.abspath(checkpoint_path)
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT: {checkpoint_path}")
        print(f"{'='*60}")

        # Update run_dir to the parent directory of the checkpoint
        if "checkpoint_step_" in os.path.basename(checkpoint_path) or \
           os.path.basename(checkpoint_path) in ["best_model", "final_model"]:
            self.run_dir = os.path.dirname(checkpoint_path)
            print(f"📂 Run directory: {self.run_dir}")
        else:
            self.run_dir = checkpoint_path
            print(f"📂 Run directory: {self.run_dir}")

        # Load training state FIRST to get the original config
        state_path = os.path.join(checkpoint_path, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)

            # CRITICAL: Restore the original config from checkpoint
            if 'config' in state:
                self.config = state['config']
                print("✅ Restored original config from checkpoint")
                print(f"   Data file: {self.config['data']['input_file']}")
                print(f"   Batch size: {self.config['data']['batch_size']}")
            else:
                print("⚠️  WARNING: No config found in checkpoint, using provided config")

            self.current_step = state.get('step', 0)
            self.current_epoch = state.get('epoch', 0)
            self.best_loss = state.get('best_loss', float('inf'))
            self.patience_counter = state.get('patience_counter', 0)
            self.best_checkpoint_path = state.get('best_checkpoint', None)
            print(f"✅ Resuming from step {self.current_step}, epoch {self.current_epoch}")
            if self.best_checkpoint_path:
                print(f"   Best checkpoint was: {self.best_checkpoint_path}")
        else:
            print("⚠️  WARNING: No training state found")

        # Load condition processor config from run directory
        cond_config_path = os.path.join(self.run_dir, "condition_config.json")
        if os.path.exists(cond_config_path):
            self.condition_processor.load_config(cond_config_path)
            print(f"✅ Condition processor config loaded (dim={self.condition_processor.total_condition_dim})")
        else:
            raise ValueError(f"Condition processor config not found at {cond_config_path}")

        # Load data split indices
        indices_path = os.path.join(self.run_dir, "data_split_indices.npz")
        resume_indices = None
        if os.path.exists(indices_path):
            indices_data = np.load(indices_path)
            train_indices = indices_data['train_indices']
            valid_indices = indices_data['valid_indices']
            resume_indices = (train_indices, valid_indices)
            print(f"✅ Loaded data split indices (train: {len(train_indices)}, val: {len(valid_indices)})")
        else:
            print("⚠️  WARNING: No saved data split found, will create new random split")

        # Store checkpoint path for loading weights later
        self.checkpoint_to_load = checkpoint_path

        # Update TensorBoard directory to match run directory
        run_name = os.path.basename(self.run_dir)
        self.tb_dir = os.path.join(self.config['output']['log_dir'], run_name)
        self.summary_writer = tf.summary.create_file_writer(self.tb_dir)
        print(f"📊 TensorBoard logs will continue in: {self.tb_dir}")
        print(f"{'='*60}\n")

        return resume_indices

    def load_checkpoint_weights(self):
        """Load model weights from checkpoint (after building models)"""
        if not hasattr(self, 'checkpoint_to_load'):
            return

        checkpoint_path = self.checkpoint_to_load
        print(f"\nLoading model weights from checkpoint...")

        # Load model weights
        generator_path = os.path.join(checkpoint_path, "generator.weights.h5")
        critic_path = os.path.join(checkpoint_path, "critic.weights.h5")

        if not os.path.exists(generator_path) or not os.path.exists(critic_path):
            raise ValueError(f"Checkpoint files not found in {checkpoint_path}")

        self.generator.load_weights(generator_path)
        self.critic.load_weights(critic_path)
        print("✅ Model weights loaded successfully")

        # Clean up
        delattr(self, 'checkpoint_to_load')

    @tf.function
    def train_critic_step(self, real_expression, conditions):
        """Single training step for critic"""
        batch_size = tf.shape(real_expression)[0]

        # Sample noise
        noise = tf.random.normal((batch_size, self.config['model']['latent_dim']))

        with tf.GradientTape() as tape:
            # Generate fake samples
            fake_expression = self.generator((noise, conditions), training=True)

            # Critic scores
            real_score = self.critic((real_expression, conditions), training=True)
            fake_score = self.critic((fake_expression, conditions), training=True)

            # Wasserstein loss
            c_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)

            # Gradient penalty
            gp = gradient_penalty(
                self.critic,
                real_expression,
                fake_expression,
                conditions,
                lambda_gp=self.config['model']['lambda_gp']
            )

            # Total critic loss
            total_c_loss = c_loss + gp

        # Update critic
        c_gradients = tape.gradient(total_c_loss, self.critic.trainable_variables)
        self.c_optimizer.apply_gradients(zip(c_gradients, self.critic.trainable_variables))

        return c_loss, gp

    @tf.function
    def train_generator_step(self, conditions):
        """Single training step for generator"""
        batch_size = tf.shape(conditions)[0]

        # Sample noise
        noise = tf.random.normal((batch_size, self.config['model']['latent_dim']))

        with tf.GradientTape() as tape:
            # Generate fake samples
            fake_expression = self.generator((noise, conditions), training=True)

            # Critic score for fake samples
            fake_score = self.critic((fake_expression, conditions), training=True)

            # Generator loss (negative of critic score)
            g_loss = -tf.reduce_mean(fake_score)

        # Update generator
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return g_loss

    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60)

        # Training parameters
        critic_iters = self.config['training']['critic_iters']
        max_steps = self.config['training']['max_steps']
        log_freq = self.config['training']['log_freq']
        save_freq = self.config['training']['save_freq']

        # Training metrics
        c_losses = []
        g_losses = []
        gp_losses = []

        start_time = time.time()
        step = self.current_step  # Resume from current step if loaded from checkpoint

        # Create iterator for training data
        train_iter = iter(self.train_dataset)

        while step < max_steps:
            # Train critic for multiple iterations
            for _ in range(critic_iters):
                try:
                    real_expression, conditions = next(train_iter)
                except StopIteration:
                    # Reset iterator when we reach the end
                    train_iter = iter(self.train_dataset)
                    real_expression, conditions = next(train_iter)
                    self.current_epoch += 1

                c_loss, gp = self.train_critic_step(real_expression, conditions)
                c_losses.append(c_loss.numpy())
                gp_losses.append(gp.numpy())

            # Train generator once
            g_loss = self.train_generator_step(conditions)
            g_losses.append(g_loss.numpy())

            # Logging
            if step % log_freq == 0:
                avg_c_loss = np.mean(c_losses[-log_freq:]) if c_losses else 0
                avg_g_loss = np.mean(g_losses[-log_freq:]) if g_losses else 0
                avg_gp = np.mean(gp_losses[-log_freq:]) if gp_losses else 0

                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0

                print(f"Step {step}/{max_steps} | Epoch {self.current_epoch} | "
                      f"C loss: {avg_c_loss:.4f} | G loss: {avg_g_loss:.4f} | "
                      f"GP: {avg_gp:.4f} | {steps_per_sec:.2f} steps/s")

                # TensorBoard logging
                with self.summary_writer.as_default():
                    tf.summary.scalar('loss/critic', avg_c_loss, step=step)
                    tf.summary.scalar('loss/generator', avg_g_loss, step=step)
                    tf.summary.scalar('loss/gradient_penalty', avg_gp, step=step)
                    tf.summary.scalar('training/steps_per_sec', steps_per_sec, step=step)
                    tf.summary.scalar('training/epoch', self.current_epoch, step=step)

            # Save checkpoint
            if step % save_freq == 0 and step > 0:
                self.save_checkpoint(step)

            # Validation (less frequent)
            if step % (save_freq * 5) == 0 and step > 0:
                self.validate(step)

            step += 1
            self.current_step = step

        print("\nTraining completed!")

        # Run final validation to determine if this is the best model
        print("\nRunning final validation...")
        final_val_loss = self.validate(step)

        # Save as final model
        self.save_checkpoint(step, final=True)

        # Check if final model is also the best
        if final_val_loss is not False:  # validate returns False on early stopping
            print(f"\nFinal validation loss: {final_val_loss:.4f}")
            if final_val_loss == self.best_loss:
                print("✨ Final model IS the best model!")
            elif self.best_checkpoint_path:
                print(f"📌 Best model remains at: {self.best_checkpoint_path}")
                print(f"   Best validation loss: {self.best_loss:.4f}")
                print(f"   Final validation loss: {final_val_loss:.4f}")
                print(f"   Difference: {final_val_loss - self.best_loss:.4f} (higher is worse)")
            else:
                print("Note: No best model was saved during training")

    def validate(self, step):
        """Validation step"""
        print(f"\nRunning validation at step {step}, epoch {self.current_epoch}...")

        val_c_losses = []
        val_g_losses = []
        val_real_scores = []
        val_fake_scores = []

        for real_expression, conditions in self.valid_dataset.take(10):
            # Critic loss
            batch_size = tf.shape(real_expression)[0]
            noise = tf.random.normal((batch_size, self.config['model']['latent_dim']))
            fake_expression = self.generator((noise, conditions), training=False)

            real_score = self.critic((real_expression, conditions), training=False)
            fake_score = self.critic((fake_expression, conditions), training=False)

            c_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)
            g_loss = -tf.reduce_mean(fake_score)

            val_c_losses.append(c_loss.numpy())
            val_g_losses.append(g_loss.numpy())
            val_real_scores.append(tf.reduce_mean(real_score).numpy())
            val_fake_scores.append(tf.reduce_mean(fake_score).numpy())

        avg_val_c_loss = np.mean(val_c_losses)
        avg_val_g_loss = np.mean(val_g_losses)
        avg_real_score = np.mean(val_real_scores)
        avg_fake_score = np.mean(val_fake_scores)
        wasserstein_distance = avg_real_score - avg_fake_score

        print(f"Validation - C loss: {avg_val_c_loss:.4f} | G loss: {avg_val_g_loss:.4f}")
        print(f"            Real score: {avg_real_score:.4f} | Fake score: {avg_fake_score:.4f}")
        print(f"            Wasserstein distance: {wasserstein_distance:.4f}")

        # TensorBoard logging with enhanced metrics
        with self.summary_writer.as_default():
            # Loss metrics
            tf.summary.scalar('validation/critic_loss', avg_val_c_loss, step=step)
            tf.summary.scalar('validation/generator_loss', avg_val_g_loss, step=step)
            tf.summary.scalar('validation/total_loss', avg_val_c_loss + avg_val_g_loss, step=step)

            # Score metrics
            tf.summary.scalar('validation/real_score', avg_real_score, step=step)
            tf.summary.scalar('validation/fake_score', avg_fake_score, step=step)
            tf.summary.scalar('validation/wasserstein_distance', wasserstein_distance, step=step)

            # Epoch tracking
            tf.summary.scalar('validation/epoch', self.current_epoch, step=step)

            # Quality metrics
            tf.summary.scalar('validation/generator_quality', -avg_val_g_loss, step=step)  # Higher is better
            tf.summary.scalar('validation/critic_margin', abs(avg_val_c_loss), step=step)  # Distance between real/fake

        # Early stopping check
        val_loss = avg_val_c_loss + avg_val_g_loss
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            self.save_checkpoint(step, best=True)
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config['training']['patience']:
                print(f"Early stopping triggered after {self.patience_counter} validations without improvement")
                return False

        return val_loss  # Return the validation loss instead of True

    def save_checkpoint(self, step, best=False, final=False):
        """Save model checkpoint"""
        if best:
            checkpoint_path = os.path.join(self.run_dir, "best_model")
            self.best_checkpoint_path = checkpoint_path
            print("\n" + "="*60)
            print(f"🏆 NEW BEST MODEL SAVED at step {step}!")
            print(f"   Validation loss improved to: {self.best_loss:.4f}")
            print(f"   Location: {checkpoint_path}")
            print("="*60 + "\n")
        elif final:
            checkpoint_path = os.path.join(self.run_dir, "final_model")
            print(f"\n📦 Final model saved to: {checkpoint_path}")
            if self.best_checkpoint_path:
                print(f"📌 Best model was saved at: {self.best_checkpoint_path}")
        else:
            checkpoint_path = os.path.join(self.run_dir, f"checkpoint_step_{step}")
            print(f"💾 Checkpoint saved at step {step}")

        # Save generator and critic with .weights.h5 extension for TF 2.20+
        os.makedirs(checkpoint_path, exist_ok=True)

        # Get absolute paths for debugging
        abs_checkpoint_path = os.path.abspath(checkpoint_path)
        gen_path = os.path.join(abs_checkpoint_path, "generator.weights.h5")
        critic_path = os.path.join(abs_checkpoint_path, "critic.weights.h5")

        print(f"📁 Saving to absolute path: {abs_checkpoint_path}")
        print(f"   - Generator: {gen_path}")
        print(f"   - Critic: {critic_path}")

        self.generator.save_weights(gen_path)
        self.critic.save_weights(critic_path)

        # Save training state - convert numpy types to Python types for JSON serialization
        state = {
            'step': int(step),
            'epoch': int(self.current_epoch),
            'best_loss': float(self.best_loss) if self.best_loss is not None else None,
            'patience_counter': int(self.patience_counter),
            'config': self.config,
            'best_checkpoint': self.best_checkpoint_path
        }

        with open(os.path.join(checkpoint_path, "training_state.json"), 'w') as f:
            json.dump(state, f, indent=2)


def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Multi-Conditional scGAN')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint directory to resume from')
    parser.add_argument('--config', type=str, default='multi_cond/configs/parameters.json',
                        help='Path to config file')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Custom run directory for checkpoints and logs')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Check TensorFlow GPU availability
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {len(gpus)} device(s)")
    else:
        print("No GPU available, using CPU")

    # Create trainer and run training
    trainer = MultiCondGANTrainer(config_path=args.config, run_dir=args.run_dir)

    # Resume from checkpoint if specified
    resume_indices = None
    if args.resume:
        # Step 1: Load checkpoint state (config, condition processor, data split indices)
        resume_indices = trainer.load_checkpoint_state(args.resume)
        # Step 2: Load data with the saved split
        trainer.load_data(resume_indices=resume_indices)
        # Step 3: Build models (now we have correct config and condition_dim)
        trainer.build_models()
        # Step 4: Load model weights
        trainer.load_checkpoint_weights()
    else:
        # Normal training: load data first, then build models
        trainer.load_data()
        trainer.build_models()

    trainer.train()


if __name__ == "__main__":
    main()