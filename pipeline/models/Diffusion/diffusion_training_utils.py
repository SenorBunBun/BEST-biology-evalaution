import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Dict, List, Optional, Tuple
import os
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Import from the condition processing and generation utilities
from diffusion_generation_utils import (
    ConditionProcessor,
    save_model_with_config
)

# Import the model architecture
from diffusion_model_cfg import ClassifierFreeDiffusionModel


class H5ADConditionDataset(Dataset):
    """Dataset for pre-processed h5ad files with conditions"""

    def __init__(self, adata, condition_processor: ConditionProcessor, vae_model=None, use_latent: bool = True):
        self.adata = adata
        self.condition_processor = condition_processor
        self.vae_model = vae_model
        self.use_latent = use_latent

        # Get gene expression data (already normalized)
        if hasattr(adata.X, 'toarray'):
            self.gene_expression = adata.X.toarray().astype(np.float32)
        else:
            self.gene_expression = adata.X.astype(np.float32)

        # Encode to latent if VAE provided
        if self.use_latent and self.vae_model is not None:
            print("Encoding to latent space...")
            latent_repr, _ = self.vae_model.get_latent_representation(self.gene_expression)
            self.latent_data = latent_repr.astype(np.float32)
        else:
            self.latent_data = self.gene_expression

        # Prepare conditions
        self.conditions = condition_processor.prepare_conditions_for_training(adata)

        print(f"Dataset created: {len(self)} cells, {self.latent_data.shape[1]} latent dims")
        print(f"Conditions: {list(self.conditions.keys())}")

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        data = torch.tensor(self.latent_data[idx], dtype=torch.float32)
        conditions = {name: tensor[idx] for name, tensor in self.conditions.items()}
        return data, conditions


class DiffusionTrainer:
    """Handles diffusion model training with metrics and checkpointing"""

    def __init__(self, model: ClassifierFreeDiffusionModel, condition_processor: ConditionProcessor,
                 vae_model=None, num_timesteps: int = 1000, beta_schedule: str = "cosine",
                 device: str = None, resume_checkpoint: str = None):
        self.model = model
        self.condition_processor = condition_processor
        self.vae_model = vae_model
        self.num_timesteps = num_timesteps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.resume_checkpoint = resume_checkpoint

        # Move models to device
        self.model = self.model.to(self.device)
        if self.vae_model is not None:
            self.vae_model = self.vae_model.to(self.device)
            self.vae_model.eval()

        # Initialize beta schedule
        self.betas = self._get_beta_schedule(beta_schedule, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)

        # Convert to tensors and move to device for efficient indexing
        self.alphas_cumprod_tensor = torch.tensor(self.alphas_cumprod, dtype=torch.float32, device=self.device)

        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [], 'epoch_times': [], 'learning_rates': [],
            'best_val_loss': float('inf'), 'best_epoch': 0
        }

        # Training state
        self.start_epoch = 0

    def _get_beta_schedule(self, schedule: str, timesteps: int):
        if schedule == "linear":
            return np.linspace(0.0001, 0.02, timesteps)
        elif schedule == "cosine":
            s = 0.008
            steps = timesteps + 1
            x = np.linspace(0, timesteps, steps)
            alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # Use tensor indexing instead of numpy indexing
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod_tensor[t])
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - self.alphas_cumprod_tensor[t])

        # Reshape for broadcasting if needed
        if len(sqrt_alphas_cumprod_t.shape) > 0:
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start, conditions):
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_t, t, conditions, training=True)
        return nn.functional.mse_loss(predicted_noise, noise)

    def load_checkpoint(self, checkpoint_path: str, optimizer=None, scheduler=None):
        """Load checkpoint and resume training state"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Optimizer state loaded")

        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✅ Scheduler state loaded")

        # Load training history and state
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            print("✅ Training history loaded")

        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"✅ Resuming from epoch {self.start_epoch}")

        # Load other training parameters
        if 'num_timesteps' in checkpoint:
            self.num_timesteps = checkpoint['num_timesteps']
        if 'betas' in checkpoint:
            self.betas = checkpoint['betas']
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = np.cumprod(self.alphas)
            self.alphas_cumprod_tensor = torch.tensor(self.alphas_cumprod, dtype=torch.float32, device=self.device)

        print("✅ Checkpoint loaded successfully!")
        return checkpoint

    def train_epoch(self, train_loader, optimizer, epoch):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for batch_data, batch_conditions in pbar:
            batch_data = batch_data.to(self.device)
            for cond_name in batch_conditions:
                batch_conditions[cond_name] = batch_conditions[cond_name].to(self.device)

            optimizer.zero_grad()
            loss = self.compute_loss(batch_data, batch_conditions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate_epoch(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]")
        for batch_data, batch_conditions in pbar:
            batch_data = batch_data.to(self.device)
            for cond_name in batch_conditions:
                batch_conditions[cond_name] = batch_conditions[cond_name].to(self.device)

            loss = self.compute_loss(batch_data, batch_conditions)
            total_loss += loss.item()
            pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})

        return total_loss / len(val_loader)

    def save_checkpoint(self, epoch, optimizer, scheduler, save_dir, is_best=False):
        os.makedirs(save_dir, exist_ok=True)

        if is_best:
            # Use save_model_with_config for best models
            model_name = f"best_model_epoch_{epoch + 1}"
            save_model_with_config(
                model=self.model,
                condition_processor=self.condition_processor,
                save_dir=save_dir,
                model_name=model_name
            )
            print(
                f"💾 New best model saved using save_model_with_config (val_loss: {self.history['best_val_loss']:.6f})")

        # Also save regular checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'history': self.history,
            'num_timesteps': self.num_timesteps,
            'betas': self.betas
        }

        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save(checkpoint, checkpoint_path)

        # Keep only last few checkpoints to save space
        self._cleanup_old_checkpoints(save_dir, keep_last=3)

    def _cleanup_old_checkpoints(self, save_dir, keep_last=3):
        """Keep only the last N checkpoints to save disk space"""
        checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        if len(checkpoints) > keep_last:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint_path = os.path.join(save_dir, checkpoint)
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    print(f"🗑️ Removed old checkpoint: {checkpoint}")

    @torch.no_grad()
    def evaluate_mmd(self, val_loader, sigma, real_gene_data, num_inference_steps=50, pca_model=None):
        """
        Generate synthetic gene expression matching validation conditions and compute MMD.

        The diffusion model generates latent samples, which are then decoded by the VAE
        to get full gene expression for MMD computation.

        Args:
            val_loader: Validation data loader
            sigma: Pre-computed kernel bandwidth (from real gene expression)
            real_gene_data: Real validation data as gene expression (numpy array, already PCA-transformed if PCA enabled)
            num_inference_steps: Number of diffusion inference steps
            pca_model: Fitted PCA model for transforming synthetic data (optional)

        Returns:
            mmd_value: float
        """
        from utils.mmd import compute_mmd
        from diffusion_generation_utils import DiffusionModelGenerator

        if self.vae_model is None:
            raise ValueError("VAE model required for MMD evaluation on gene expression")

        self.model.eval()
        self.vae_model.eval()

        # Collect all conditions from validation data and convert back to original values
        condition_list = []
        for batch_data, batch_conditions in val_loader:
            for i in range(batch_data.shape[0]):
                # Get the condition tensors for this sample
                sample_conds = {}
                for name, tensor in batch_conditions.items():
                    sample_conds[name] = tensor[i]

                # Use denormalize_conditions to convert back to original values
                real_conds = self.condition_processor.denormalize_conditions(sample_conds)
                condition_list.append(real_conds)

        # Create generator (uses same num_timesteps and default cosine schedule)
        generator = DiffusionModelGenerator(
            self.model,
            self.condition_processor,
            self.device,
            num_timesteps=self.num_timesteps
        )

        # Group by unique conditions and generate
        from collections import Counter
        # Create hashable tuples for counting
        cond_tuples = [tuple(sorted(c.items())) for c in condition_list]
        cond_counts = Counter(cond_tuples)

        generated_latent_list = []
        for cond_tuple, count in cond_counts.items():
            cond_dict = dict(cond_tuple)

            # Generate latent samples using original (denormalized) conditions
            latent_samples = generator.generate_specific_conditions(
                cond_dict, count, num_inference_steps=num_inference_steps
            )
            generated_latent_list.append(latent_samples)

        # Stack all latent samples
        generated_latent = np.vstack(generated_latent_list)

        # Decode latent to gene expression using VAE (same approach as generation)
        # Use batching for memory efficiency
        batch_size = 100
        decoded_samples = []

        for i in range(0, len(generated_latent), batch_size):
            batch = torch.tensor(
                generated_latent[i:i + batch_size],
                dtype=torch.float32,
                device=self.device
            )
            decoded_batch = self.vae_model.decode(batch)
            decoded_samples.append(decoded_batch.cpu())

        # Concatenate and clamp to non-negative
        generated_gene_expr = torch.cat(decoded_samples, dim=0)
        generated_gene_expr = torch.clamp(generated_gene_expr, min=0).numpy()

        if pca_model is not None:
            generated_gene_expr = pca_model.transform(generated_gene_expr)
        mmd_value = compute_mmd(real_gene_data, generated_gene_expr, sigma)
        return mmd_value

    def train(self, train_loader, val_loader, num_epochs: int, learning_rate: float = 1e-4,
              weight_decay: float = 1e-5, patience: int = 20, save_every: int = 10,
              save_dir: str = "checkpoints", mmd_eval: bool = False, mmd_eval_freq: int = 5,
              real_val_gene_data: np.ndarray = None, mmd_tracker=None, pca_model=None):
        """
        Train the diffusion model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            save_every: Checkpoint frequency
            save_dir: Directory for checkpoints
            mmd_eval: Enable MMD evaluation
            mmd_eval_freq: MMD evaluation frequency (epochs)
            real_val_gene_data: Real validation gene expression data (required if mmd_eval=True)
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                         patience=patience // 2)

        # Load checkpoint if resuming
        if self.resume_checkpoint and os.path.exists(self.resume_checkpoint):
            self.load_checkpoint(self.resume_checkpoint, optimizer, scheduler)

        # MMD evaluation setup
        mmd_history = []
        mmd_sigma = None

        if mmd_eval:
            if real_val_gene_data is None:
                raise ValueError("real_val_gene_data is required when mmd_eval=True")

            from utils.mmd import compute_sigma_from_real_data

            # Compute sigma from real gene expression data
            print("[MMD] Computing sigma from validation gene expression data...")
            mmd_sigma = compute_sigma_from_real_data(real_val_gene_data)
            print(f"[MMD] Computed sigma from validation data: {mmd_sigma:.6f}")

        print(f"🚀 Training for {num_epochs} epochs (starting from epoch {self.start_epoch})")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"💻 Device: {self.device}")
        print("-" * 60)

        early_stop_counter = 0
        start_time = time.time()

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start_time = time.time()

            train_loss = self.train_epoch(train_loader, optimizer, epoch)
            val_loss = self.validate_epoch(val_loader, epoch)

            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch_times'].append(epoch_time)
            self.history['learning_rates'].append(current_lr)

            scheduler.step(val_loss)

            print(f"Epoch {epoch + 1:3d}/{num_epochs} | Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

            if mmd_tracker is not None:
                # === MMD-BASED MODE ===
                if mmd_tracker.should_evaluate(epoch):
                    # Save checkpoint
                    self.save_checkpoint(epoch, optimizer, scheduler, save_dir, is_best=False)
                    ckpt_filename = f"best_model_epoch_{epoch + 1}_model.pth"

                    # Evaluate MMD
                    mmd_value = self.evaluate_mmd(val_loader, mmd_tracker.sigma, real_val_gene_data,
                                                  pca_model=pca_model)
                    is_best, should_stop = mmd_tracker.record(epoch, mmd_value, val_loss, ckpt_filename)

                    if is_best:
                        self.history['best_val_loss'] = val_loss
                        self.history['best_epoch'] = epoch
                        self.save_checkpoint(epoch, optimizer, scheduler, save_dir, is_best=True)

                    if should_stop:
                        mmd_tracker.save_history()
                        break
                # No loss-based stopping or periodic checkpoints in MMD mode
            else:
                # === EXISTING LOSS-BASED MODE (unchanged) ===
                is_best = val_loss < self.history['best_val_loss']
                if is_best:
                    self.history['best_val_loss'] = val_loss
                    self.history['best_epoch'] = epoch
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if (epoch + 1) % save_every == 0 or is_best:
                    self.save_checkpoint(epoch, optimizer, scheduler, save_dir, is_best)

                # Legacy passive MMD evaluation
                if mmd_eval and (epoch + 1) % mmd_eval_freq == 0:
                    mmd_value = self.evaluate_mmd(val_loader, mmd_sigma, real_val_gene_data,
                                                  pca_model=pca_model)
                    mmd_history.append({'epoch': epoch + 1, 'mmd': float(mmd_value)})
                    print(f"[MMD] Epoch {epoch + 1}: {mmd_value:.6f}")

                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        total_time = time.time() - start_time
        print(f"\nTraining completed! Time: {total_time / 60:.1f} min")
        print(f"Best val loss: {self.history['best_val_loss']:.6f} (epoch {self.history['best_epoch'] + 1})")

        if mmd_tracker is not None:
            # Save history if max epochs reached (not already saved by early stop)
            if mmd_tracker.stop_reason is None:
                mmd_tracker.save_history(stop_reason_override='max_epochs_reached')
        elif mmd_eval:
            # Legacy passive MMD: evaluate best model
            print("\n[MMD] Evaluating best model...")
            best_model_path = os.path.join(save_dir, f"best_model_epoch_{self.history['best_epoch'] + 1}_model.pth")
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"[MMD] Loaded best model from: {best_model_path}")

            final_mmd = self.evaluate_mmd(val_loader, mmd_sigma, real_val_gene_data,
                                                 pca_model=pca_model)
            mmd_history.append({'epoch': 'best', 'mmd': float(final_mmd)})
            print(f"[MMD] Best model: {final_mmd:.6f}")

            mmd_output = {'sigma': float(mmd_sigma), 'history': mmd_history}
            with open(os.path.join(save_dir, 'mmd_history.json'), 'w') as f:
                json.dump(mmd_output, f, indent=2)
            print(f"[OK] MMD history saved to: {os.path.join(save_dir, 'mmd_history.json')}")

        # Save training history
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

        self.plot_training_curves(save_dir)

    def plot_training_curves(self, save_dir):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train', color='blue')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val', color='red')
        axes[0, 0].axvline(x=self.history['best_epoch'] + 1, color='green', linestyle='--',
                           label=f'Best (epoch {self.history["best_epoch"] + 1})')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Learning rate
        axes[0, 1].plot(epochs, self.history['learning_rates'], color='purple')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)

        # Epoch times
        axes[1, 0].plot(epochs, self.history['epoch_times'], color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Training Time per Epoch')
        axes[1, 0].grid(True, alpha=0.3)

        # Overfitting indicator
        loss_diff = np.array(self.history['val_loss']) - np.array(self.history['train_loss'])
        axes[1, 1].plot(epochs, loss_diff, color='brown')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Val Loss - Train Loss')
        axes[1, 1].set_title('Overfitting Indicator')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()


def validate_condition_config(adata, condition_config: Dict[str, str]):
    """Validate that condition config matches available columns"""
    available_columns = set(adata.obs.columns)
    config_columns = set(condition_config.keys())

    # Check if all config columns exist
    missing_columns = config_columns - available_columns
    if missing_columns:
        raise ValueError(f"Condition columns not found in data: {missing_columns}")

    print("✅ Condition validation:")
    for col_name, col_type in condition_config.items():
        unique_vals = len(adata.obs[col_name].dropna().unique())
        print(f"   {col_name}: {col_type} ({unique_vals} unique values)")

        # Special validation for categorical with binary values
        if col_type == 'categorical':
            unique_values = adata.obs[col_name].dropna().unique()
            print(f"      Categories: {list(unique_values)}")

    return True


def create_data_loaders(adata, condition_config: Dict[str, str], vae_model=None, batch_size: int = 128,
                        train_split: float = 0.8, val_split: float = 0.1, seed: int = 42,
                        split_indices: str = None):
    """Create data loaders from h5ad data

    Args:
        adata: AnnData object with gene expression data
        condition_config: Dict mapping condition column names to types
        vae_model: Optional VAE model for latent encoding
        batch_size: Batch size for data loaders
        train_split: Fraction for training (used if split_indices is None)
        val_split: Fraction for validation (used if split_indices is None)
        seed: Random seed (used if split_indices is None)
        split_indices: Path to pre-computed split indices file (.npz)
    """

    # Validate conditions
    validate_condition_config(adata, condition_config)

    # Convert simple config to full config format
    full_condition_config = {}
    for col_name, col_type in condition_config.items():
        if col_type == 'categorical':
            n_categories = len(adata.obs[col_name].dropna().unique())
            full_condition_config[col_name] = {
                'type': 'categorical',
                'n_categories': n_categories,
                'description': f'Categorical condition: {col_name}'
            }
        elif col_type == 'continuous':
            full_condition_config[col_name] = {
                'type': 'continuous',
                'description': f'Continuous condition: {col_name}'
            }
        else:
            raise ValueError(f"Unknown condition type: {col_type}")

    # Setup condition processor
    processor = ConditionProcessor(full_condition_config)
    processor.fit_conditions(adata)

    # Create dataset
    dataset = H5ADConditionDataset(adata, processor, vae_model, use_latent=(vae_model is not None))

    # Split dataset
    n_samples = len(dataset)

    if split_indices:
        # Use pre-computed stratified split indices
        from utils.data_splitter import load_split_indices
        train_idx, val_idx = load_split_indices(split_indices)

        # Remaining indices become test set
        all_indices = set(range(n_samples))
        used_indices = set(train_idx.tolist()) | set(val_idx.tolist())
        test_idx = list(all_indices - used_indices)

        train_dataset = torch.utils.data.Subset(dataset, train_idx.tolist())
        val_dataset = torch.utils.data.Subset(dataset, val_idx.tolist())
        test_dataset = torch.utils.data.Subset(dataset, test_idx) if test_idx else torch.utils.data.Subset(dataset, [])

        n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)
        print(f"[OK] Loaded stratified split from: {split_indices}")
    else:
        # Fallback to random split
        n_train = int(n_samples * train_split)
        n_val = int(n_samples * val_split)
        n_test = n_samples - n_train - n_val

        torch.manual_seed(seed)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test]
        )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"📊 Data splits: Train={n_train:,}, Val={n_val:,}, Test={n_test:,}")

    return train_loader, val_loader, test_loader, processor, full_condition_config
