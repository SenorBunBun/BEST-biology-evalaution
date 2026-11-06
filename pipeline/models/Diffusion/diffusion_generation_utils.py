import torch
import torch.nn as nn
from torch import nn as nn_reg
import numpy as np
import pandas as pd
import json
import os
import math
from typing import Dict, List, Optional, Union, Tuple
import pickle
from pathlib import Path


class ConditionProcessor:
    """
    Handles condition processing, normalization, and conversion for diffusion model generation
    """

    def __init__(self, condition_config: Dict[str, Dict]):
        self.condition_config = condition_config
        self.categorical_mappings = {}
        self.continuous_stats = {}
        self.is_fitted = False

    def fit_conditions(self, adata):
        """Fit the condition processor to the h5ad data"""
        for cond_name, cond_info in self.condition_config.items():
            if cond_name not in adata.obs.columns:
                print(f"Warning: Condition '{cond_name}' not found in adata.obs")
                continue

            values = adata.obs[cond_name].values

            if cond_info['type'] == 'categorical':
                unique_values = pd.Series(values).dropna().unique()
                self.categorical_mappings[cond_name] = {
                    'categories': unique_values.tolist(),
                    'category_to_idx': {cat: idx for idx, cat in enumerate(unique_values)},
                    'idx_to_category': {idx: cat for idx, cat in enumerate(unique_values)}
                }
                self.condition_config[cond_name]['n_categories'] = len(unique_values)

            elif cond_info['type'] == 'continuous':
                clean_values = pd.Series(values).dropna().astype(float)
                self.continuous_stats[cond_name] = {
                    'mean': float(clean_values.mean()),
                    'std': float(clean_values.std()),
                    'min': float(clean_values.min()),
                    'max': float(clean_values.max())
                }

        self.is_fitted = True
        print("Condition processor fitted successfully!")

    def save_config(self, save_path: str):
        """Save condition configuration and statistics"""
        config_data = {
            'condition_config': self.condition_config,
            'categorical_mappings': self.categorical_mappings,
            'continuous_stats': self.continuous_stats,
            'is_fitted': self.is_fitted
        }

        with open(save_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"Configuration saved to: {save_path}")

    @classmethod
    def load_config(cls, config_path: str):
        """Load condition configuration and statistics"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        processor = cls(config_data['condition_config'])
        processor.categorical_mappings = config_data['categorical_mappings']
        processor.continuous_stats = config_data['continuous_stats']
        processor.is_fitted = config_data['is_fitted']

        print(f"Configuration loaded from: {config_path}")
        return processor

    def prepare_conditions_for_generation(self, condition_specs: Dict[str, Union[str, float, int]]) -> Dict[
        str, torch.Tensor]:
        """Prepare conditions from real-world values for generation"""
        if not self.is_fitted:
            raise ValueError("ConditionProcessor must be fitted before use")

        conditions = {}

        for cond_name, cond_value in condition_specs.items():
            if cond_name not in self.condition_config:
                print(f"Warning: Condition '{cond_name}' not in configuration, skipping")
                continue

            cond_info = self.condition_config[cond_name]

            if cond_info['type'] == 'categorical':
                mapping = self.categorical_mappings[cond_name]
                if cond_value in mapping['category_to_idx']:
                    idx = mapping['category_to_idx'][cond_value]
                    one_hot = torch.zeros(1, len(mapping['categories']))
                    one_hot[0, idx] = 1.0
                    conditions[cond_name] = one_hot
                else:
                    available_cats = list(mapping['categories'])
                    raise ValueError(f"Category '{cond_value}' not found for condition '{cond_name}'. "
                                     f"Available categories: {available_cats}")

            elif cond_info['type'] == 'continuous':
                stats = self.continuous_stats[cond_name]
                normalized = (float(cond_value) - stats['mean']) / (stats['std'] + 1e-8)
                conditions[cond_name] = torch.tensor([normalized], dtype=torch.float32)

        return conditions

    def prepare_conditions_for_training(self, adata) -> Dict[str, torch.Tensor]:
        """
        Prepare conditions from h5ad for training

        Args:
            adata: AnnData object

        Returns:
            Dict of condition tensors ready for model input
        """
        conditions = {}

        for cond_name, cond_info in self.condition_config.items():
            if cond_name not in adata.obs.columns:
                continue

            values = adata.obs[cond_name].values

            if cond_info['type'] == 'categorical':
                # Convert to one-hot
                mapping = self.categorical_mappings[cond_name]
                indices = [mapping['category_to_idx'].get(val, 0) for val in values]
                one_hot = torch.zeros(len(indices), len(mapping['categories']))
                for i, idx in enumerate(indices):
                    one_hot[i, idx] = 1.0
                conditions[cond_name] = one_hot

            elif cond_info['type'] == 'continuous':
                # Normalize continuous values
                clean_values = pd.Series(values).fillna(0).astype(float)
                stats = self.continuous_stats[cond_name]
                normalized = (clean_values - stats['mean']) / (stats['std'] + 1e-8)
                conditions[cond_name] = torch.tensor(normalized.values, dtype=torch.float32)

        return conditions

    def denormalize_conditions(self, conditions: Dict[str, torch.Tensor]) -> Dict[str, Union[str, float]]:
        """Convert normalized conditions back to real-world values"""
        real_conditions = {}

        for cond_name, cond_tensor in conditions.items():
            if cond_name not in self.condition_config:
                continue

            cond_info = self.condition_config[cond_name]

            if cond_info['type'] == 'continuous':
                stats = self.continuous_stats[cond_name]
                normalized_value = float(
                    cond_tensor.cpu().item() if cond_tensor.numel() == 1 else cond_tensor.cpu().mean())
                real_value = (normalized_value * stats['std']) + stats['mean']
                real_conditions[cond_name] = real_value

            elif cond_info['type'] == 'categorical':
                if cond_name in self.categorical_mappings:
                    idx = torch.argmax(cond_tensor).item()
                    mapping = self.categorical_mappings[cond_name]
                    if idx in mapping['idx_to_category']:
                        real_conditions[cond_name] = mapping['idx_to_category'][idx]

        return real_conditions


class DiffusionModelGenerator:
    """
    Main interface for generating samples from trained diffusion models
    """

    def __init__(self, model, condition_processor: ConditionProcessor, device: str = None,
                 num_timesteps: int = 1000, beta_schedule: str = "cosine"):
        self.model = model
        self.condition_processor = condition_processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_timesteps = num_timesteps

        # Initialize diffusion parameters
        self.betas = self._get_beta_schedule(beta_schedule, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # Convert to tensors for efficient computation
        self.betas_tensor = torch.tensor(self.betas, dtype=torch.float32, device=self.device)
        self.alphas_tensor = torch.tensor(self.alphas, dtype=torch.float32, device=self.device)
        self.alphas_cumprod_tensor = torch.tensor(self.alphas_cumprod, dtype=torch.float32, device=self.device)
        self.alphas_cumprod_prev_tensor = torch.tensor(self.alphas_cumprod_prev, dtype=torch.float32,
                                                       device=self.device)

        self.model.eval()
        self.model.to(self.device)

    def _get_beta_schedule(self, schedule: str, timesteps: int):
        """Get beta schedule for diffusion process"""
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

    @classmethod
    def load_model_from_path(cls, directory: str, model_name: str, model_class=None, **model_kwargs):
        """
        Load a trained model and its configuration from specific files
        
        Args:
            directory: Directory containing the model files
            model_name: Base name of the model files (without extensions)
            model_class: Model class to instantiate (if None, tries to auto-detect)
            **model_kwargs: Additional model parameters (will override saved ones)
        """
        directory = Path(directory)
        
        # Define file paths
        model_file = directory / f"{model_name}_model.pth"
        config_file = directory / f"{model_name}_config.json"
        info_file = directory / f"{model_name}_info.json"
        
        try:
            # Load condition processor
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
            condition_processor = ConditionProcessor.load_config(str(config_file))
            
            # Load model info if available
            model_params = {}
            if info_file.exists():
                with open(info_file, 'r') as f:
                    model_info = json.load(f)
                    model_params = model_info.get('model_params', {})
                    
                    # Try to auto-detect model class if not provided
                    if model_class is None:
                        class_name = model_info.get('model_class')
                        if class_name:
                            print(f"Auto-detected model class: {class_name}")
            
            # Override with any user-provided parameters
            model_params.update({k: v for k, v in model_kwargs.items() if v is not None})
            
            if model_class is None:
                raise ValueError("model_class must be provided or saved in model info file")
            
            # Load model
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
                
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            
            # Create model instance
            model = model_class(
                condition_config=condition_processor.condition_config,
                **model_params
            )
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            
            print(f"Successfully loaded model: {model_name}")
            return cls(model, condition_processor)
            
        except Exception as e:
            print(f"Failed to load model '{model_name}' from '{directory}': {str(e)}")
            raise

    @classmethod
    def load_model_from_checkpoint(cls, directory: str, checkpoint_num: int, model_class=None, **model_kwargs):
        """
        Load a model from a checkpoint with standard naming convention
        
        Args:
            directory: Directory containing the checkpoint files
            checkpoint_num: Checkpoint epoch number
            model_class: Model class to instantiate (if None, tries to auto-detect)
            **model_kwargs: Additional model parameters (will override saved ones)
        """
        model_name = f"best_model_epoch_{checkpoint_num}"
        print(f"Loading checkpoint {checkpoint_num} (model name: {model_name})")
        
        return cls.load_model_from_path(directory, model_name, model_class, **model_kwargs)

    def generate_specific_conditions(
            self,
            conditions: Dict[str, Union[str, float, int]],
            num_samples: int = 100,
            guidance_scale: float = 2.0,
            num_inference_steps: int = 50,
            return_conditions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Generate samples with specific condition values"""

        # Prepare condition tensors
        condition_tensors = self.condition_processor.prepare_conditions_for_generation(conditions)

        # Expand to batch size
        batch_condition_tensors = {}
        for cond_name, cond_tensor in condition_tensors.items():
            if cond_tensor.dim() == 2:  # One-hot encoded
                batch_condition_tensors[cond_name] = cond_tensor.repeat(num_samples, 1).to(self.device)
            else:  # Continuous
                batch_condition_tensors[cond_name] = cond_tensor.repeat(num_samples).to(self.device)

        # Generate samples
        with torch.no_grad():
            generated = self._sample_from_model(
                conditions=batch_condition_tensors,
                num_samples=num_samples,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )

        if return_conditions:
            return generated.cpu().numpy(), conditions
        return generated.cpu().numpy()

    def _sample_from_model(
            self,
            conditions: Dict[str, torch.Tensor],
            num_samples: int,
            guidance_scale: float = 2.0,
            num_inference_steps: int = 50
    ):
        """Complete DDPM sampling implementation with classifier-free guidance"""

        # Start with random noise in latent space
        latent_shape = (num_samples, self.model.latent_dim)
        x = torch.randn(latent_shape, device=self.device)

        # Create timestep schedule for inference (can be different from training)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device)

        # Prepare unconditional conditions for classifier-free guidance
        uncond_conditions = {}
        for cond_name, cond_tensor in conditions.items():
            if cond_name in self.condition_processor.condition_config:
                cond_info = self.condition_processor.condition_config[cond_name]
                if cond_info['type'] == 'categorical':
                    # Zero one-hot vector for unconditional
                    uncond_conditions[cond_name] = torch.zeros_like(cond_tensor)
                else:  # continuous
                    # Zero normalized value for unconditional
                    uncond_conditions[cond_name] = torch.zeros_like(cond_tensor)
            else:
                uncond_conditions[cond_name] = torch.zeros_like(cond_tensor)

        #print(uncond_conditions)

        #print(conditions)

        # Denoising loop
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(num_samples)

            # Get model predictions
            if guidance_scale > 1.0:
                # Classifier-free guidance: predict both conditional and unconditional
                # Conditional prediction
                noise_pred_cond = self.model(x, t_batch, conditions, training=False)

                # Unconditional prediction
                noise_pred_uncond = self.model(x, t_batch, uncond_conditions, training=False)

                # Apply classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # Standard conditional prediction
                noise_pred = self.model(x, t_batch, conditions, training=False)

            # DDPM reverse step - Fixed implementation
            if i < len(timesteps) - 1:
                # Not the last step, add noise
                x = self._ddpm_step(x, noise_pred, t_batch, add_noise=True)
            else:
                # Last step, don't add noise
                x = self._ddpm_step(x, noise_pred, t_batch, add_noise=False)

        return x

    def _ddpm_step(self, x_t, noise_pred, t, add_noise=True):
        """
        Standard DDPM denoising step implementation
        Based on Ho et al. "Denoising Diffusion Probabilistic Models" (2020)

        Args:
            x_t: Current noisy sample [batch_size, latent_dim]
            noise_pred: Predicted noise from the model [batch_size, latent_dim]
            t: Current timestep tensor [batch_size]
            add_noise: Whether to add noise (False for final step)

        Returns:
            x_{t-1}: Denoised sample for previous timestep
        """
        batch_size = x_t.shape[0]

        # Ensure t is properly shaped
        if t.dim() == 0:
            t = t.unsqueeze(0).repeat(batch_size)
        elif t.shape[0] != batch_size:
            t = t[0].unsqueeze(0).repeat(batch_size)

        # Extract diffusion parameters for current timestep
        # These should be tensors of shape [batch_size, 1] for broadcasting
        alpha_t = self.alphas_tensor[t].view(batch_size, 1)
        alpha_cumprod_t = self.alphas_cumprod_tensor[t].view(batch_size, 1)
        alpha_cumprod_prev_t = self.alphas_cumprod_prev_tensor[t].view(batch_size, 1)
        beta_t = self.betas_tensor[t].view(batch_size, 1)

        # Compute coefficients for the mean calculation
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)

        # Predict x_0 from current x_t and predicted noise
        # x_0 = (x_t - sqrt(1-ᾱ_t) * ε) / sqrt(ᾱ_t)
        pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t

        # Compute the mean of q(x_{t-1} | x_t, x_0)
        # μ = sqrt(ᾱ_{t-1}) * β_t / (1 - ᾱ_t) * x_0 + sqrt(α_t) * (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * x_t
        coeff1 = torch.sqrt(alpha_cumprod_prev_t) * beta_t / (1.0 - alpha_cumprod_t)
        coeff2 = torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t)
        mean = coeff1 * pred_x0 + coeff2 * x_t

        if not add_noise:
            # For the final step (t=0), return mean without noise
            return mean

        # Create mask for non-zero timesteps
        # Only add noise when t > 0
        nonzero_mask = (t != 0).float().view(batch_size, 1)

        # Compute variance for q(x_{t-1} | x_t, x_0)
        # σ²_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_variance = beta_t * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t)
        posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-20))

        # Sample random noise
        noise = torch.randn_like(x_t)

        # Apply noise only for non-zero timesteps
        # x_{t-1} = μ + σ * ε (where ε ~ N(0,I))
        sample = mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise

        return sample

    def decode_latent_to_gene_expression(self, latent_samples: np.ndarray, vae_model) -> np.ndarray:
        """
        Decode latent samples back to gene expression space using VAE decoder

        Args:
            latent_samples: Generated latent representations [num_samples, latent_dim]
            vae_model: Trained VAE model with decode() method

        Returns:
            Gene expression data [num_samples, num_genes]
        """
        # Convert to tensor and move to device
        latent_tensor = torch.tensor(latent_samples, dtype=torch.float32).to(self.device)

        # Ensure VAE is in eval mode and on correct device
        vae_model.eval()
        vae_model.to(self.device)

        # Decode in batches to avoid memory issues
        batch_size = 100
        decoded_samples = []

        with torch.no_grad():
            for i in range(0, len(latent_tensor), batch_size):
                batch = latent_tensor[i:i + batch_size]

                # Decode batch - assuming VAE has a decode method
                if hasattr(vae_model, 'decode'):
                    decoded_batch = vae_model.decode(batch)
                elif hasattr(vae_model, 'decoder'):
                    decoded_batch = vae_model.decoder(batch)
                else:
                    # Try calling the model directly (some VAEs work this way)
                    decoded_batch = vae_model(batch, encode=False)

                decoded_samples.append(decoded_batch.cpu())

        # Concatenate all batches
        decoded_tensor = torch.cat(decoded_samples, dim=0)

        # Clamping to 0 on onwards
        decoded_tensor = torch.clamp(decoded_tensor, min=0)

        return decoded_tensor.numpy()

    def generate_and_decode(
            self,
            conditions: Dict[str, Union[str, float, int]],
            vae_model,
            num_samples: int = 100,
            guidance_scale: float = 2.0,
            num_inference_steps: int = 50,
            return_conditions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Generate samples and decode them to gene expression space

        Args:
            conditions: Conditions for generation
            vae_model: VAE model for decoding
            num_samples: Number of samples to generate
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            return_conditions: Whether to return conditions along with samples

        Returns:
            Gene expression data, optionally with conditions
        """
        # Generate latent samples
        latent_samples = self.generate_specific_conditions(
            conditions=conditions,
            num_samples=num_samples,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            return_conditions=False
        )

        # Decode to gene expression space
        gene_expression = self.decode_latent_to_gene_expression(latent_samples, vae_model)

        if return_conditions:
            return gene_expression, conditions
        return gene_expression

    def get_available_conditions(self) -> Dict[str, Dict]:
        """Get information about available conditions"""
        available = {}

        for cond_name, cond_info in self.condition_processor.condition_config.items():
            if cond_info['type'] == 'categorical':
                available[cond_name] = {
                    'type': 'categorical',
                    'categories': self.condition_processor.categorical_mappings[cond_name]['categories']
                }
            elif cond_info['type'] == 'continuous':
                stats = self.condition_processor.continuous_stats[cond_name]
                available[cond_name] = {
                    'type': 'continuous',
                    'range': [stats['min'], stats['max']],
                    'mean': stats['mean'],
                    'std': stats['std']
                }

        return available

    def validate_conditions(self, conditions: Dict[str, Union[str, float, int]]) -> bool:
        """Validate that provided conditions are valid"""
        available = self.get_available_conditions()

        for cond_name, cond_value in conditions.items():
            if cond_name not in available:
                print(f"Warning: Condition '{cond_name}' not available")
                return False

            cond_info = available[cond_name]

            if cond_info['type'] == 'categorical':
                if cond_value not in cond_info['categories']:
                    print(f"Warning: Value '{cond_value}' not valid for categorical condition '{cond_name}'")
                    print(f"Available categories: {cond_info['categories']}")
                    return False

            elif cond_info['type'] == 'continuous':
                if not isinstance(cond_value, (int, float)):
                    print(f"Warning: Value '{cond_value}' should be numeric for continuous condition '{cond_name}'")
                    return False

                if cond_value < cond_info['range'][0] or cond_value > cond_info['range'][1]:
                    print(
                        f"Warning: Value '{cond_value}' outside valid range {cond_info['range']} for condition '{cond_name}'")
                    return False

        return True


# Utility functions for saving and loading complete model setups
def save_model_with_config(model, condition_processor: ConditionProcessor, save_dir: str, model_name: str):
    """Save model and configuration together"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = save_dir / f"{model_name}_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }, model_path)

    # Save config
    config_path = save_dir / f"{model_name}_config.json"
    condition_processor.save_config(str(config_path))

    # Save model architecture info
    model_info = {
        'model_class': model.__class__.__name__,
        'model_params': {
            'latent_dim': getattr(model, 'latent_dim', None),
            'hidden_dims': getattr(model, 'hidden_dims', None),
            'dropout': getattr(model, 'dropout', None),
            'classifier_free_prob': getattr(model, 'classifier_free_prob', None),
            'time_embed_dim': getattr(model, 'time_embed_dim', None),
            'cond_embed_dim': getattr(model, 'cond_embed_dim', None),
        }
    }

    # Remove None values
    model_info['model_params'] = {k: v for k, v in model_info['model_params'].items() if v is not None}

    info_path = save_dir / f"{model_name}_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"Model saved to: {save_dir}")
    print(f"Files: {model_name}_model.pth, {model_name}_config.json, {model_name}_info.json")
    return str(save_dir)



def load_model_by_name(directory: Union[str, Path], model_name: str, model_class=None, **model_kwargs):
    """Load model by name from directory"""
    return DiffusionModelGenerator.load_model_from_path(directory, model_name, model_class, **model_kwargs)


def load_model_by_checkpoint(directory: Union[str, Path], checkpoint_num: int, model_class=None, **model_kwargs):
    """Load model from checkpoint number"""
    return DiffusionModelGenerator.load_model_from_checkpoint(directory, checkpoint_num, model_class, **model_kwargs)