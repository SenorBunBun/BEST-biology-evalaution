import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Dict, List, Optional, Union, Tuple
import math
from tqdm import tqdm
import json
import os


class ConditionEmbedder(nn.Module):
    """
    Handles embedding of mixed categorical and continuous conditions from h5ad files.
    """

    def __init__(
            self,
            condition_config: Dict[str, Dict],
            embed_dim: int = 128,
            dropout: float = 0.1
    ):
        super().__init__()
        self.condition_config = condition_config
        self.embed_dim = embed_dim

        # Create embedders for each condition type
        self.embedders = nn.ModuleDict()
        self.condition_dims = {}

        # Store normalization parameters for continuous conditions
        self.continuous_stats = {}

        total_dim = 0
        for cond_name, cond_info in condition_config.items():
            if cond_info['type'] == 'categorical':
                # One-hot + linear projection
                n_categories = cond_info['n_categories']
                self.embedders[cond_name] = nn.Sequential(
                    nn.Linear(n_categories, embed_dim // len(condition_config)),
                    nn.SiLU(),
                    nn.Dropout(dropout)
                )
                self.condition_dims[cond_name] = n_categories
                total_dim += embed_dim // len(condition_config)

            elif cond_info['type'] == 'continuous':
                # Normalize continuous values and project
                self.embedders[cond_name] = nn.Sequential(
                    nn.Linear(1, embed_dim // len(condition_config)),
                    nn.SiLU(),
                    nn.Dropout(dropout)
                )
                self.condition_dims[cond_name] = 1
                total_dim += embed_dim // len(condition_config)

                # Initialize stats for continuous conditions
                self.continuous_stats[cond_name] = {
                    'mean': 0.0,
                    'std': 1.0,
                    'min': 0.0,
                    'max': 1.0
                }

        # Final projection to desired embedding dimension
        self.final_proj = nn.Sequential(
            nn.Linear(total_dim, embed_dim),
            nn.SiLU(),
            nn.LayerNorm(embed_dim)
        )

    def set_continuous_stats(self, stats: Dict[str, Dict[str, float]]):
        """Set normalization statistics for continuous conditions"""
        for cond_name, cond_stats in stats.items():
            if cond_name in self.continuous_stats:
                self.continuous_stats[cond_name].update(cond_stats)

    def forward(self, conditions: Dict[str, torch.Tensor], mask: Optional[Dict[str, bool]] = None):
        """
        Forward pass with optional masking for partial conditioning

        Args:
            conditions: Dict mapping condition names to tensors
            mask: Dict mapping condition names to boolean (True = use condition, False = mask out)
        """
        batch_size = list(conditions.values())[0].shape[0]
        embedded_conditions = []

        for cond_name, cond_info in self.condition_config.items():
            if cond_name not in conditions:
                # If condition not provided, create zero embedding
                embed_dim = self.embed_dim // len(self.condition_config)
                embedded_conditions.append(torch.zeros(batch_size, embed_dim, device=next(self.parameters()).device))
                continue

            cond_tensor = conditions[cond_name]

            # Check if this condition should be masked (for classifier-free guidance)
            should_mask = mask is not None and mask.get(cond_name, False)

            if cond_info['type'] == 'categorical':
                if should_mask:
                    # Create zero one-hot vector for unconditional generation
                    cond_tensor = torch.zeros_like(cond_tensor)
                embedded = self.embedders[cond_name](cond_tensor.float())

            elif cond_info['type'] == 'continuous':
                if should_mask:
                    # Use normalized mean (which should be 0) for unconditional generation
                    cond_tensor = torch.zeros_like(cond_tensor)
                # cond_tensor should already be normalized when passed in
                embedded = self.embedders[cond_name](cond_tensor.unsqueeze(-1).float())

            embedded_conditions.append(embedded)

        # Concatenate all condition embeddings
        combined = torch.cat(embedded_conditions, dim=-1)
        return self.final_proj(combined)


class ResidualBlock(nn.Module):
    """Residual block with time and condition embedding integration"""

    def __init__(self, in_dim: int, out_dim: int, time_embed_dim: int, cond_embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.main_layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_dim)
        )

        # Condition embedding projection
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_embed_dim, out_dim)
        )

        # Residual connection
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self.final_activation = nn.SiLU()

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor, cond_embed: torch.Tensor):
        residual = self.residual_proj(x)

        h = self.main_layers(x)
        h = h + self.time_proj(time_embed) + self.cond_proj(cond_embed)
        h = h + residual

        return self.final_activation(h)


class ClassifierFreeDiffusionModel(nn.Module):
    """
    Classifier-free diffusion model for single-cell data with mixed condition types.
    """

    def __init__(
            self,
            latent_dim: int = 128,
            condition_config: Dict[str, Dict] = None,
            hidden_dims: List[int] = [512, 512, 256, 128], #[512, 512, 256, 128, 256, 512, 512, 1024, 128]
            time_embed_dim: int = 128,
            cond_embed_dim: int = 128,
            dropout: float = 0.1,
            classifier_free_prob: float = 0.1
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.condition_config = condition_config or {}
        self.classifier_free_prob = classifier_free_prob
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.time_embed_dim = time_embed_dim
        self.cond_embed_dim = cond_embed_dim

        # Time embedding (sinusoidal)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Condition embedding
        self.condition_embedder = ConditionEmbedder(
            condition_config,
            embed_dim=cond_embed_dim,
            dropout=dropout
        )

        # U-Net style architecture with skip connections
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dims[0])

        # Encoder (downsampling path)
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims[1:]:
            self.encoder_blocks.append(
                ResidualBlock(in_dim, out_dim, time_embed_dim, cond_embed_dim, dropout)
            )
            in_dim = out_dim

        # Middle block
        self.middle_block = ResidualBlock(
            hidden_dims[-1], hidden_dims[-1], time_embed_dim, cond_embed_dim, dropout
        )

        # Decoder (upsampling path with skip connections)
        for i, out_dim in enumerate(reversed(hidden_dims[:-1])):
            # Skip connection doubles the input dimension
            skip_in_dim = hidden_dims[-(i + 1)] + hidden_dims[-(i + 2)] if i > 0 else hidden_dims[-1] + hidden_dims[-2]
            self.decoder_blocks.append(
                ResidualBlock(skip_in_dim, out_dim, time_embed_dim, cond_embed_dim, dropout)
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[0] + hidden_dims[0], hidden_dims[0]),  # Skip connection from input
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], latent_dim)
        )

    def sinusoidal_timestep_embedding(self, timesteps: torch.Tensor, dim: int):
        """Create sinusoidal timestep embeddings"""
        device = timesteps.device
        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        if dim % 2 == 1:  # zero pad
            embeddings = torch.nn.functional.pad(embeddings, (0, 1, 0, 0))
        return embeddings

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            conditions: Dict[str, torch.Tensor],
            training: bool = True
    ):
        """
        Forward pass with classifier-free guidance support

        Args:
            x: Latent representations [batch_size, latent_dim]
            timesteps: Timestep indices [batch_size]
            conditions: Dict of condition tensors
            training: Whether in training mode (affects classifier-free masking)
        """
        batch_size = x.shape[0]

        # Time embedding
        time_embed = self.sinusoidal_timestep_embedding(timesteps, self.time_embed_dim)
        time_embed = self.time_proj(time_embed)

        # Condition embedding with optional masking for classifier-free guidance
        if training and self.classifier_free_prob > 0:
            # Randomly mask conditions for classifier-free guidance
            mask = {}
            for cond_name in self.condition_config.keys():
                mask[cond_name] = torch.rand(batch_size, device=x.device) < self.classifier_free_prob

            # Apply masking
            masked_conditions = {}
            for cond_name, cond_tensor in conditions.items():
                if cond_name in mask:
                    # Create mask for this condition
                    cond_mask = mask[cond_name]
                    if self.condition_config[cond_name]['type'] == 'categorical':
                        masked_cond = cond_tensor.clone()
                        masked_cond[cond_mask] = 0  # Zero out one-hot for masked samples
                    else:  # continuous
                        masked_cond = cond_tensor.clone()
                        masked_cond[cond_mask] = 0  # Use zero (normalized mean)
                    masked_conditions[cond_name] = masked_cond
                else:
                    masked_conditions[cond_name] = cond_tensor

            cond_embed = self.condition_embedder(masked_conditions)
        else:
            cond_embed = self.condition_embedder(conditions)

        # U-Net forward pass
        h = self.input_proj(x)
        input_h = h

        # Downsampling path (store skip connections)
        skip_connections = [h]
        for block in self.encoder_blocks:
            h = block(h, time_embed, cond_embed)
            skip_connections.append(h)

        # Middle block
        h = self.middle_block(h, time_embed, cond_embed)

        # Upsampling path (with skip connections)
        skip_connections.pop()  # Remove the last skip connection (it's the current h)
        for block in self.decoder_blocks:
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=-1)  # Concatenate skip connection
            h = block(h, time_embed, cond_embed)

        # Output with skip connection from input
        h = torch.cat([h, input_h], dim=-1)
        noise_pred = self.output_proj(h)

        return noise_pred