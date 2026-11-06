"""
Condition processor for multi-conditional scGAN
Handles normalization and encoding of mixed categorical and continuous conditions
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import json
from typing import Dict, List, Union, Tuple, Optional


class ConditionProcessor:
    """
    Processes conditions for multi-conditional GAN:
    - Strain: Categorical (one-hot encoded)
    - Age at Launch: Continuous (normalized to [-1, 1])
    - Duration: Continuous (normalized to [-1, 1])
    - Flight: Binary (0 or 1)
    """

    def __init__(self):
        self.condition_names = ['Strain', 'Age at Launch', 'Duration', 'Flight']
        self.categorical_mappings = {}
        self.continuous_stats = {}
        self.condition_dims = {}
        self.total_condition_dim = 0
        self.is_fitted = False

    def fit(self, adata):
        """
        Fit the processor to the data to learn categories and statistics

        Args:
            adata: AnnData object with conditions in .obs
        """
        print("Fitting condition processor...")

        # Process Strain (categorical)
        strain_values = adata.obs['Strain'].values
        unique_strains = np.unique(strain_values)
        self.categorical_mappings['Strain'] = {
            'categories': list(unique_strains),
            'cat_to_idx': {cat: idx for idx, cat in enumerate(unique_strains)},
            'idx_to_cat': {idx: cat for idx, cat in enumerate(unique_strains)}
        }
        self.condition_dims['Strain'] = len(unique_strains)
        print(f"  Strain: {len(unique_strains)} categories")

        # Process Age at Launch (continuous)
        age_values = adata.obs['Age at Launch'].values.astype(float)
        self.continuous_stats['Age at Launch'] = {
            'min': float(np.min(age_values)),
            'max': float(np.max(age_values)),
            'mean': float(np.mean(age_values)),
            'std': float(np.std(age_values))
        }
        self.condition_dims['Age at Launch'] = 1
        print(f"  Age at Launch: range [{self.continuous_stats['Age at Launch']['min']}, "
              f"{self.continuous_stats['Age at Launch']['max']}]")

        # Process Duration (continuous)
        duration_values = adata.obs['Duration'].values.astype(float)
        self.continuous_stats['Duration'] = {
            'min': float(np.min(duration_values)),
            'max': float(np.max(duration_values)),
            'mean': float(np.mean(duration_values)),
            'std': float(np.std(duration_values))
        }
        self.condition_dims['Duration'] = 1
        print(f"  Duration: range [{self.continuous_stats['Duration']['min']}, "
              f"{self.continuous_stats['Duration']['max']}]")

        # Process Flight (binary - treat as continuous)
        flight_values = adata.obs['Flight'].values.astype(float)
        self.continuous_stats['Flight'] = {
            'min': 0.0,
            'max': 1.0,
            'mean': float(np.mean(flight_values)),
            'std': float(np.std(flight_values))
        }
        self.condition_dims['Flight'] = 1
        print(f"  Flight: binary (0=ground, 1=flight)")

        # Calculate total dimension
        self.total_condition_dim = sum(self.condition_dims.values())
        print(f"Total condition dimension: {self.total_condition_dim}")

        self.is_fitted = True

    def encode_conditions(self, obs_df) -> tf.Tensor:
        """
        Encode conditions from dataframe to tensor

        Args:
            obs_df: DataFrame with condition columns

        Returns:
            Tensor of shape (n_samples, total_condition_dim)
        """
        if not self.is_fitted:
            raise ValueError("ConditionProcessor must be fitted before encoding")

        encoded_parts = []

        # Encode Strain (one-hot)
        strain_mapping = self.categorical_mappings['Strain']
        strain_indices = [strain_mapping['cat_to_idx'][val]
                          for val in obs_df['Strain'].values]
        strain_one_hot = tf.one_hot(strain_indices, len(strain_mapping['categories']))
        encoded_parts.append(strain_one_hot)

        # Encode Age at Launch (normalize to [-1, 1])
        age_stats = self.continuous_stats['Age at Launch']
        age_values = obs_df['Age at Launch'].values.astype(np.float32)
        # Handle case where all values are the same
        if age_stats['max'] == age_stats['min']:
            age_norm = np.zeros_like(age_values)  # All same value -> 0
        else:
            age_norm = 2.0 * (age_values - age_stats['min']) / (age_stats['max'] - age_stats['min']) - 1.0
        encoded_parts.append(tf.expand_dims(tf.constant(age_norm), axis=1))

        # Encode Duration (normalize to [-1, 1])
        dur_stats = self.continuous_stats['Duration']
        dur_values = obs_df['Duration'].values.astype(np.float32)
        # Handle case where all values are the same
        if dur_stats['max'] == dur_stats['min']:
            dur_norm = np.zeros_like(dur_values)  # All same value -> 0
        else:
            dur_norm = 2.0 * (dur_values - dur_stats['min']) / (dur_stats['max'] - dur_stats['min']) - 1.0
        encoded_parts.append(tf.expand_dims(tf.constant(dur_norm), axis=1))

        # Encode Flight (already 0/1, convert to [-1, 1])
        flight_values = obs_df['Flight'].values.astype(np.float32)
        flight_norm = 2.0 * flight_values - 1.0  # 0 -> -1, 1 -> 1
        encoded_parts.append(tf.expand_dims(tf.constant(flight_norm), axis=1))

        # Concatenate all conditions
        conditions = tf.concat(encoded_parts, axis=1)
        return conditions

    def decode_conditions(self, condition_tensor) -> List[Dict]:
        """
        Decode condition tensor back to human-readable format

        Args:
            condition_tensor: Tensor of shape (n_samples, total_condition_dim)

        Returns:
            List of dictionaries with decoded conditions
        """
        if not self.is_fitted:
            raise ValueError("ConditionProcessor must be fitted before decoding")

        condition_array = condition_tensor.numpy() if hasattr(condition_tensor, 'numpy') else condition_tensor
        n_samples = condition_array.shape[0]
        decoded_conditions = []

        for i in range(n_samples):
            cond_vec = condition_array[i]
            idx = 0
            decoded = {}

            # Decode Strain
            strain_dim = self.condition_dims['Strain']
            strain_one_hot = cond_vec[idx:idx + strain_dim]
            strain_idx = np.argmax(strain_one_hot)
            decoded['Strain'] = self.categorical_mappings['Strain']['idx_to_cat'][strain_idx]
            idx += strain_dim

            # Decode Age at Launch
            age_stats = self.continuous_stats['Age at Launch']
            age_norm = cond_vec[idx]
            age_val = (age_norm + 1.0) / 2.0 * (age_stats['max'] - age_stats['min']) + age_stats['min']
            decoded['Age at Launch'] = float(age_val)
            idx += 1

            # Decode Duration
            dur_stats = self.continuous_stats['Duration']
            dur_norm = cond_vec[idx]
            dur_val = (dur_norm + 1.0) / 2.0 * (dur_stats['max'] - dur_stats['min']) + dur_stats['min']
            decoded['Duration'] = float(dur_val)
            idx += 1

            # Decode Flight
            flight_norm = cond_vec[idx]
            flight_val = (flight_norm + 1.0) / 2.0  # -1 -> 0, 1 -> 1
            decoded['Flight'] = 1 if flight_val > 0.5 else 0
            idx += 1

            decoded_conditions.append(decoded)

        return decoded_conditions

    def get_condition_combinations(self, obs_df) -> Tuple[List[Dict], List[int]]:
        """
        Get unique condition combinations and their counts

        Args:
            obs_df: DataFrame with condition columns

        Returns:
            Tuple of (list of condition dicts, list of counts)
        """
        # Group by all conditions
        grouped = obs_df[self.condition_names].value_counts()

        condition_dicts = []
        counts = []

        for combo, count in grouped.items():
            cond_dict = {}
            for i, cond_name in enumerate(self.condition_names):
                cond_dict[cond_name] = combo[i]
            condition_dicts.append(cond_dict)
            counts.append(count)

        return condition_dicts, counts

    def sample_conditions_matching_distribution(self, obs_df, n_samples: int) -> tf.Tensor:
        """
        Sample conditions matching the distribution in the data

        Args:
            obs_df: DataFrame with condition columns
            n_samples: Number of samples to generate

        Returns:
            Tensor of encoded conditions
        """
        # Get combinations and their probabilities
        combos, counts = self.get_condition_combinations(obs_df)
        probs = np.array(counts) / np.sum(counts)

        # Sample combinations
        sampled_indices = np.random.choice(len(combos), size=n_samples, p=probs)

        # Create dataframe with sampled conditions
        sampled_conditions = []
        for idx in sampled_indices:
            sampled_conditions.append(combos[idx])

        sampled_df = pd.DataFrame(sampled_conditions)

        # Encode and return
        return self.encode_conditions(sampled_df)

    def save_config(self, filepath: str):
        """Save processor configuration to JSON"""
        config = {
            'condition_names': self.condition_names,
            'categorical_mappings': self.categorical_mappings,
            'continuous_stats': self.continuous_stats,
            'condition_dims': self.condition_dims,
            'total_condition_dim': self.total_condition_dim,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Condition processor config saved to {filepath}")

    def load_config(self, filepath: str):
        """Load processor configuration from JSON"""
        with open(filepath, 'r') as f:
            config = json.load(f)

        self.condition_names = config['condition_names']
        self.categorical_mappings = config['categorical_mappings']
        self.continuous_stats = config['continuous_stats']
        self.condition_dims = config['condition_dims']
        self.total_condition_dim = config['total_condition_dim']
        self.is_fitted = config['is_fitted']
        print(f"Condition processor config loaded from {filepath}")