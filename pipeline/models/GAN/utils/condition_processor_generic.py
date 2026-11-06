"""
Generic Condition Processor for multi-conditional scGAN
Handles encoding of arbitrary categorical conditions from CLI specification
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import json
from typing import Dict, List, Tuple


class ConditionProcessorGeneric:
    """
    Processes arbitrary categorical conditions for multi-conditional GAN.
    Supports any set of categorical columns specified via condition_names.
    """

    def __init__(self, condition_names: List[str] = None):
        self.condition_names = condition_names or []
        self.categorical_mappings = {}
        self.condition_dims = {}
        self.total_condition_dim = 0
        self.is_fitted = False

    def fit(self, adata):
        """
        Fit the processor to the data to learn categories

        Args:
            adata: AnnData object with conditions in .obs
        """
        print(f"Fitting generic condition processor...")
        print(f"  Condition columns: {self.condition_names}")

        for col in self.condition_names:
            if col not in adata.obs.columns:
                raise ValueError(f"Column '{col}' not found in adata.obs. "
                               f"Available: {list(adata.obs.columns)}")

            values = adata.obs[col].astype(str).values
            unique_values = sorted(np.unique(values))

            self.categorical_mappings[col] = {
                'categories': list(unique_values),
                'cat_to_idx': {cat: idx for idx, cat in enumerate(unique_values)},
                'idx_to_cat': {idx: cat for idx, cat in enumerate(unique_values)}
            }
            self.condition_dims[col] = len(unique_values)
            print(f"  {col}: {len(unique_values)} categories")

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
            raise ValueError("ConditionProcessorGeneric must be fitted before encoding")

        encoded_parts = []

        for col in self.condition_names:
            mapping = self.categorical_mappings[col]
            values = obs_df[col].astype(str).values
            indices = [mapping['cat_to_idx'][val] for val in values]
            one_hot = tf.one_hot(indices, len(mapping['categories']))
            encoded_parts.append(one_hot)

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
            raise ValueError("ConditionProcessorGeneric must be fitted before decoding")

        condition_array = condition_tensor.numpy() if hasattr(condition_tensor, 'numpy') else condition_tensor
        n_samples = condition_array.shape[0]
        decoded_conditions = []

        for i in range(n_samples):
            cond_vec = condition_array[i]
            idx = 0
            decoded = {}

            for col in self.condition_names:
                dim = self.condition_dims[col]
                one_hot = cond_vec[idx:idx + dim]
                cat_idx = np.argmax(one_hot)
                decoded[col] = self.categorical_mappings[col]['idx_to_cat'][cat_idx]
                idx += dim

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
        grouped = obs_df[self.condition_names].value_counts()

        condition_dicts = []
        counts = []

        for combo, count in grouped.items():
            cond_dict = {}
            for i, cond_name in enumerate(self.condition_names):
                cond_dict[cond_name] = combo[i] if isinstance(combo, tuple) else combo
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
        combos, counts = self.get_condition_combinations(obs_df)
        probs = np.array(counts) / np.sum(counts)
        sampled_indices = np.random.choice(len(combos), size=n_samples, p=probs)

        sampled_conditions = [combos[idx] for idx in sampled_indices]
        sampled_df = pd.DataFrame(sampled_conditions)

        return self.encode_conditions(sampled_df)

    def sample_specific_conditions(self, condition_dict: Dict, n_samples: int) -> tf.Tensor:
        """
        Sample conditions with specific values

        Args:
            condition_dict: Dictionary with condition values
            n_samples: Number of samples to generate

        Returns:
            Tensor of encoded conditions
        """
        if not self.is_fitted:
            raise ValueError("ConditionProcessorGeneric must be fitted before sampling")

        for col in self.condition_names:
            if col not in condition_dict:
                raise ValueError(f"Missing condition: {col}")
            val = str(condition_dict[col])
            if val not in self.categorical_mappings[col]['cat_to_idx']:
                raise ValueError(f"Unknown value for {col}: {val}")

        sampled_df = pd.DataFrame([condition_dict] * n_samples)
        return self.encode_conditions(sampled_df)

    def save_config(self, filepath: str):
        """Save processor configuration to JSON"""
        config = {
            'condition_names': self.condition_names,
            'categorical_mappings': self.categorical_mappings,
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
        self.condition_dims = config['condition_dims']
        self.total_condition_dim = config['total_condition_dim']
        self.is_fitted = config['is_fitted']
        print(f"Condition processor config loaded from {filepath}")
