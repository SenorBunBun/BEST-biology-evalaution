"""
Condition processor for cancer multi-conditional scGAN
Handles encoding of categorical conditions for cancer data
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import json
from typing import Dict, List, Union, Tuple, Optional


class ConditionProcessorCancer:
    """
    Processes conditions for cancer multi-conditional GAN:
    - annotation_major: Categorical (one-hot encoded)
    - tissue: Categorical (one-hot encoded)
    """

    def __init__(self):
        self.condition_names = ['annotation_major', 'tissue']
        self.categorical_mappings = {}
        self.condition_dims = {}
        self.total_condition_dim = 0
        self.is_fitted = False

    def fit(self, adata):
        """
        Fit the processor to the data to learn categories and statistics

        Args:
            adata: AnnData object with conditions in .obs
        """
        print("Fitting condition processor for cancer data...")

        # Process annotation_major (categorical)
        annotation_values = adata.obs['annotation_major'].values
        unique_annotations = np.unique(annotation_values)
        self.categorical_mappings['annotation_major'] = {
            'categories': list(unique_annotations),
            'cat_to_idx': {cat: idx for idx, cat in enumerate(unique_annotations)},
            'idx_to_cat': {idx: cat for idx, cat in enumerate(unique_annotations)}
        }
        self.condition_dims['annotation_major'] = len(unique_annotations)
        print(f"  annotation_major: {len(unique_annotations)} categories")
        print(f"    Categories: {list(unique_annotations)}")

        # Process tissue (categorical)
        tissue_values = adata.obs['tissue'].values
        unique_tissues = np.unique(tissue_values)
        self.categorical_mappings['tissue'] = {
            'categories': list(unique_tissues),
            'cat_to_idx': {cat: idx for idx, cat in enumerate(unique_tissues)},
            'idx_to_cat': {idx: cat for idx, cat in enumerate(unique_tissues)}
        }
        self.condition_dims['tissue'] = len(unique_tissues)
        print(f"  tissue: {len(unique_tissues)} categories")
        print(f"    Categories: {list(unique_tissues)}")

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
            raise ValueError("ConditionProcessorCancer must be fitted before encoding")

        encoded_parts = []

        # Encode annotation_major (one-hot)
        annotation_mapping = self.categorical_mappings['annotation_major']
        annotation_indices = [annotation_mapping['cat_to_idx'][val]
                             for val in obs_df['annotation_major'].values]
        annotation_one_hot = tf.one_hot(annotation_indices, len(annotation_mapping['categories']))
        encoded_parts.append(annotation_one_hot)

        # Encode tissue (one-hot)
        tissue_mapping = self.categorical_mappings['tissue']
        tissue_indices = [tissue_mapping['cat_to_idx'][val]
                         for val in obs_df['tissue'].values]
        tissue_one_hot = tf.one_hot(tissue_indices, len(tissue_mapping['categories']))
        encoded_parts.append(tissue_one_hot)

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
            raise ValueError("ConditionProcessorCancer must be fitted before decoding")

        condition_array = condition_tensor.numpy() if hasattr(condition_tensor, 'numpy') else condition_tensor
        n_samples = condition_array.shape[0]
        decoded_conditions = []

        for i in range(n_samples):
            cond_vec = condition_array[i]
            idx = 0
            decoded = {}

            # Decode annotation_major
            annotation_dim = self.condition_dims['annotation_major']
            annotation_one_hot = cond_vec[idx:idx + annotation_dim]
            annotation_idx = np.argmax(annotation_one_hot)
            decoded['annotation_major'] = self.categorical_mappings['annotation_major']['idx_to_cat'][annotation_idx]
            idx += annotation_dim

            # Decode tissue
            tissue_dim = self.condition_dims['tissue']
            tissue_one_hot = cond_vec[idx:idx + tissue_dim]
            tissue_idx = np.argmax(tissue_one_hot)
            decoded['tissue'] = self.categorical_mappings['tissue']['idx_to_cat'][tissue_idx]
            idx += tissue_dim

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

    def sample_specific_conditions(self, annotation_major: str, tissue: str, n_samples: int) -> tf.Tensor:
        """
        Sample conditions with specific values

        Args:
            annotation_major: Specific annotation_major value
            tissue: Specific tissue value
            n_samples: Number of samples to generate

        Returns:
            Tensor of encoded conditions
        """
        if not self.is_fitted:
            raise ValueError("ConditionProcessorCancer must be fitted before sampling")

        # Validate inputs
        if annotation_major not in self.categorical_mappings['annotation_major']['cat_to_idx']:
            raise ValueError(f"Unknown annotation_major: {annotation_major}")
        if tissue not in self.categorical_mappings['tissue']['cat_to_idx']:
            raise ValueError(f"Unknown tissue: {tissue}")

        # Create dataframe with repeated conditions
        sampled_df = pd.DataFrame({
            'annotation_major': [annotation_major] * n_samples,
            'tissue': [tissue] * n_samples
        })

        # Encode and return
        return self.encode_conditions(sampled_df)

    def get_condition_info(self) -> Dict:
        """
        Get information about available conditions

        Returns:
            Dictionary with condition information
        """
        if not self.is_fitted:
            raise ValueError("ConditionProcessorCancer must be fitted before getting info")

        info = {
            'annotation_major': {
                'type': 'categorical',
                'n_categories': len(self.categorical_mappings['annotation_major']['categories']),
                'categories': self.categorical_mappings['annotation_major']['categories'],
                'dimension': self.condition_dims['annotation_major']
            },
            'tissue': {
                'type': 'categorical',
                'n_categories': len(self.categorical_mappings['tissue']['categories']),
                'categories': self.categorical_mappings['tissue']['categories'],
                'dimension': self.condition_dims['tissue']
            },
            'total_dimension': self.total_condition_dim
        }
        return info

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
