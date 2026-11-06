"""
Enhanced Vanilla CVAE with proper categorical condition handling
This version ensures consistent encoding of categorical conditions across train/val splits
"""

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt
import time
import json
import os
from sklearn.preprocessing import LabelEncoder
import pickle


# Import the base model from local file
from unbatch_vanilla_cvae_relu_qol import Enhanced_Vanilla_CVAE


class Enhanced_Vanilla_CVAE_Categorical(Enhanced_Vanilla_CVAE):
    """
    Extended CVAE that properly handles categorical conditions
    Stores label encoders for consistent encoding across train/val/test
    """

    def __init__(self, n_genes, n_labels, latent_size, **kwargs):
        # Initialize parent class
        super().__init__(n_genes, n_labels, latent_size, **kwargs)

        # Store for label encoders
        self.label_encoders = {}
        self.condition_cols = kwargs.get('condition_cols', None)

    def fit_label_encoders(self, adata, condition_cols):
        """
        Fit label encoders on the full dataset before train/val split
        This ensures consistent encoding across all data splits

        Args:
            adata: Full AnnData object (before splitting)
            condition_cols: List of condition column names
        """
        print(f"Fitting label encoders for {len(condition_cols)} condition columns...", flush=True)
        self.condition_cols = condition_cols
        self.label_encoders = {}

        for col in condition_cols:
            if adata.obs[col].dtype == 'object' or isinstance(adata.obs[col].iloc[0], str):
                # Categorical column - create and fit encoder
                le = LabelEncoder()
                le.fit(adata.obs[col].values)
                self.label_encoders[col] = le

                print(f"  - {col}: {len(le.classes_)} categories -> {list(le.classes_)}", flush=True)
            else:
                # Numeric column - no encoding needed
                self.label_encoders[col] = None
                print(f"  - {col}: numeric (no encoding)", flush=True)

    def save_label_encoders(self, path=None):
        """Save label encoders to disk"""
        if path is None:
            # Save alongside model
            model_dir = os.path.dirname(self.model)
            path = os.path.join(model_dir, 'label_encoders.pkl')

        with open(path, 'wb') as f:
            pickle.dump({
                'encoders': self.label_encoders,
                'condition_cols': self.condition_cols
            }, f)

        print(f"Label encoders saved to: {path}", flush=True)

    def load_label_encoders(self, path=None):
        """Load label encoders from disk"""
        if path is None:
            model_dir = os.path.dirname(self.model)
            path = os.path.join(model_dir, 'label_encoders.pkl')

        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.label_encoders = data['encoders']
            self.condition_cols = data['condition_cols']

        print(f"Label encoders loaded from: {path}", flush=True)

    def train_net(self, train_loader, n_epochs=100, patience=15,
                  test_loader=None, save_model=True,
                  overfitting_patience=8, min_delta=1e-4,
                  epoch_callback=None, checkpoint_freq=None, checkpoint_dir=None):
        """
        Enhanced training that also saves label encoders

        Args:
            epoch_callback: Optional callback function called after each epoch.
                           Signature: callback(epoch, model, history) -> None
            checkpoint_freq: Save checkpoint every N epochs (optional)
            checkpoint_dir: Directory for checkpoints (required if checkpoint_freq is set)
        """
        # Call parent training method
        history = super().train_net(
            train_loader=train_loader,
            n_epochs=n_epochs,
            patience=patience,
            test_loader=test_loader,
            save_model=save_model,
            overfitting_patience=overfitting_patience,
            min_delta=min_delta,
            epoch_callback=epoch_callback,
            checkpoint_freq=checkpoint_freq,
            checkpoint_dir=checkpoint_dir
        )

        # Save label encoders alongside model
        if save_model and self.label_encoders:
            try:
                self.save_label_encoders()
            except Exception as e:
                print(f"[WARN] Could not save label encoders: {e}", flush=True)

        return history


def create_dataloader_with_encoders(adata, condition_cols, label_encoders, batch_size=32, shuffle=True):
    """
    Create dataloader using pre-fitted label encoders

    Args:
        adata: AnnData object (train or val split)
        condition_cols: List of condition column names
        label_encoders: Dict of pre-fitted LabelEncoders (from model.label_encoders)
        batch_size: Batch size
        shuffle: Whether to shuffle data
    """
    from torch.utils.data import Dataset, DataLoader

    class CVAEDatasetWithEncoders(Dataset):
        def __init__(self, adata, condition_cols, label_encoders):
            # Store expression data
            self.expression_data = adata.X
            self.is_sparse = sparse.issparse(self.expression_data)

            # Encode conditions using provided encoders
            self.conditions = np.zeros((adata.shape[0], len(condition_cols)), dtype=np.float32)
            for i, col in enumerate(condition_cols):
                if label_encoders[col] is not None:
                    # Use pre-fitted encoder
                    self.conditions[:, i] = label_encoders[col].transform(adata.obs[col].values)
                else:
                    # Numeric column
                    self.conditions[:, i] = adata.obs[col].values.astype(np.float32)

            self.condition_cols = condition_cols

        def __len__(self):
            return self.expression_data.shape[0]

        def __getitem__(self, idx):
            # Handle sparse matrices
            if self.is_sparse:
                if isinstance(self.expression_data, sparse.csr_matrix):
                    expression_row = self.expression_data[idx].toarray().flatten()
                else:
                    expression_row = self.expression_data[idx].toarray().flatten()
                expression = torch.tensor(expression_row, dtype=torch.float32)
            else:
                expression = torch.tensor(self.expression_data[idx], dtype=torch.float32)

            condition = torch.tensor(self.conditions[idx, :], dtype=torch.float32)
            return expression, condition

    dataset = CVAEDatasetWithEncoders(adata, condition_cols, label_encoders)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dataloader
