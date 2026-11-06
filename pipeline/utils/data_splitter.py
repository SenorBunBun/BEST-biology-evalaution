"""
Stratified Data Splitting Utility

Provides stratified train/val splitting that ensures proportional representation
of all condition combinations in both sets. Indices are saved to a shared file
so all models (CVAE, Diffusion, GAN) use the exact same split.
"""

import numpy as np
from pathlib import Path


def stratified_split(adata, condition_cols, val_ratio=0.2, seed=42):
    """
    Split data stratified by condition combinations.

    For each unique combination of conditions (e.g., Flight=1, Strain=A, Sex=M),
    randomly assign val_ratio% to validation, rest to training.

    Args:
        adata: AnnData object with obs containing condition columns
        condition_cols: List of column names to stratify by
        val_ratio: Fraction of each group to assign to validation (default: 0.2)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        train_indices, val_indices: numpy arrays of row indices
    """
    # Create combined condition key for each cell
    combined = adata.obs[condition_cols].astype(str).agg('_'.join, axis=1)

    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []

    # Process each unique condition combination
    for group_key in combined.unique():
        group_mask = combined == group_key
        group_indices = np.where(group_mask)[0].copy()

        # Shuffle within group
        rng.shuffle(group_indices)

        # Take val_ratio% for validation (at least 1 sample)
        n_val = max(1, int(len(group_indices) * val_ratio))

        val_indices.extend(group_indices[:n_val])
        train_indices.extend(group_indices[n_val:])

    return np.array(train_indices), np.array(val_indices)


def save_split_indices(path, train_indices, val_indices, metadata=None):
    """
    Save split indices to .npz file.

    Args:
        path: Path to save file (should end in .npz)
        train_indices: Array of training row indices
        val_indices: Array of validation row indices
        metadata: Optional dict of additional data to save
    """
    save_dict = {
        'train_indices': train_indices,
        'val_indices': val_indices
    }
    if metadata:
        save_dict.update(metadata)

    np.savez(path, **save_dict)
    return path


def load_split_indices(path):
    """
    Load pre-computed split indices from .npz file.

    Args:
        path: Path to .npz file

    Returns:
        train_indices, val_indices: numpy arrays
    """
    data = np.load(path)
    return data['train_indices'], data['val_indices']


def print_split_summary(adata, train_indices, val_indices, condition_cols):
    """
    Print summary of stratification results.

    Args:
        adata: AnnData object
        train_indices: Training indices
        val_indices: Validation indices
        condition_cols: Columns used for stratification
    """
    print(f"\n{'='*50}")
    print("STRATIFIED SPLIT SUMMARY")
    print(f"{'='*50}")
    print(f"Total samples: {len(train_indices) + len(val_indices)}")
    print(f"Train samples: {len(train_indices)} ({100*len(train_indices)/(len(train_indices)+len(val_indices)):.1f}%)")
    print(f"Val samples: {len(val_indices)} ({100*len(val_indices)/(len(train_indices)+len(val_indices)):.1f}%)")
    print(f"\nStratified by: {condition_cols}")

    # Show per-group breakdown
    combined = adata.obs[condition_cols].astype(str).agg('_'.join, axis=1)

    print(f"\nPer-group breakdown:")
    print(f"{'Condition Combo':<40} {'Train':>8} {'Val':>8} {'Val%':>8}")
    print("-" * 66)

    for group_key in sorted(combined.unique()):
        group_mask = combined == group_key
        group_indices = np.where(group_mask)[0]

        n_train = len(set(group_indices) & set(train_indices))
        n_val = len(set(group_indices) & set(val_indices))
        val_pct = 100 * n_val / (n_train + n_val) if (n_train + n_val) > 0 else 0

        # Truncate long group names
        display_name = group_key[:38] + ".." if len(group_key) > 40 else group_key
        print(f"{display_name:<40} {n_train:>8} {n_val:>8} {val_pct:>7.1f}%")

    print(f"{'='*50}\n")
