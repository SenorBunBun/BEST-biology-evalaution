"""
Maximum Mean Discrepancy (MMD) Computation Utility

PyTorch implementation using efficient distance computation.
The sigma (bandwidth) is computed once from real validation data using the
median heuristic and reused for all subsequent measurements.
"""

import numpy as np
import torch


def compute_sigma_from_real_data(real_data, max_samples=1000):
    """
    Compute sigma using median heuristic from real validation data.

    Args:
        real_data: numpy array or torch tensor (n_samples, n_features)
        max_samples: Max samples to use for distance computation

    Returns:
        sigma: float - Gaussian kernel bandwidth
    """
    if isinstance(real_data, np.ndarray):
        real_data = torch.from_numpy(real_data).float()

    # Subsample for memory efficiency
    n = min(max_samples, len(real_data))
    if n < len(real_data):
        idx = torch.randperm(len(real_data))[:n]
        subset = real_data[idx]
    else:
        subset = real_data

    # Efficient pairwise distance using cdist
    dists = torch.cdist(subset, subset)
    sigma = torch.median(dists[dists > 0]).item()

    return sigma


def compute_sigma_stratified(train_data, condition_labels, max_total_samples=1000, seed=42):
    """
    Compute sigma using median heuristic with stratified sampling from training data.

    Samples proportionally from each condition group to ensure representation.

    Args:
        train_data: numpy array or torch tensor (n_samples, n_features)
        condition_labels: numpy array of condition labels (n_samples,)
        max_total_samples: Max total samples to use for distance computation
        seed: Random seed for reproducibility

    Returns:
        (sigma, sampled_indices): bandwidth float and indices used
    """
    if isinstance(train_data, torch.Tensor):
        train_data = train_data.numpy()

    rng = np.random.RandomState(seed)
    n_total = len(train_data)
    unique_labels = np.unique(condition_labels)

    # Proportional sampling per group
    sampled_indices = []
    for label in unique_labels:
        group_idx = np.where(condition_labels == label)[0]
        proportion = len(group_idx) / n_total
        n_sample = max(1, int(proportion * max_total_samples))
        n_sample = min(n_sample, len(group_idx))
        chosen = rng.choice(group_idx, size=n_sample, replace=False)
        sampled_indices.append(chosen)

    sampled_indices = np.concatenate(sampled_indices)
    subset = torch.from_numpy(train_data[sampled_indices]).float()

    # Compute median heuristic
    dists = torch.cdist(subset, subset)
    sigma = torch.median(dists[dists > 0]).item()

    return sigma, sampled_indices


def gaussian_kernel_matrix(x, y, sigma):
    """
    Compute Gaussian kernel matrix efficiently.

    Uses ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x·y to avoid (n,m,d) intermediate.

    Args:
        x: (n, d) tensor
        y: (m, d) tensor
        sigma: bandwidth

    Returns:
        (n, m) kernel matrix
    """
    # ||x||^2 and ||y||^2
    x_sqnorm = (x ** 2).sum(dim=1, keepdim=True)  # (n, 1)
    y_sqnorm = (y ** 2).sum(dim=1, keepdim=True)  # (m, 1)

    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x·y
    dist_sq = x_sqnorm + y_sqnorm.t() - 2 * torch.mm(x, y.t())

    # Gaussian kernel
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def compute_mmd(real_data, generated_data, sigma, max_samples=5000):
    """
    Compute Maximum Mean Discrepancy between real and generated data.

    Args:
        real_data: numpy array or torch tensor (n_real, n_features)
        generated_data: numpy array or torch tensor (n_gen, n_features)
        sigma: Gaussian kernel bandwidth (pre-computed from real data)
        max_samples: Maximum samples to use (subsamples if larger)

    Returns:
        mmd_value: float
    """
    if isinstance(real_data, np.ndarray):
        real_data = torch.from_numpy(real_data).float()
    if isinstance(generated_data, np.ndarray):
        generated_data = torch.from_numpy(generated_data).float()

    n_real, n_gen = len(real_data), len(generated_data)

    # Subsample if too large
    if n_real > max_samples:
        idx = torch.randperm(n_real)[:max_samples]
        real_data = real_data[idx]
        n_real = max_samples

    if n_gen > max_samples:
        idx = torch.randperm(n_gen)[:max_samples]
        generated_data = generated_data[idx]
        n_gen = max_samples

    # Compute kernel matrices
    k_xx = gaussian_kernel_matrix(real_data, real_data, sigma)
    k_yy = gaussian_kernel_matrix(generated_data, generated_data, sigma)
    k_xy = gaussian_kernel_matrix(real_data, generated_data, sigma)

    # Unbiased MMD^2 estimator
    mmd_sq = (k_xx.sum() - k_xx.trace()) / (n_real * (n_real - 1)) + \
             (k_yy.sum() - k_yy.trace()) / (n_gen * (n_gen - 1)) - \
             2 * k_xy.mean()

    return float(torch.sqrt(torch.clamp(mmd_sq, min=0)))
