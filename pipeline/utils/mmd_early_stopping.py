"""
MMD-Based Early Stopping Tracker

Shared utility for MMD-based early stopping across CVAE, Diffusion, and GAN models.
Also provides sigma heuristic save/load for pre-computed bandwidth values.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class MMDEarlyStoppingTracker:
    """Tracks MMD values during training and determines when to stop."""

    def __init__(self, model_type: str, dataset_name: str, sigma: float,
                 eval_epoch_interval: int, patience_mmd: int = 3,
                 min_delta: float = 0.001, save_dir: Optional[str] = None):
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.sigma = sigma
        self.eval_epoch_interval = eval_epoch_interval
        self.patience_mmd = patience_mmd
        self.min_delta = min_delta
        self.save_dir = Path(save_dir) if save_dir else None

        self.best_mmd = float('inf')
        self.best_epoch = -1
        self.patience_counter = 0
        self.stopped_epoch = None
        self.stop_reason = None
        self.history = []

    def should_evaluate(self, epoch: int) -> bool:
        """Check if MMD should be evaluated at this epoch (0-indexed)."""
        return (epoch + 1) % self.eval_epoch_interval == 0

    def record(self, epoch: int, mmd_value: float, loss_value: Optional[float] = None,
               checkpoint_filename: Optional[str] = None) -> Tuple[bool, bool]:
        """
        Record an MMD measurement and check stopping criteria.

        Args:
            epoch: Current epoch (0-indexed)
            mmd_value: Computed MMD value
            loss_value: Optional loss value for logging
            checkpoint_filename: Optional checkpoint filename

        Returns:
            (is_best, should_stop)
        """
        is_best = mmd_value < (self.best_mmd - self.min_delta)

        if is_best:
            self.best_mmd = mmd_value
            self.best_epoch = epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        should_stop = self.patience_counter >= self.patience_mmd

        entry = {
            'epoch': epoch,
            'mmd': float(mmd_value),
            'is_best': is_best
        }
        if loss_value is not None:
            entry['loss'] = float(loss_value)
        if checkpoint_filename is not None:
            entry['checkpoint'] = checkpoint_filename

        self.history.append(entry)

        if should_stop:
            self.stopped_epoch = epoch
            self.stop_reason = 'mmd_patience_exceeded'

        print(f"[MMD] Epoch {epoch + 1}: {mmd_value:.6f}"
              f"{' (best)' if is_best else ''}"
              f" | patience: {self.patience_counter}/{self.patience_mmd}")

        return is_best, should_stop

    def save_history(self, stop_reason_override: Optional[str] = None) -> Optional[Path]:
        """Save MMD history to concurrent_mmd_history.json."""
        if self.save_dir is None:
            return None

        if stop_reason_override:
            self.stop_reason = stop_reason_override

        if self.stop_reason is None:
            self.stop_reason = 'max_epochs_reached'

        output = {
            'model_type': self.model_type,
            'dataset': self.dataset_name,
            'sigma': float(self.sigma),
            'best_epoch': int(self.best_epoch) if self.best_epoch >= 0 else None,
            'stopped_epoch': int(self.stopped_epoch) if self.stopped_epoch is not None else None,
            'stop_reason': self.stop_reason,
            'history': self.history
        }

        output_path = self.save_dir / 'concurrent_mmd_history.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"[OK] MMD history saved to: {output_path}")
        return output_path


def save_sigma_heuristic(path: str, sigma: float, sampled_indices: np.ndarray,
                         sampled_data: np.ndarray, sampled_conditions: np.ndarray,
                         seed: int, max_total_samples: int):
    """Save pre-computed sigma heuristic and metadata."""
    np.savez(
        path,
        sigma=np.array([sigma]),
        sampled_indices=sampled_indices,
        sampled_data=sampled_data,
        sampled_conditions=sampled_conditions,
        seed=np.array([seed]),
        max_total_samples=np.array([max_total_samples])
    )


def load_sigma_heuristic(path: str) -> float:
    """Load pre-computed sigma from file."""
    data = np.load(path, allow_pickle=True)
    return float(data['sigma'][0])
