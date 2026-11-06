"""
Model tracker for managing trained model checkpoints and latest_model.json files.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt


class ModelTracker:
    """
    Tracks trained models, manages checkpoints, and maintains latest_model.json.

    For each model type (cvae, diffusion, gan), maintains:
    - Timestamped run directories with checkpoints
    - latest_model.json pointing to the best checkpoint of the latest run
    - Training metrics and loss curves
    """

    def __init__(self, model_dir: Path, model_type: str):
        """
        Initialize model tracker.

        Args:
            model_dir: Base directory for this model type (e.g., outputs/cancer/trained_models/cvae)
            model_type: Model type identifier (cvae, diffusion, gan)
        """
        self.model_dir = Path(model_dir)
        self.model_type = model_type
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.latest_model_path = self.model_dir / "latest_model.json"

    def create_run_dir(self) -> Path:
        """Create a new timestamped run directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.model_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create checkpoints subdirectory
        (run_dir / "checkpoints").mkdir(exist_ok=True)

        return run_dir

    def save_checkpoint(self, run_dir: Path, epoch: int, model_state: Any,
                        train_loss: float, val_loss: float,
                        save_fn: callable) -> str:
        """
        Save a checkpoint for an epoch.

        Args:
            run_dir: Run directory path
            epoch: Epoch number
            model_state: Model state to save (passed to save_fn)
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
            save_fn: Function to save model state (receives path and state)

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"

        # Use provided save function to save model state
        save_fn(checkpoint_path, model_state)

        return str(checkpoint_path)

    def save_training_metrics(self, run_dir: Path, metrics: Dict) -> None:
        """
        Save training metrics to JSON file.

        Args:
            run_dir: Run directory path
            metrics: Dictionary containing:
                - best_epoch: int
                - best_val_loss: float
                - total_epochs: int
                - early_stopped: bool
                - epoch_losses: List[Dict] with epoch, train_loss, val_loss, checkpoint
        """
        metrics_path = run_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def save_loss_curves(self, run_dir: Path, epoch_losses: List[Dict],
                         x_key: str = "epoch", y_keys: List[str] = None,
                         show: bool = False) -> None:
        """
        Generate and save loss curve plot.

        Args:
            run_dir: Run directory path
            epoch_losses: List of dicts with loss values
            x_key: Key for x-axis (default: "epoch")
            y_keys: Keys for y-axis lines (default: ["train_loss", "val_loss"])
            show: Whether to display the plot with plt.show() (default: False)
        """
        if not epoch_losses:
            return

        if y_keys is None:
            y_keys = ["train_loss", "val_loss"]

        x_values = [e[x_key] for e in epoch_losses]

        colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']
        plt.figure(figsize=(10, 6))

        for i, y_key in enumerate(y_keys):
            if y_key in epoch_losses[0]:
                y_values = [e[y_key] for e in epoch_losses]
                color = colors[i % len(colors)]
                label = y_key.replace('_', ' ').title()
                plt.plot(x_values, y_values, '-', color=color, label=label, linewidth=2)

        # Try to mark best point (if val_loss or first y_key exists)
        marker_key = "val_loss" if "val_loss" in epoch_losses[0] else y_keys[0]
        if marker_key in epoch_losses[0]:
            y_values = [e[marker_key] for e in epoch_losses]
            best_idx = y_values.index(min(y_values))
            plt.axvline(x=x_values[best_idx], color='g', linestyle='--',
                        label=f'Best ({x_values[best_idx]})', alpha=0.7)
            plt.scatter([x_values[best_idx]], [y_values[best_idx]], color='g', s=100, zorder=5)

        x_label = x_key.replace('_', ' ').title()
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'{self.model_type.upper()} Training Loss Curves', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(run_dir / "loss_curves.png", dpi=150)
        if show:
            plt.show()
        plt.close()

    def save_best_model(self, run_dir: Path, best_checkpoint: str) -> str:
        """
        Copy best checkpoint to best_model.pt in run directory.

        Args:
            run_dir: Run directory path
            best_checkpoint: Path to best checkpoint file

        Returns:
            Path to best_model.pt
        """
        best_model_path = run_dir / "best_model.pt"
        shutil.copy2(best_checkpoint, best_model_path)
        return str(best_model_path)

    def save_run_config(self, run_dir: Path, config: Dict) -> None:
        """Save the config snapshot for this run."""
        config_path = run_dir / "run_config.json"
        # Remove internal fields
        config_copy = {k: v for k, v in config.items() if not k.startswith("_")}
        with open(config_path, 'w') as f:
            json.dump(config_copy, f, indent=2, default=str)

    def update_latest_model(self, run_dir: Path, best_epoch: int,
                            best_val_loss: float, config: Dict,
                            additional_paths: Optional[Dict[str, str]] = None) -> None:
        """
        Update latest_model.json to point to this run's best model.

        Args:
            run_dir: Run directory path
            best_epoch: Best epoch number
            best_val_loss: Best validation loss
            config: Model config used for this run
            additional_paths: Optional dict of additional model paths to include
                              (e.g., {"vae_model_path": "run_xxx/vae_model.pt"})
        """
        run_name = run_dir.name
        latest_info = {
            "run_dir": run_name,
            "best_model_path": f"{run_name}/best_model.pt",
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config_used": config.get("models", {}).get(self.model_type, {})
        }

        # Add any additional paths (e.g., VAE model for diffusion)
        if additional_paths:
            latest_info.update(additional_paths)

        with open(self.latest_model_path, 'w') as f:
            json.dump(latest_info, f, indent=2)

    def get_latest_model_info(self) -> Optional[Dict]:
        """
        Get info about the latest trained model.

        Returns:
            Dict with run_dir, best_model_path, best_epoch, etc. or None if no model exists
        """
        if not self.latest_model_path.exists():
            return None

        with open(self.latest_model_path, 'r') as f:
            return json.load(f)

    def get_model_path(self, checkpoint_spec: str) -> Optional[Path]:
        """
        Resolve a checkpoint specification to an actual path.

        Args:
            checkpoint_spec: Either "latest" or a specific path

        Returns:
            Absolute path to the model file, or None if not found
        """
        if checkpoint_spec == "latest":
            info = self.get_latest_model_info()
            if info is None:
                return None
            return self.model_dir / info["best_model_path"]
        else:
            # Specific path provided
            path = Path(checkpoint_spec)
            if path.is_absolute():
                return path if path.exists() else None
            else:
                # Relative to model_dir
                full_path = self.model_dir / path
                return full_path if full_path.exists() else None

    def list_runs(self) -> List[Dict]:
        """
        List all training runs for this model.

        Returns:
            List of dicts with run info, sorted by date (newest first)
        """
        runs = []
        for run_dir in self.model_dir.glob("run_*"):
            if run_dir.is_dir():
                metrics_path = run_dir / "training_metrics.json"
                if metrics_path.exists():
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    runs.append({
                        "run_dir": run_dir.name,
                        "best_epoch": metrics.get("best_epoch"),
                        "best_val_loss": metrics.get("best_val_loss"),
                        "total_epochs": metrics.get("total_epochs"),
                        "early_stopped": metrics.get("early_stopped", False)
                    })
                else:
                    runs.append({
                        "run_dir": run_dir.name,
                        "best_epoch": None,
                        "best_val_loss": None,
                        "total_epochs": None,
                        "early_stopped": None
                    })

        # Sort by run name (which includes timestamp)
        runs.sort(key=lambda x: x["run_dir"], reverse=True)
        return runs


def create_generation_metadata(model_path: str, source_data: str,
                               model_params: Dict, output_path: Path) -> None:
    """
    Create metadata JSON file for generated synthetic data.

    Args:
        model_path: Path to the model used for generation
        source_data: Path to the source data used
        model_params: Model parameters dict
        output_path: Path where metadata JSON will be saved
    """
    metadata = {
        "model_path": str(model_path),
        "source_data": str(source_data),
        "model_params": model_params,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    # Test model tracker
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ModelTracker(Path(tmpdir) / "cvae", "cvae")

        # Create a run
        run_dir = tracker.create_run_dir()
        print(f"Created run dir: {run_dir}")

        # Simulate training metrics
        epoch_losses = [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6, "checkpoint": "checkpoints/epoch_1.pt"},
            {"epoch": 2, "train_loss": 0.4, "val_loss": 0.5, "checkpoint": "checkpoints/epoch_2.pt"},
            {"epoch": 3, "train_loss": 0.3, "val_loss": 0.35, "checkpoint": "checkpoints/epoch_3.pt"},
            {"epoch": 4, "train_loss": 0.25, "val_loss": 0.38, "checkpoint": "checkpoints/epoch_4.pt"},
        ]

        metrics = {
            "best_epoch": 3,
            "best_val_loss": 0.35,
            "total_epochs": 4,
            "early_stopped": False,
            "best_checkpoint": "checkpoints/epoch_3.pt",
            "epoch_losses": epoch_losses
        }

        tracker.save_training_metrics(run_dir, metrics)
        tracker.save_loss_curves(run_dir, epoch_losses)

        print(f"Saved metrics to: {run_dir / 'training_metrics.json'}")
        print(f"Saved loss curves to: {run_dir / 'loss_curves.png'}")
