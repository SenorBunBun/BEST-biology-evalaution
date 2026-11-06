#!/usr/bin/env python3
"""
Plot MMD history for all three models (CVAE, Diffusion, GAN) from a training run.

Usage:
    python analysis/plot_mmd_history.py --output-dir outputs/OSDR-405-2
    python analysis/plot_mmd_history.py --output-dir outputs/OSDR-405-2 --show-plots
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# Default colors from pipeline config
DEFAULT_COLORS = {
    "cvae": "#8E44AD",
    "diffusion": "#E57373",
    "gan": "#E69F00"
}

DEFAULT_DISPLAY_NAMES = {
    "cvae": "CVAE",
    "diffusion": "Diffusion",
    "gan": "GAN"
}


def find_latest_run(model_dir: Path) -> Path:
    """Find the latest run directory for a model."""
    latest_json = model_dir / "latest_model.json"
    if latest_json.exists():
        with open(latest_json) as f:
            data = json.load(f)
            run_dir = data.get("run_dir", "")
            if run_dir:
                # Handle relative paths (stored as just "run_YYYYMMDD_HHMMSS")
                run_path = Path(run_dir)
                if not run_path.is_absolute():
                    run_path = model_dir / run_path
                return run_path

    # Fallback: find most recent run_* directory
    runs = sorted(model_dir.glob("run_*"), reverse=True)
    if runs:
        return runs[0]
    return None


def load_mmd_history(run_dir: Path) -> dict:
    """Load MMD history from a run directory.

    Priority: posthoc > concurrent (training-time with early stopping) > legacy mmd_history.
    Returns dict with 'source' key indicating which file was loaded.
    """
    for filename in ["posthoc_mmd_history.json", "concurrent_mmd_history.json", "mmd_history.json"]:
        mmd_file = run_dir / filename
        if mmd_file.exists():
            with open(mmd_file) as f:
                data = json.load(f)
            data['_source'] = filename
            return data
    return None


def load_gan_training_metrics(run_dir: Path) -> dict:
    """Load GAN training metrics to get step-to-epoch conversion."""
    metrics_file = run_dir / "training_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None


def convert_gan_steps_to_epochs(mmd_data: dict, training_metrics: dict) -> list:
    """
    Convert GAN steps to epochs using the training metrics.

    Uses the formula from driver.py:
        steps_per_epoch = (n_samples * train_split / batch_size) / critic_iters

    Or directly from training_metrics: steps_per_epoch = total_steps / total_epochs
    """
    history = mmd_data.get("history", [])
    converted = []

    # Calculate steps_per_epoch from training metrics
    total_steps = training_metrics.get("total_steps", 1)
    total_epochs = training_metrics.get("total_epochs", 1)

    if total_epochs > 0:
        steps_per_epoch = total_steps / total_epochs
    else:
        # Fallback: try to infer from epoch_losses
        epoch_losses = training_metrics.get("epoch_losses", [])
        if len(epoch_losses) >= 2:
            # Use first two entries to calculate
            step1, epoch1 = epoch_losses[0]["step"], epoch_losses[0]["epoch"]
            step2, epoch2 = epoch_losses[1]["step"], epoch_losses[1]["epoch"]
            steps_per_epoch = (step2 - step1) / (epoch2 - epoch1)
        else:
            steps_per_epoch = 21  # Default fallback

    for entry in history:
        step = entry.get("step")
        mmd = entry.get("mmd")

        # Skip non-numeric steps (best, final, best_model, final_model, etc.)
        if not isinstance(step, (int, float)):
            continue

        epoch = step / steps_per_epoch
        converted.append({"epoch": epoch, "mmd": mmd})

    return converted, steps_per_epoch


def plot_mmd_history(output_dir: Path, colors: dict = None, display_names: dict = None,
                     dpi: int = 300, show_plots: bool = False):
    """
    Plot MMD history for all three models.

    Args:
        output_dir: Path to the output directory (e.g., outputs/OSDR-405-2)
        colors: Dict mapping model names to hex colors
        display_names: Dict mapping model names to display names
        dpi: Resolution for saved plot
        show_plots: Whether to display plot interactively
    """
    colors = colors or DEFAULT_COLORS
    display_names = display_names or DEFAULT_DISPLAY_NAMES

    trained_models_dir = output_dir / "trained_models"

    # Collect data for each model
    model_data = {}

    for model_name in ["cvae", "diffusion", "gan"]:
        model_dir = trained_models_dir / model_name
        if not model_dir.exists():
            print(f"[WARN] Model directory not found: {model_dir}")
            continue

        run_dir = find_latest_run(model_dir)
        if run_dir is None or not run_dir.exists():
            print(f"[WARN] No run directory found for {model_name}")
            continue

        mmd_data = load_mmd_history(run_dir)
        if mmd_data is None:
            print(f"[WARN] No MMD history found for {model_name} in {run_dir}")
            continue

        source = mmd_data.get("_source", "")
        history = mmd_data.get("history", [])

        if source == "concurrent_mmd_history.json":
            # Concurrent format: epochs are directly available
            epochs = [e["epoch"] for e in history]
            mmds = [e["mmd"] for e in history]
        elif model_name == "gan":
            # Legacy GAN format: convert steps to epochs
            training_metrics = load_gan_training_metrics(run_dir)
            if training_metrics:
                converted, steps_per_epoch = convert_gan_steps_to_epochs(mmd_data, training_metrics)
                print(f"[INFO] GAN: {steps_per_epoch:.2f} steps/epoch")
                epochs = [e["epoch"] for e in converted]
                mmds = [e["mmd"] for e in converted]
            else:
                print(f"[WARN] No training metrics for GAN, cannot convert steps to epochs")
                continue
        else:
            # Legacy CVAE/Diffusion format: epochs already present
            epochs = [e["epoch"] for e in history if e["epoch"] != "best"]
            mmds = [e["mmd"] for e in history if e["epoch"] != "best"]

        if epochs and mmds:
            model_data[model_name] = {"epochs": epochs, "mmds": mmds}
            print(f"[INFO] {display_names[model_name]}: {len(epochs)} data points, "
                  f"epoch range [{min(epochs):.1f}, {max(epochs):.1f}]"
                  f" (from {source})")

    if not model_data:
        print("[ERROR] No MMD data found for any model")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name in ["cvae", "diffusion", "gan"]:
        if model_name not in model_data:
            continue

        data = model_data[model_name]
        ax.plot(data["epochs"], data["mmds"],
                color=colors[model_name],
                label=display_names[model_name],
                linewidth=2,
                marker='o',
                markersize=4)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MMD", fontsize=12)
    ax.set_title("MMD History During Training", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set axis limits
    ax.set_xlim(left=0)

    plt.tight_layout()

    # Save plot
    metrics_dir = output_dir / "analysis" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    save_path = metrics_dir / "mmd_history_comparison.png"
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"[INFO] Plot saved to: {save_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot MMD history for all three models"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output directory (e.g., outputs/OSDR-405-2)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Optional: path to dataset config JSON to load colors and display names"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved plot (default: 300)"
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plot interactively"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"[ERROR] Output directory not found: {output_dir}")
        return 1

    # Load colors from config if provided
    colors = None
    display_names = None

    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            viz = config.get("visualization", {})
            colors = viz.get("colors", DEFAULT_COLORS)
            display_names = viz.get("display_names", DEFAULT_DISPLAY_NAMES)

    plot_mmd_history(
        output_dir=output_dir,
        colors=colors,
        display_names=display_names,
        dpi=args.dpi,
        show_plots=args.show_plots
    )

    return 0


if __name__ == "__main__":
    exit(main())
