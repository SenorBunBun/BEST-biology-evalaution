#!/usr/bin/env python3
"""
UMAP Analysis Runner for Pipeline Integration

Creates UMAP visualizations comparing real vs synthetic data for all models.

Usage:
    python run_umap_analysis.py --real path/to/real.h5ad \
        --cvae path/to/cvae.h5ad --diffusion path/to/diffusion.h5ad --gan path/to/gan.h5ad \
        --output-dir path/to/output --dataset-name "OSDR-402"
"""

import argparse
import sys
import os
import gc
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import scanpy as sc
import warnings

warnings.filterwarnings('ignore')

# Try to import umap
try:
    import umap
except ImportError:
    print("[ERROR] umap-learn not installed. Install with: pip install umap-learn")
    sys.exit(1)

from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Plotting parameters
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'Arial',
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.linewidth': 1.2,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True
})

# Model order for consistent plotting (CVAE, Diffusion, GAN)
MODEL_ORDER = ['cvae', 'diffusion', 'gan']

# Color scheme matching pipeline visualization
MODEL_COLORS = {
    'cvae': '#8E44AD',          # Purple
    'diffusion': 'lightcoral',  # Light coral
    'gan': '#E69F00',           # Orange
}

# Display names for models
MODEL_DISPLAY_NAMES = {
    'cvae': 'CVAE',
    'diffusion': 'Diffusion',
    'gan': 'GAN'
}


def format_model_name(model: str) -> str:
    """Get proper display name for model."""
    return MODEL_DISPLAY_NAMES.get(model.lower(), model.upper())


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run UMAP analysis comparing real vs synthetic data'
    )

    parser.add_argument('--real', required=True,
                        help='Path to real data h5ad file')
    parser.add_argument('--cvae', default=None,
                        help='Path to CVAE synthetic data h5ad file')
    parser.add_argument('--diffusion', default=None,
                        help='Path to Diffusion synthetic data h5ad file')
    parser.add_argument('--gan', default=None,
                        help='Path to GAN synthetic data h5ad file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for UMAP plots')
    parser.add_argument('--core-output-dir', default=None,
                        help='Directory for core plots (defaults to output-dir)')
    parser.add_argument('--dataset-name', default='Dataset',
                        help='Dataset name for plot titles')
    parser.add_argument('--show-plots', action='store_true',
                        help='Display plots with plt.show() (default: False)')

    return parser.parse_args()


def load_data(path):
    """Load h5ad data and convert to numpy array."""
    print(f"  Loading {path}...")
    adata = sc.read_h5ad(path)
    data = adata.X
    if hasattr(data, 'toarray'):
        data = data.toarray()
    print(f"    Shape: {data.shape}")
    return data


def create_umap_plot(real_data, synthetic_data, model_name, color, output_dir,
                     dataset_name, timestamp, show_plots=False, core_output_dir=None):
    """Create UMAP visualization for real vs synthetic data."""
    print(f"\n  Creating UMAP for {model_name.upper()}...")

    # Combine data
    combined_data = np.vstack([real_data, synthetic_data])
    data_labels = ['Real'] * len(real_data) + ['Synthetic'] * len(synthetic_data)

    print(f"    Combined shape: {combined_data.shape}")

    # Standardize
    print("    Standardizing data...")
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_data)

    # UMAP
    print("    Computing UMAP embedding...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42,
        metric='euclidean',
        low_memory=True,
        verbose=False
    )
    embedding = reducer.fit_transform(combined_scaled)

    # Create plot
    print("    Creating scatter plot...")
    fig, ax = plt.subplots(figsize=(12, 10))

    df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'Data_Type': data_labels
    })

    # Plot real data first (background)
    real_mask = df['Data_Type'] == 'Real'
    ax.scatter(df.loc[real_mask, 'UMAP1'],
               df.loc[real_mask, 'UMAP2'],
               c='steelblue', alpha=0.7, s=30,
               label='Real Data',
               edgecolors='black', linewidth=0.3)

    # Plot synthetic data second (foreground)
    synthetic_mask = df['Data_Type'] == 'Synthetic'
    ax.scatter(df.loc[synthetic_mask, 'UMAP1'],
               df.loc[synthetic_mask, 'UMAP2'],
               c=color, alpha=0.7, s=30,
               label='Synthetic Data',
               edgecolors='black', linewidth=0.3)

    # Formatting
    ax.set_xlabel('UMAP1', fontsize=16, fontweight='bold')
    ax.set_ylabel('UMAP2', fontsize=16, fontweight='bold')
    ax.set_title(f'UMAP: Real vs {format_model_name(model_name)} Synthetic Data\n{dataset_name}',
                 fontsize=18, fontweight='bold', pad=20)

    ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True,
              loc='upper right', bbox_to_anchor=(0.98, 0.98))

    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save plot
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"umap_{model_name}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    Saved: {filename}")

    # Also save without timestamp for easy reference
    latest_filename = output_dir / f"umap_{model_name}.png"
    plt.savefig(latest_filename, dpi=300, bbox_inches='tight', facecolor='white')

    # Save core plot to main analysis directory if specified
    if core_output_dir:
        core_dir = Path(core_output_dir)
        core_dir.mkdir(parents=True, exist_ok=True)
        core_filename = core_dir / f"umap_{model_name}.png"
        plt.savefig(core_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"    Core plot: {core_filename}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # Cleanup
    del combined_data, combined_scaled, embedding, df
    gc.collect()

    return str(filename)


def main():
    args = parse_args()

    print("=" * 60)
    print("UMAP Analysis Pipeline")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Real data: {args.real}")
    print(f"  CVAE data: {args.cvae}")
    print(f"  Diffusion data: {args.diffusion}")
    print(f"  GAN data: {args.gan}")
    print(f"  Output: {args.output_dir}")
    print(f"  Dataset: {args.dataset_name}")

    # Load real data
    print("\n" + "-" * 40)
    print("Loading real data...")
    try:
        real_data = load_data(args.real)
    except Exception as e:
        print(f"[ERROR] Failed to load real data: {e}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    # Process each model
    models = [
        ('cvae', args.cvae),
        ('diffusion', args.diffusion),
        ('gan', args.gan)
    ]

    for model_name, data_path in models:
        if data_path is None or not Path(data_path).exists():
            print(f"\n[SKIP] {model_name.upper()}: No data path provided or file not found")
            continue

        print(f"\n" + "-" * 40)
        print(f"Processing {model_name.upper()}...")

        try:
            synthetic_data = load_data(data_path)
            color = MODEL_COLORS.get(model_name, '#808080')

            plot_path = create_umap_plot(
                real_data=real_data,
                synthetic_data=synthetic_data,
                model_name=model_name,
                color=color,
                output_dir=args.output_dir,
                dataset_name=args.dataset_name,
                timestamp=timestamp,
                show_plots=args.show_plots,
                core_output_dir=args.core_output_dir
            )
            results[model_name] = plot_path

            # Cleanup
            del synthetic_data
            gc.collect()

        except Exception as e:
            print(f"[ERROR] Failed to process {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("UMAP ANALYSIS COMPLETE")
    print("=" * 60)

    if results:
        print("\nGenerated plots:")
        for model, path in results.items():
            print(f"  {model.upper()}: {path}")
    else:
        print("\n[WARN] No UMAP plots were generated")

    sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
