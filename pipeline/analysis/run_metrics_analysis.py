#!/usr/bin/env python3
"""
Scientific Metrics Analysis Runner for Pipeline Integration

Computes scientific metrics (Wasserstein distance, cosine similarity, etc.)
comparing real vs synthetic data for all models.

Usage:
    python run_metrics_analysis.py --real path/to/real.h5ad \
        --cvae path/to/cvae.h5ad --diffusion path/to/diffusion.h5ad --gan path/to/gan.h5ad \
        --output-dir path/to/output --dataset-name "cancer"
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import scanpy as sc
import json
import warnings

warnings.filterwarnings('ignore')

from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import mahalanobis
from scipy import linalg

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
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
COLORS = {
    'cvae': '#8E44AD',          # Purple
    'diffusion': 'lightcoral',  # Light coral
    'gan': '#E69F00',           # Orange
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run scientific metrics analysis comparing real vs synthetic data'
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
                        help='Output directory for results')
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


def calculate_feature_wise_metrics(synthetic_data, real_data):
    """Calculate feature-wise Wasserstein distances and absolute differences."""
    print("  Calculating feature-wise metrics...")

    n_features = min(synthetic_data.shape[1], real_data.shape[1])

    wasserstein_distances = []
    absolute_differences = []

    for i in range(n_features):
        if i % 500 == 0:
            print(f"    Processing feature {i}/{n_features}...")

        # Wasserstein distance
        wd = wasserstein_distance(real_data[:, i], synthetic_data[:, i])
        wasserstein_distances.append(wd)

        # Mean absolute difference
        mad = np.abs(np.mean(real_data[:, i]) - np.mean(synthetic_data[:, i]))
        absolute_differences.append(mad)

    print(f"  Calculated metrics for {n_features} features")

    return np.array(wasserstein_distances), np.array(absolute_differences)


def calculate_cosine_similarity_metric(synthetic_data, real_data):
    """Calculate cosine similarity between synthetic and real data means."""
    real_mean = np.mean(real_data, axis=0).reshape(1, -1)
    synthetic_mean = np.mean(synthetic_data, axis=0).reshape(1, -1)

    similarity = cosine_similarity(real_mean, synthetic_mean)[0][0]
    return similarity


def calculate_mahalanobis_distance(synthetic_data, real_data):
    """Calculate Mahalanobis distance between synthetic and real data distributions."""
    try:
        real_mean = np.mean(real_data, axis=0)
        synthetic_mean = np.mean(synthetic_data, axis=0)

        cov_matrix = np.cov(real_data.T)
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
        inv_cov_matrix = linalg.inv(cov_matrix)

        distance = mahalanobis(real_mean, synthetic_mean, inv_cov_matrix)
        return distance
    except Exception as e:
        print(f"  Warning: Could not calculate Mahalanobis distance: {e}")
        return None


def compute_model_metrics(model_name, synthetic_data, real_data):
    """Compute all metrics for a model."""
    print(f"\n  Computing metrics for {model_name}...")

    # Calculate feature-wise metrics
    wasserstein_dist, mad_dist = calculate_feature_wise_metrics(synthetic_data, real_data)

    # Calculate global metrics
    cosine_sim = calculate_cosine_similarity_metric(synthetic_data, real_data)
    mahal_dist = calculate_mahalanobis_distance(synthetic_data, real_data)

    metrics = {
        'model': model_name,
        'cosine_similarity': cosine_sim if cosine_sim is not None else 0,
        'mahalanobis_distance': mahal_dist if mahal_dist is not None else 0,
        'wasserstein_distances': wasserstein_dist,
        'absolute_differences': mad_dist,
        'mean_wasserstein_distance': np.mean(wasserstein_dist),
        'mean_absolute_difference': np.mean(mad_dist),
    }

    print(f"    Cosine similarity: {metrics['cosine_similarity']:.6f}")
    if mahal_dist is not None:
        print(f"    Mahalanobis distance: {metrics['mahalanobis_distance']:.6f}")
    print(f"    Mean Wasserstein distance: {metrics['mean_wasserstein_distance']:.6f}")
    print(f"    Mean absolute difference: {metrics['mean_absolute_difference']:.6f}")

    return metrics


def format_model_name(model):
    """Proper capitalization for model names."""
    if model.lower() == 'cvae':
        return 'CVAE'
    elif model.lower() == 'gan':
        return 'GAN'
    else:
        return model.capitalize()


def create_comparison_plot(model_metrics, available_models, output_dir, dataset_name,
                           timestamp, show_plots=False, core_output_dir=None):
    """Create comprehensive comparison plot."""
    print("\nCreating comparison visualization...")

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    model_names = [format_model_name(m) for m in available_models]
    model_colors = [COLORS[m] for m in available_models]

    # Subplot 1: Cosine Similarity
    ax1 = fig.add_subplot(gs[0, 0])
    cosine_values = [model_metrics[m]['cosine_similarity'] for m in available_models]

    bars1 = ax1.bar(model_names, cosine_values, color=model_colors, alpha=0.7,
                    edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Cosine Similarity', fontweight='bold')
    ax1.set_title('A) Cosine Similarity\n(Higher = Better)', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')
    if max(cosine_values) > 0:
        ax1.set_ylim(0, max(cosine_values) * 1.1)

    for bar, value in zip(bars1, cosine_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

    # Subplot 2: Mahalanobis Distance
    ax2 = fig.add_subplot(gs[0, 1])
    mahal_values = [model_metrics[m]['mahalanobis_distance'] for m in available_models]

    bars2 = ax2.bar(model_names, mahal_values, color=model_colors, alpha=0.7,
                    edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Mahalanobis Distance', fontweight='bold')
    ax2.set_title('B) Mahalanobis Distance\n(Lower = Better)', fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')
    if max(mahal_values) > 0:
        ax2.set_ylim(0, max(mahal_values) * 1.1)

    for bar, value in zip(bars2, mahal_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    # Subplot 3: Wasserstein Distance Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    wasserstein_data = [model_metrics[m]['wasserstein_distances'] for m in available_models]

    positions = list(range(1, len(available_models) + 1))
    bp3 = ax3.boxplot(wasserstein_data, positions=positions, widths=0.5,
                      patch_artist=True, showfliers=True)

    for i, model in enumerate(available_models):
        bp3['boxes'][i].set_facecolor(COLORS[model])
        bp3['boxes'][i].set_alpha(0.7)

    ax3.set_xlim(0.5, len(available_models) + 0.5)
    ax3.set_xticks(positions)
    ax3.set_xticklabels(model_names)
    ax3.set_ylabel('Wasserstein Distance', fontweight='bold')
    ax3.set_title('C) Wasserstein Distance Distribution\n(Lower = Better)', fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add mean value labels below the boxplots
    y_min = ax3.get_ylim()[0]
    for i, data in enumerate(wasserstein_data):
        mean_val = np.mean(data)
        ax3.text(positions[i], y_min - (ax3.get_ylim()[1] - y_min) * 0.08,
                f'μ={mean_val:.4f}', ha='center', va='top', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    # Subplot 4: Absolute Difference Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    mad_data = [model_metrics[m]['absolute_differences'] for m in available_models]

    bp4 = ax4.boxplot(mad_data, positions=positions, widths=0.5,
                      patch_artist=True, showfliers=True)

    for i, model in enumerate(available_models):
        bp4['boxes'][i].set_facecolor(COLORS[model])
        bp4['boxes'][i].set_alpha(0.7)

    ax4.set_xlim(0.5, len(available_models) + 0.5)
    ax4.set_xticks(positions)
    ax4.set_xticklabels(model_names)
    ax4.set_ylabel('Absolute Difference', fontweight='bold')
    ax4.set_title('D) Absolute Difference Distribution\n(Lower = Better)', fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add mean value labels below the boxplots
    y_min = ax4.get_ylim()[0]
    for i, data in enumerate(mad_data):
        mean_val = np.mean(data)
        ax4.text(positions[i], y_min - (ax4.get_ylim()[1] - y_min) * 0.08,
                f'μ={mean_val:.4f}', ha='center', va='top', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    # Main title
    title_models = ' vs '.join(model_names)
    fig.suptitle(f'Synthetic Data Quality Comparison: {title_models}\n{dataset_name}',
                 fontsize=18, fontweight='bold', y=0.98)

    # Adjust subplot layout
    plt.subplots_adjust(top=0.852, bottom=0.11)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"scientific_metrics_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved as: {filename}")

    # Also save without timestamp
    latest_filename = output_dir / "scientific_metrics_comparison.png"
    plt.savefig(latest_filename, dpi=300, bbox_inches='tight', facecolor='white')

    # Save core plot to main analysis directory if specified
    if core_output_dir:
        core_dir = Path(core_output_dir)
        core_dir.mkdir(parents=True, exist_ok=True)
        core_filename = core_dir / "scientific_metrics_comparison.png"
        plt.savefig(core_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Core plot: {core_filename}")

    if show_plots:
        plt.show()
    plt.close()

    return str(filename)


def create_summary_table(model_metrics, available_models, output_dir, timestamp):
    """Create summary table."""
    print("Creating summary table...")

    data = {
        'Metric': [
            'Cosine Similarity',
            'Mahalanobis Distance',
            'Mean Wasserstein Distance',
            'Mean Absolute Difference'
        ]
    }

    for model in available_models:
        model_name = format_model_name(model)
        metrics = model_metrics[model]
        data[model_name] = [
            f"{metrics['cosine_similarity']:.6f}",
            f"{metrics['mahalanobis_distance']:.4f}",
            f"{metrics['mean_wasserstein_distance']:.6f}",
            f"{metrics['mean_absolute_difference']:.6f}"
        ]

    data['Best Model'] = []
    metric_keys = ['cosine_similarity', 'mahalanobis_distance', 'mean_wasserstein_distance', 'mean_absolute_difference']
    higher_better = [True, False, False, False]

    for key, is_higher_better in zip(metric_keys, higher_better):
        values = {model: model_metrics[model][key] for model in available_models}
        best_model = max(values, key=values.get) if is_higher_better else min(values, key=values.get)
        data['Best Model'].append(format_model_name(best_model))

    df = pd.DataFrame(data)

    output_dir = Path(output_dir)
    csv_filename = output_dir / f"metrics_comparison_table_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Table saved as: {csv_filename}")

    # Also save without timestamp
    latest_csv = output_dir / "metrics_comparison_table.csv"
    df.to_csv(latest_csv, index=False)

    print("\nSUMMARY TABLE:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    return str(csv_filename)


def main():
    args = parse_args()

    print("=" * 60)
    print("Scientific Metrics Analysis Pipeline")
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

    # Process each model
    models = [
        ('cvae', args.cvae),
        ('diffusion', args.diffusion),
        ('gan', args.gan)
    ]

    available_models = []
    model_metrics = {}

    for model_name, data_path in models:
        if data_path is None or not Path(data_path).exists():
            print(f"\n[SKIP] {model_name.upper()}: No data path provided or file not found")
            continue

        print(f"\n" + "-" * 40)
        print(f"Processing {model_name.upper()}...")

        try:
            synthetic_data = load_data(data_path)
            metrics = compute_model_metrics(model_name, synthetic_data, real_data)
            available_models.append(model_name)
            model_metrics[model_name] = metrics

            # Cleanup
            del synthetic_data

        except Exception as e:
            print(f"[ERROR] Failed to process {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if not available_models:
        print("\n[ERROR] No models could be processed")
        sys.exit(1)

    # Create visualizations
    print("\n" + "-" * 40)
    print("Creating visualizations...")

    plot_filename = create_comparison_plot(
        model_metrics, available_models, args.output_dir,
        args.dataset_name, timestamp, args.show_plots,
        core_output_dir=args.core_output_dir
    )

    table_filename = create_summary_table(
        model_metrics, available_models, args.output_dir, timestamp
    )

    # Summary
    print("\n" + "=" * 60)
    print("SCIENTIFIC METRICS ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  Plot: {plot_filename}")
    print(f"  Table: {table_filename}")

    sys.exit(0)


if __name__ == "__main__":
    main()
