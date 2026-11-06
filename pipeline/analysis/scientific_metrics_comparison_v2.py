#!/usr/bin/env python3
"""
Scientific Metrics Comparison V2: Simplified Multi-Model Comparison
Supports h5ad files for all models, assumes normalized data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import scanpy as sc
from datetime import datetime
from scipy.stats import wasserstein_distance
import warnings
import os

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - SIMPLE VERSION
# =============================================================================

# ==================== ORIGINAL CONFIG (COMMENTED OUT) ====================
# REAL_DATA_PATH = "data/raw_0.997_filtered_hvg_2000_augmented_combined_norm.h5ad"
# CVAE_DATA_PATH = "cvae_dataset_reconstruction_raw_enhanced_cvae_synthetic_data_20250805_105327.h5ad"
# DIFFUSION_DATA_PATH = "diffusion_model_v1_100_synthetic_data_matching_training_dist.h5ad"  # Can be .h5ad or .npy
# GAN_DATA_PATH = "data/gan_v1_60000_gpfix_raw_0.997_filtered_hvg_2000_augmented.h5ad"
# DATASET_PREFIX = ""  # Empty for default dataset
# NORMALIZE_REAL_DATA = False  # Data is already normalized
# =========================================================================

# ==================== CANCER DATA CONFIG ====================
REAL_DATA_PATH = "cancer_data/cancer_0.997_filtered_hvg_2000_norm.h5ad"
CVAE_DATA_PATH = "cancer_data/cvae_cancer_relu_raw_filtered_0.997_800_400_32.pt_20251106_013022.h5ad"  # Update with actual path
DIFFUSION_DATA_PATH = "cancer_data/diffusion_cancer_synthetic_20251106_090734.h5ad"  # Can be .h5ad or .npy
GAN_DATA_PATH = "cancer_data/cancer_gan_output.h5ad"  # Update with actual path
DATASET_PREFIX = "cancer"  # Prefix for cache files
NORMALIZE_REAL_DATA = False  # Set True if real data needs normalization, False if already normalized
# ============================================================

# Auto-generate cache file paths based on dataset prefix
if DATASET_PREFIX:
    CACHE_CVAE_WASSERSTEIN = f"cache_cvae_wasserstein_{DATASET_PREFIX}.npy"
    CACHE_CVAE_MAD = f"cache_cvae_mad_{DATASET_PREFIX}.npy"
    CACHE_DIFFUSION_WASSERSTEIN = f"cache_diffusion_wasserstein_{DATASET_PREFIX}.npy"
    CACHE_DIFFUSION_MAD = f"cache_diffusion_mad_{DATASET_PREFIX}.npy"
    CACHE_GAN_WASSERSTEIN = f"cache_gan_wasserstein_{DATASET_PREFIX}.npy"
    CACHE_GAN_MAD = f"cache_gan_mad_{DATASET_PREFIX}.npy"
else:
    CACHE_CVAE_WASSERSTEIN = "cache_cvae_wasserstein.npy"
    CACHE_CVAE_MAD = "cache_cvae_mad.npy"
    CACHE_DIFFUSION_WASSERSTEIN = "cache_diffusion_wasserstein.npy"
    CACHE_DIFFUSION_MAD = "cache_diffusion_mad.npy"
    CACHE_GAN_WASSERSTEIN = "cache_gan_wasserstein.npy"
    CACHE_GAN_MAD = "cache_gan_mad.npy"

# =============================================================================
# PLOTTING PARAMETERS FOR PUBLICATION QUALITY
# =============================================================================
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

# Color scheme (matching shared_deg_comparison.py)
COLORS = {
    'diffusion': 'lightcoral',
    'cvae': '#8E44AD',
    'gan': '#E69F00',
}


def load_data_from_path(file_path, apply_normalization=False):
    """
    Load data from either .h5ad or .npy file

    Parameters:
    -----------
    file_path : str
        Path to data file (.h5ad or .npy)
    apply_normalization : bool
        Whether to apply normalization (for diffusion compatibility)

    Returns:
    --------
    np.ndarray
        Data array
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"  Loading from {file_path}...")

    if file_path.endswith('.h5ad'):
        adata = sc.read_h5ad(file_path)
        data = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    elif file_path.endswith('.npy'):
        data = np.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Must be .h5ad or .npy")

    print(f"  Loaded data shape: {data.shape}")

    # Apply normalization if requested (for diffusion compatibility)
    if apply_normalization:
        print("  Applying normalization...")
        try:
            from diffusion.utils import preprocess_for_diffusion
            data = preprocess_for_diffusion(data)
            print("  Normalization applied")
        except Exception as e:
            print(f"  Warning: Could not apply normalization: {e}")

    return data


def load_real_data():
    """Load real data"""
    print(f"Loading real data...")
    return load_data_from_path(REAL_DATA_PATH, apply_normalization=NORMALIZE_REAL_DATA)


def load_cvae_data():
    """Load CVAE synthetic data and real data"""
    print("Loading CVAE data...")
    synthetic_data = load_data_from_path(CVAE_DATA_PATH, apply_normalization=False)
    real_data = load_real_data()
    return synthetic_data, real_data


def load_diffusion_data():
    """Load Diffusion synthetic data and real data"""
    print("Loading Diffusion data...")
    synthetic_data = load_data_from_path(DIFFUSION_DATA_PATH, apply_normalization=False)
    real_data = load_real_data()
    return synthetic_data, real_data


def load_gan_data():
    """Load GAN synthetic data and real data"""
    print("Loading GAN data...")
    synthetic_data = load_data_from_path(GAN_DATA_PATH, apply_normalization=False)
    real_data = load_real_data()
    return synthetic_data, real_data


def calculate_feature_wise_metrics(synthetic_data, real_data):
    """Calculate feature-wise Wasserstein distances and absolute differences"""
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


def calculate_cosine_similarity(synthetic_data, real_data):
    """Calculate cosine similarity between synthetic and real data means"""
    from sklearn.metrics.pairwise import cosine_similarity

    real_mean = np.mean(real_data, axis=0).reshape(1, -1)
    synthetic_mean = np.mean(synthetic_data, axis=0).reshape(1, -1)

    similarity = cosine_similarity(real_mean, synthetic_mean)[0][0]
    return similarity


def calculate_mahalanobis_distance(synthetic_data, real_data):
    """Calculate Mahalanobis distance between synthetic and real data distributions"""
    from scipy.spatial.distance import mahalanobis
    from scipy import linalg

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


def load_or_compute_metrics(model_name, load_data_func, cache_wasserstein, cache_mad):
    """Load metrics from cache or compute them"""

    # Try loading from cache
    if os.path.exists(cache_wasserstein) and os.path.exists(cache_mad):
        print(f"  Loading cached metrics for {model_name}...")
        wasserstein_dist = np.load(cache_wasserstein)
        mad_dist = np.load(cache_mad)

        # Still load data for cosine similarity and Mahalanobis distance
        try:
            synthetic_data, real_data = load_data_func()
        except Exception as e:
            print(f"  Warning: Could not load data: {e}")
            synthetic_data, real_data = None, None
    else:
        print(f"  Computing metrics for {model_name}...")
        synthetic_data, real_data = load_data_func()

        # Calculate metrics
        wasserstein_dist, mad_dist = calculate_feature_wise_metrics(synthetic_data, real_data)

        # Save to cache
        np.save(cache_wasserstein, wasserstein_dist)
        np.save(cache_mad, mad_dist)
        print(f"  Saved metrics to cache")

    # Calculate additional metrics
    print(f"  Calculating cosine similarity and Mahalanobis distance...")
    if synthetic_data is not None and real_data is not None:
        cosine_sim = calculate_cosine_similarity(synthetic_data, real_data)
        mahal_dist = calculate_mahalanobis_distance(synthetic_data, real_data)
    else:
        cosine_sim = None
        mahal_dist = None

    metrics = {
        'model': model_name,
        'cosine_similarity': cosine_sim if cosine_sim is not None else 0,
        'mahalanobis_distance': mahal_dist if mahal_dist is not None else 0,
        'wasserstein_distances': wasserstein_dist,
        'absolute_differences': mad_dist,
        'mean_wasserstein_distance': np.mean(wasserstein_dist),
        'mean_absolute_difference': np.mean(mad_dist),
    }

    print(f"  {model_name} metrics:")
    print(f"    Cosine similarity: {metrics['cosine_similarity']:.6f}")
    if mahal_dist is not None:
        print(f"    Mahalanobis distance: {metrics['mahalanobis_distance']:.6f}")
    print(f"    Mean Wasserstein distance: {metrics['mean_wasserstein_distance']:.6f}")
    print(f"    Mean absolute difference: {metrics['mean_absolute_difference']:.6f}")

    return metrics


def create_comparison_plot(model_metrics, available_models, timestamp, show_plots=False):
    """Create comprehensive comparison plot"""
    print("\nCreating comparison visualization...")

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Proper capitalization: CVAE, GAN, but Diffusion
    def format_model_name(model):
        if model.lower() == 'cvae':
            return 'CVAE'
        elif model.lower() == 'gan':
            return 'GAN'
        else:
            return model.capitalize()

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

    # Add mean value labels below the box plot area
    for i, (pos, data) in enumerate(zip(positions, wasserstein_data)):
        mean_val = np.mean(data)
        # Position labels below the box plot area to avoid overlap
        ax3.text(pos, ax3.get_ylim()[0] - (ax3.get_ylim()[1] - ax3.get_ylim()[0]) * 0.1,
                f'μ={mean_val:.4f}', ha='center', va='top', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

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

    # Add mean value labels below the box plot area
    for i, (pos, data) in enumerate(zip(positions, mad_data)):
        mean_val = np.mean(data)
        # Position labels below the box plot area to avoid overlap
        ax4.text(pos, ax4.get_ylim()[0] - (ax4.get_ylim()[1] - ax4.get_ylim()[0]) * 0.1,
                f'μ={mean_val:.4f}', ha='center', va='top', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Main title
    title_models = ' vs '.join(model_names)
    title_suffix = f" ({DATASET_PREFIX.upper()})" if DATASET_PREFIX else ""
    fig.suptitle(f'Synthetic Data Quality Comparison: {title_models}{title_suffix}',
                 fontsize=18, fontweight='bold', y=0.95)

    # Save
    filename_suffix = f"_{DATASET_PREFIX}" if DATASET_PREFIX else ""
    filename = f"scientific_metrics_comparison{filename_suffix}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved as: {filename}")

    if show_plots:
        plt.show()
    plt.close()
    return filename


def create_summary_table(model_metrics, available_models, timestamp):
    """Create summary table"""
    print("Creating summary table...")

    # Proper capitalization helper
    def format_model_name(model):
        if model.lower() == 'cvae':
            return 'CVAE'
        elif model.lower() == 'gan':
            return 'GAN'
        else:
            return model.capitalize()

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

    filename_suffix = f"_{DATASET_PREFIX}" if DATASET_PREFIX else ""
    csv_filename = f"metrics_comparison_table{filename_suffix}_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Table saved as: {csv_filename}")

    print("\nSUMMARY TABLE:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    return csv_filename


def main():
    """Main execution"""
    print("=" * 80)
    print("SCIENTIFIC METRICS COMPARISON V2")
    print("=" * 80)

    # Proper capitalization helper
    def format_model_name(model):
        if model.lower() == 'cvae':
            return 'CVAE'
        elif model.lower() == 'gan':
            return 'GAN'
        else:
            return model.capitalize()

    if DATASET_PREFIX:
        print(f"Dataset: {DATASET_PREFIX.upper()}")
    print(f"Normalize real data: {NORMALIZE_REAL_DATA}")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        available_models = []
        model_metrics = {}

        # Process each model
        model_configs = [
            ('cvae', load_cvae_data, CACHE_CVAE_WASSERSTEIN, CACHE_CVAE_MAD),
            ('diffusion', load_diffusion_data, CACHE_DIFFUSION_WASSERSTEIN, CACHE_DIFFUSION_MAD),
            ('gan', load_gan_data, CACHE_GAN_WASSERSTEIN, CACHE_GAN_MAD),
        ]

        for model_name, load_func, cache_w, cache_m in model_configs:
            print(f"\n{'='*60}")
            print(f"Processing {format_model_name(model_name)} model")
            print('='*60)

            try:
                metrics = load_or_compute_metrics(model_name, load_func, cache_w, cache_m)
                available_models.append(model_name)
                model_metrics[model_name] = metrics
            except FileNotFoundError as e:
                print(f"Skipping {format_model_name(model_name)}: {e}")
            except Exception as e:
                print(f"Error processing {format_model_name(model_name)}: {e}")

        if len(available_models) == 0:
            print("\nError: No models could be processed. Check file paths.")
            return False

        print(f"\n{'='*60}")
        print(f"Available models: {', '.join([format_model_name(m) for m in available_models])}")
        print('='*60)

        # Create visualizations
        plot_filename = create_comparison_plot(model_metrics, available_models, timestamp)
        table_filename = create_summary_table(model_metrics, available_models, timestamp)

        print("\n" + "=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"Generated files:")
        print(f"  1. Plot: {plot_filename}")
        print(f"  2. Table: {table_filename}")

        return True

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
