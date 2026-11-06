#!/usr/bin/env python3
"""
Full UMAP Comparison: Run UMAP for all three models consecutively
Step-based execution like simple_umap_comparison.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import umap
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import gc
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - DATASET PATHS
# =============================================================================

# Real data path
REAL_DATA_PATH = 'cancer_0.997_filtered_hvg_2000_norm.h5ad'

# Synthetic data paths for each model
SYNTHETIC_DATA_PATHS = {
    'cvae': 'cvae_cancer_relu_raw_filtered_0.997_800_400_32.pt_20251106_013022.h5ad',
    'diffusion': 'diffusion_cancer_synthetic_20251106_090734.h5ad',
    'gan': 'cancer_gan_output.h5ad'
}

# Dataset prefix for cache files (e.g., "_cancer" for cancer dataset)
DATASET_PREFIX = "_cancer"

# Model colors (matching shared_deg_comparison.py)
MODEL_COLORS = {
    'cvae': '#8E44AD',          # Purple
    'diffusion': 'lightcoral',  # Light coral
    'gan': '#E69F00',           # Orange
}

# =============================================================================
# PLOTTING PARAMETERS FOR PUBLICATION QUALITY
# =============================================================================
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


def get_cache_paths(model_name):
    """Generate cache file paths for a model"""
    scaled_path = f'umap_scaled_data_{model_name}{DATASET_PREFIX}.npy'
    labels_path = f'umap_data_labels_{model_name}{DATASET_PREFIX}.npy'
    umap_path = f'umap_embedding_{model_name}{DATASET_PREFIX}.npy'
    return scaled_path, labels_path, umap_path


def load_and_prepare_data(real_path, synthetic_path):
    """Load and prepare data with memory optimizations"""
    print(f"  Loading real data from {real_path}...")
    adata_real = sc.read_h5ad(real_path)
    real_data = adata_real.X
    if hasattr(real_data, 'toarray'):
        real_data = real_data.toarray()

    print(f"  Loading synthetic data from {synthetic_path}...")
    adata_synthetic = sc.read_h5ad(synthetic_path)
    synthetic_data = adata_synthetic.X
    if hasattr(synthetic_data, 'toarray'):
        synthetic_data = synthetic_data.toarray()

    # Clear AnnData objects to free memory
    del adata_real, adata_synthetic
    gc.collect()

    print(f"  Real data shape: {real_data.shape}")
    print(f"  Synthetic data shape: {synthetic_data.shape}")

    return real_data, synthetic_data


def standardize_and_save_data(real_data, synthetic_data, scaled_path, labels_path):
    """Standardize data and save to disk"""
    print("  Combining and standardizing data...")

    # Combine data
    combined_data = np.vstack([real_data, synthetic_data])
    data_labels = np.array(['Real'] * len(real_data) + ['Synthetic'] * len(synthetic_data))

    print(f"  Combined data shape: {combined_data.shape}")

    # Standardize data
    print("  Standardizing data...")
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data)

    # Save standardized data and labels
    print(f"  Saving to {scaled_path}...")
    np.save(scaled_path, combined_data_scaled)
    np.save(labels_path, data_labels)

    # Clear memory
    del combined_data, scaler
    gc.collect()

    print("  ✓ Standardization complete!")
    return True


def load_standardized_data(scaled_path, labels_path):
    """Load previously standardized data from disk"""
    if not os.path.exists(scaled_path) or not os.path.exists(labels_path):
        return None, None

    print(f"  Loading previously standardized data from {scaled_path}...")
    combined_data_scaled = np.load(scaled_path)
    data_labels = np.load(labels_path)

    print(f"  Loaded scaled data shape: {combined_data_scaled.shape}")
    return combined_data_scaled, data_labels.tolist()


def create_and_save_umap_embedding(combined_data_scaled, umap_path):
    """Create UMAP embedding and save it"""
    print("  Computing UMAP embedding...")

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42,
        metric='euclidean',
        low_memory=True,
        verbose=True
    )

    try:
        embedding = reducer.fit_transform(combined_data_scaled)

        # Save embedding
        print(f"  Saving UMAP embedding to {umap_path}...")
        np.save(umap_path, embedding)

        print("  ✓ UMAP embedding complete!")
        return True

    except Exception as e:
        print(f"  Error during UMAP computation: {e}")
        return False


def load_umap_embedding(umap_path):
    """Load previously computed UMAP embedding from disk"""
    if not os.path.exists(umap_path):
        return None

    print(f"  Loading previously computed UMAP embedding from {umap_path}...")
    embedding = np.load(umap_path)

    print(f"  Loaded embedding shape: {embedding.shape}")
    return embedding


def create_scatter_plot(embedding, data_labels, model_name, color, timestamp, plot_real_only=False):
    """Create scatter plot overlay with synthetic data on top

    Parameters:
    -----------
    embedding : np.ndarray
        UMAP embedding coordinates
    data_labels : list
        Labels indicating 'Real' or 'Synthetic' for each point
    model_name : str
        Name of the model for labeling
    color : str
        Color for synthetic data points
    timestamp : str
        Timestamp for filename
    plot_real_only : bool
        If True, plot only real data (no synthetic overlay). Default: False
    """
    print("  Creating scatter plot...")

    # Create DataFrame
    df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'Data_Type': data_labels
    })

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot real data FIRST (background)
    real_mask = df['Data_Type'] == 'Real'
    ax.scatter(df.loc[real_mask, 'UMAP1'],
               df.loc[real_mask, 'UMAP2'],
               c='steelblue', alpha=0.7, s=30,
               label='Real Data',
               edgecolors='black', linewidth=0.3)

    # Plot synthetic data SECOND (foreground/on top) - ONLY if not plot_real_only
    if not plot_real_only:
        synthetic_mask = df['Data_Type'] == 'Synthetic'
        ax.scatter(df.loc[synthetic_mask, 'UMAP1'],
                   df.loc[synthetic_mask, 'UMAP2'],
                   c=color, alpha=0.7, s=30,
                   label='Synthetic Data',
                   edgecolors='black', linewidth=0.3)

    # Formatting
    ax.set_xlabel('UMAP1', fontsize=16, fontweight='bold')
    ax.set_ylabel('UMAP2', fontsize=16, fontweight='bold')

    if plot_real_only:
        ax.set_title(f'UMAP: Real Single-Cell Data Only - {model_name.upper()}',
                    fontsize=18, fontweight='bold', pad=20)
    else:
        ax.set_title(f'UMAP: Real vs Synthetic Single-Cell Data - {model_name.upper()}',
                    fontsize=18, fontweight='bold', pad=20)

    # Legend
    ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True,
              loc='upper right', bbox_to_anchor=(0.98, 0.98))

    # Grid and styling
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save plot
    if plot_real_only:
        filename = f"umap_scatter_{model_name}{DATASET_PREFIX}_real_only_{timestamp}.png"
    else:
        filename = f"umap_scatter_{model_name}{DATASET_PREFIX}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Plot saved as: {filename}")

    plt.close(fig)
    del df
    gc.collect()

    return filename


# =============================================================================
# STEP FUNCTIONS (like simple_umap_comparison.py)
# =============================================================================

def run_step1_standardization():
    """Run only the standardization step for all models"""
    print("=" * 80)
    print("STEP 1: DATA STANDARDIZATION FOR ALL MODELS")
    print("=" * 80)

    results = {}

    for model_name, synthetic_path in SYNTHETIC_DATA_PATHS.items():
        print(f"\n{'='*60}")
        print(f"Standardizing {model_name.upper()} data")
        print('='*60)

        scaled_path, labels_path, _ = get_cache_paths(model_name)

        try:
            # Load data
            real_data, synthetic_data = load_and_prepare_data(REAL_DATA_PATH, synthetic_path)

            # Standardize and save
            success = standardize_and_save_data(real_data, synthetic_data, scaled_path, labels_path)

            # Clear data from memory
            del real_data, synthetic_data
            gc.collect()

            results[model_name] = success

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[model_name] = False

    # Print summary
    print("\n" + "=" * 80)
    print("STEP 1 SUMMARY")
    print("=" * 80)
    for model, success in results.items():
        status = "✓" if success else "❌"
        print(f"{status} {model.upper()}")

    if all(results.values()):
        print("\n✓ All models standardized! Run step 2 to create UMAP embeddings.")
        return True
    else:
        print("\n⚠️  Some models failed. Check errors above.")
        return False


def run_step2_create_embeddings():
    """Run only the UMAP embedding creation step for all models"""
    print("=" * 80)
    print("STEP 2: CREATE UMAP EMBEDDINGS FOR ALL MODELS")
    print("=" * 80)

    results = {}

    for model_name in SYNTHETIC_DATA_PATHS.keys():
        print(f"\n{'='*60}")
        print(f"Creating UMAP for {model_name.upper()}")
        print('='*60)

        scaled_path, labels_path, umap_path = get_cache_paths(model_name)

        # Check if UMAP embedding already exists for this model
        if os.path.exists(umap_path):
            print(f"  ✓ UMAP embedding already exists for {model_name.upper()}: {umap_path}")
            print(f"  Skipping UMAP creation for this model.")
            results[model_name] = True
            continue

        try:
            # Load standardized data
            combined_data_scaled, data_labels = load_standardized_data(scaled_path, labels_path)

            if combined_data_scaled is None:
                print(f"  ❌ No standardized data found. Please run step 1 first.")
                results[model_name] = False
                continue

            # Create and save UMAP embedding
            success = create_and_save_umap_embedding(combined_data_scaled, umap_path)

            # Clear scaled data from memory
            del combined_data_scaled
            gc.collect()

            results[model_name] = success

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[model_name] = False

    # Print summary
    print("\n" + "=" * 80)
    print("STEP 2 SUMMARY")
    print("=" * 80)
    for model, success in results.items():
        status = "✓" if success else "❌"
        print(f"{status} {model.upper()}")

    if all(results.values()):
        print("\n✓ All embeddings created! Run step 3 to create visualizations.")
        return True
    else:
        print("\n⚠️  Some models failed. Check errors above.")
        return False


def run_step3_create_plots(plot_real_only=False):
    """Run only the plotting step for all models

    Parameters:
    -----------
    plot_real_only : bool
        If True, plot only real data without synthetic overlay. Default: False
    """
    print("=" * 80)
    if plot_real_only:
        print("STEP 3: CREATE VISUALIZATIONS (REAL DATA ONLY)")
    else:
        print("STEP 3: CREATE VISUALIZATIONS FOR ALL MODELS")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    for model_name in SYNTHETIC_DATA_PATHS.keys():
        print(f"\n{'='*60}")
        if plot_real_only:
            print(f"Creating real-only plot for {model_name.upper()}")
        else:
            print(f"Creating plot for {model_name.upper()}")
        print('='*60)

        scaled_path, labels_path, umap_path = get_cache_paths(model_name)
        color = MODEL_COLORS[model_name]

        try:
            # Load UMAP embedding
            embedding = load_umap_embedding(umap_path)

            if embedding is None:
                print(f"  ❌ No UMAP embedding found. Please run step 2 first.")
                results[model_name] = None
                continue

            # Load data labels
            _, data_labels = load_standardized_data(scaled_path, labels_path)

            if data_labels is None:
                print(f"  ❌ No data labels found. Please run step 1 first.")
                results[model_name] = None
                continue

            # Create and save plot
            filename = create_scatter_plot(embedding, data_labels, model_name, color, timestamp, plot_real_only=plot_real_only)

            results[model_name] = filename

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = None

    # Print summary
    print("\n" + "=" * 80)
    print("STEP 3 SUMMARY")
    print("=" * 80)
    for model, filename in results.items():
        if filename:
            print(f"✓ {model.upper()}: {filename}")
        else:
            print(f"❌ {model.upper()}: Failed")

    successful = sum(1 for f in results.values() if f is not None)
    total = len(results)

    if successful == total:
        print(f"\n✓ All {total} plots created successfully!")
        return True
    elif successful > 0:
        print(f"\n⚠️  Partial success: {successful}/{total} plots created")
        return False
    else:
        print("\n❌ All plots failed")
        return False


def main():
    """Main execution - runs all steps sequentially"""
    print("=" * 80)
    print("FULL UMAP COMPARISON - ALL MODELS")
    print("=" * 80)

    # Check what's cached and determine starting point
    all_scaled_exist = all(
        os.path.exists(get_cache_paths(model)[0]) and os.path.exists(get_cache_paths(model)[1])
        for model in SYNTHETIC_DATA_PATHS.keys()
    )
    all_umaps_exist = all(
        os.path.exists(get_cache_paths(model)[2])
        for model in SYNTHETIC_DATA_PATHS.keys()
    )

    if all_umaps_exist and all_scaled_exist:
        print("Found existing standardized data and UMAP embeddings for all models.")
        print("Skipping to step 3...")
        success = run_step3_create_plots()
    elif all_scaled_exist:
        print("Found existing standardized data for all models.")
        print("Starting from step 2...")
        success = run_step2_create_embeddings()
        if success:
            success = run_step3_create_plots()
    else:
        print("Starting from step 1...")
        # Run step 1
        success = run_step1_standardization()

        if success:
            # Run step 2
            success = run_step2_create_embeddings()

            if success:
                # Run step 3
                success = run_step3_create_plots()

    if not success:
        print("\n" + "=" * 80)
        print("MANUAL EXECUTION GUIDE")
        print("=" * 80)
        print("If you encounter errors, try running steps manually:")
        print("")
        print("1. For standardization only:")
        print("   python full_umap.py step1")
        print("")
        print("2. For UMAP embedding creation only (after step 1):")
        print("   python full_umap.py step2")
        print("")
        print("3. For visualization only (after steps 1 and 2):")
        print("   python full_umap.py step3")
        print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "step1":
            run_step1_standardization()
        elif sys.argv[1] == "step2":
            run_step2_create_embeddings()
        elif sys.argv[1] == "step3":
            # Check for optional 'real_only' argument
            plot_real_only = len(sys.argv) > 2 and sys.argv[2] == "real_only"
            run_step3_create_plots(plot_real_only=plot_real_only)
        else:
            print("Usage: python full_umap.py [step1|step2|step3 [real_only]]")
            print("  step1           : Standardize data for all models")
            print("  step2           : Create UMAP embeddings for all models")
            print("  step3           : Create visualizations for all models (real + synthetic)")
            print("  step3 real_only : Create visualizations with ONLY real data (no synthetic overlay)")
    else:
        main()
