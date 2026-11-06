"""
Differential Expression Analysis for Cancer Data - All Models

This script:
1. Runs DEG analysis on real data ONCE
2. Compares each synthetic model (CVAE, Diffusion, GAN) against the real DEGs
3. Saves all results to a unified "Differential_Expression" folder
4. Automatically generates the shared DEG comparison plot
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('..')
from diff_exp_gene_pipeline_v2 import DEGCrossComparison
import scanpy as sc
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = Path("Differential_Expression")
N_GENES = 50
METHOD = 't-test'
COLUMN_NAME = "tissue"
GROUP1 = "metastasis normal"  # Reference group
GROUP2 = "metastasis tumor"   # Target group
DATASET_NAME = "Cancer Dataset"

# File paths (update these as needed)
REAL_DATA_PATH = 'cancer_0.997_filtered_hvg_2000_norm.h5ad'
CVAE_DATA_PATH = 'cvae_cancer_relu_raw_filtered_0.997_800_400_32.pt_20251106_013022.h5ad'
DIFFUSION_DATA_PATH = 'diffusion_cancer_synthetic_20251106_090734.h5ad'
GAN_DATA_PATH = 'cancer_gan_output.h5ad'

# =============================================================================
# SETUP OUTPUT DIRECTORY
# =============================================================================
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================
print("="*80)
print("Loading datasets...")
print("="*80)

adata_real = sc.read_h5ad(REAL_DATA_PATH)
print(f"Real data shape: {adata_real.X.shape}")

adata_cvae = sc.read_h5ad(CVAE_DATA_PATH)
print(f"CVAE data shape: {adata_cvae.X.shape}")

adata_diffusion = sc.read_h5ad(DIFFUSION_DATA_PATH)
print(f"Diffusion data shape: {adata_diffusion.X.shape}")

adata_gan = sc.read_h5ad(GAN_DATA_PATH)
print(f"GAN data shape: {adata_gan.X.shape}")

# =============================================================================
# RUN DEG ANALYSIS ON REAL DATA (ONCE)
# =============================================================================
print("\n" + "="*80)
print("Running DEG analysis on Real Data (once)")
print("="*80)

# Create a temporary analyzer just to get real DEGs
from scipy import sparse
from diff_exp_gene_pipeline_v2 import DEGCrossComparison

# Prepare real data for DEG
adata_real_copy = adata_real.copy()
adata_real_copy.obs[COLUMN_NAME] = adata_real_copy.obs[COLUMN_NAME].astype(str)
adata_real_copy.X = sparse.csr_matrix(adata_real_copy.X)

# Run DEG on real data
sc.tl.rank_genes_groups(
    adata_real_copy,
    COLUMN_NAME,
    method=METHOD,
    reference=GROUP1,
    use_raw=False
)

# Extract real DEG results
real_deg_results = sc.get.rank_genes_groups_df(adata_real_copy, group=GROUP2)
real_deg_results['abs_logfoldchanges'] = np.abs(real_deg_results['logfoldchanges'])
real_deg_results = real_deg_results.sort_values(['pvals_adj', 'abs_logfoldchanges'], ascending=[True, False])
real_deg_results = real_deg_results.head(N_GENES)

print(f"✓ Real data DEG analysis complete")
print(f"  Top {N_GENES} DEGs identified")
print(f"  Top gene: {real_deg_results.iloc[0]['names']} (adj p-val: {real_deg_results.iloc[0]['pvals_adj']:.2e})")

# Save real DEG results
real_output_dir = OUTPUT_DIR / "Real"
real_output_dir.mkdir(exist_ok=True)
real_deg_results.to_csv(real_output_dir / 'deg_results_Real.csv', index=False)

# Get the set of real DEGs for comparison
real_deg_genes = set(real_deg_results['names'].tolist())

# =============================================================================
# HELPER FUNCTION FOR SYNTHETIC MODEL ANALYSIS
# =============================================================================
def analyze_synthetic_model(adata_synthetic, model_name, real_deg_genes, real_deg_results):
    """
    Analyze a synthetic model and compare with real DEGs.

    Returns:
        dict with overlap_count and other results
    """
    print(f"\n{'='*80}")
    print(f"Running {model_name} DEG Analysis")
    print(f"{'='*80}")

    # Create output directory for this model
    model_output_dir = OUTPUT_DIR / model_name
    model_output_dir.mkdir(exist_ok=True)

    # Prepare synthetic data
    adata_synth = adata_synthetic.copy()
    adata_synth.obs[COLUMN_NAME] = adata_synth.obs[COLUMN_NAME].astype(str)
    adata_synth.X = sparse.csr_matrix(adata_synth.X)

    # Run DEG on synthetic data
    sc.tl.rank_genes_groups(
        adata_synth,
        COLUMN_NAME,
        method=METHOD,
        reference=GROUP1,
        use_raw=False
    )

    # Extract synthetic DEG results
    synth_deg_results = sc.get.rank_genes_groups_df(adata_synth, group=GROUP2)
    synth_deg_results['abs_logfoldchanges'] = np.abs(synth_deg_results['logfoldchanges'])
    synth_deg_results = synth_deg_results.sort_values(['pvals_adj', 'abs_logfoldchanges'], ascending=[True, False])
    synth_deg_results = synth_deg_results.head(N_GENES)

    print(f"✓ {model_name} DEG analysis complete")
    print(f"  Top gene: {synth_deg_results.iloc[0]['names']} (adj p-val: {synth_deg_results.iloc[0]['pvals_adj']:.2e})")

    # Save synthetic DEG results
    synth_deg_results.to_csv(model_output_dir / f'deg_results_{model_name}.csv', index=False)

    # Calculate overlap with real DEGs
    synth_deg_genes = set(synth_deg_results['names'].tolist())
    overlap_genes = real_deg_genes.intersection(synth_deg_genes)
    overlap_count = len(overlap_genes)

    unique_to_real = real_deg_genes - synth_deg_genes
    unique_to_synth = synth_deg_genes - real_deg_genes

    jaccard_index = overlap_count / len(real_deg_genes.union(synth_deg_genes))
    overlap_percentage = (overlap_count / len(real_deg_genes)) * 100

    print(f"\n  Overlap Results:")
    print(f"    Shared DEGs: {overlap_count} / {N_GENES}")
    print(f"    Jaccard Index: {jaccard_index:.4f}")
    print(f"    Overlap %: {overlap_percentage:.2f}%")

    # Save overlap results
    overlap_summary = pd.DataFrame([
        {'Metric': 'Top N genes analyzed', 'Value': N_GENES},
        {'Metric': 'Real DEGs', 'Value': len(real_deg_genes)},
        {'Metric': f'{model_name} DEGs', 'Value': len(synth_deg_genes)},
        {'Metric': 'Overlapping genes', 'Value': overlap_count},
        {'Metric': 'Jaccard Index', 'Value': f"{jaccard_index:.4f}"},
        {'Metric': 'Overlap % (Real)', 'Value': f"{overlap_percentage:.2f}%"},
    ])
    overlap_summary.to_csv(model_output_dir / 'overlap_summary.csv', index=False)

    # Save gene lists
    pd.DataFrame({'Gene': sorted(list(overlap_genes))}).to_csv(
        model_output_dir / 'overlapping_genes.csv', index=False)
    pd.DataFrame({'Gene': sorted(list(unique_to_real))}).to_csv(
        model_output_dir / 'unique_to_Real.csv', index=False)
    pd.DataFrame({'Gene': sorted(list(unique_to_synth))}).to_csv(
        model_output_dir / f'unique_to_{model_name}.csv', index=False)

    # Create comparison DataFrame for overlapping genes
    if len(overlap_genes) > 0:
        overlap_data_real = real_deg_results[real_deg_results['names'].isin(overlap_genes)].copy()
        overlap_data_synth = synth_deg_results[synth_deg_results['names'].isin(overlap_genes)].copy()

        comparison_df = overlap_data_real[['names', 'logfoldchanges', 'pvals_adj']].merge(
            overlap_data_synth[['names', 'logfoldchanges', 'pvals_adj']],
            on='names',
            suffixes=('_Real', f'_{model_name}')
        )
        comparison_df.to_csv(model_output_dir / 'overlapping_genes_comparison.csv', index=False)

    return {
        'model_name': model_name,
        'overlap_count': overlap_count,
        'overlap_genes': overlap_genes,
        'jaccard_index': jaccard_index,
        'synth_deg_results': synth_deg_results
    }

# =============================================================================
# RUN ANALYSIS FOR EACH SYNTHETIC MODEL
# =============================================================================
results_cvae = analyze_synthetic_model(adata_cvae, "CVAE", real_deg_genes, real_deg_results)
results_diffusion = analyze_synthetic_model(adata_diffusion, "Diffusion", real_deg_genes, real_deg_results)
results_gan = analyze_synthetic_model(adata_gan, "GAN", real_deg_genes, real_deg_results)

# =============================================================================
# CREATE SHARED DEG COMPARISON PLOT
# =============================================================================
print("\n" + "="*80)
print("Creating Shared DEG Comparison Plot")
print("="*80)

# Color scheme
COLORS = {
    'diffusion': 'lightcoral',
    'cvae': '#8E44AD',
    'gan': '#E69F00',
}

# Plot settings
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.linewidth': 1.2,
})

# Data for plot
models = ['CVAE', 'Diffusion', 'GAN']
shared_genes = [
    results_cvae['overlap_count'],
    results_diffusion['overlap_count'],
    results_gan['overlap_count']
]
colors = [COLORS['cvae'], COLORS['diffusion'], COLORS['gan']]

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Create bar plot
bars = ax.bar(models, shared_genes, color=colors, alpha=0.8,
              edgecolor='black', linewidth=1.5, width=0.5)

# Customize plot
ax.set_ylabel('Number of Shared Genes', fontsize=16, fontweight='bold')
ax.set_title(f'Shared Differentially Expressed Genes\nSynthetic vs Real Data for {DATASET_NAME}',
             fontsize=18, fontweight='bold', pad=25)

# Set y-axis limits
ax.set_ylim(0, max(shared_genes) + 10)

# Add grid
ax.grid(True, alpha=0.3, axis='y', linewidth=1)
ax.set_axisbelow(True)

# Style the axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Add value labels on top of bars
for bar, value in zip(bars, shared_genes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{value}', ha='center', va='bottom',
            fontweight='bold', fontsize=16)

ax.set_facecolor('#FAFAFA')
plt.tight_layout()

# Save plot
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_filename = OUTPUT_DIR / f"shared_deg_comparison_{timestamp}.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Plot saved as: {plot_filename}")

plt.show()

# =============================================================================
# PRINT FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*80)
print(f"\nResults saved to: {OUTPUT_DIR}/")
print(f"  - Real/           : Real data DEG results")
print(f"  - CVAE/           : CVAE vs Real comparison ({results_cvae['overlap_count']} shared DEGs)")
print(f"  - Diffusion/      : Diffusion vs Real comparison ({results_diffusion['overlap_count']} shared DEGs)")
print(f"  - GAN/            : GAN vs Real comparison ({results_gan['overlap_count']} shared DEGs)")
print(f"  - {plot_filename.name} : Comparison plot")

print("\n" + "-"*80)
print("SHARED DEG COUNTS:")
print("-"*80)
print(f"  Diffusion: {results_diffusion['overlap_count']} shared genes")
print(f"  CVAE:      {results_cvae['overlap_count']} shared genes")
print(f"  GAN:       {results_gan['overlap_count']} shared genes")

# Find best model
best_model = max(
    [('Diffusion', results_diffusion['overlap_count']),
     ('CVAE', results_cvae['overlap_count']),
     ('GAN', results_gan['overlap_count'])],
    key=lambda x: x[1]
)
print(f"\nBest performance: {best_model[0]} ({best_model[1]} shared genes)")
print("="*80)
