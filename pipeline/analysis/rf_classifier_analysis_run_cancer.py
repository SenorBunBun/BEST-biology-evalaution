#!/usr/bin/env python3
"""
Random Forest Classifier Analysis for Cancer Data

This script uses the RF classifier pipeline to analyze cancer data,
comparing real data against three synthetic models (CVAE, Diffusion, GAN)
to identify overlapping phenotypically predictive genes.

Classification task: Metastasis Tumor vs Metastasis Normal
"""

from rf_classifier_gene_pipeline import compare_real_vs_all_models

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define paths (all in cancer_data directory)
REAL_PATH = 'cancer_data/cancer_0.997_filtered_hvg_2000_norm.h5ad'
CVAE_PATH = 'cancer_data/cvae_cancer_relu_raw_filtered_0.997_800_400_32.pt_20251106_013022.h5ad'
DIFFUSION_PATH = 'cancer_data/diffusion_cancer_synthetic_20251106_090734.h5ad'
GAN_PATH = 'cancer_data/cancer_gan_output.h5ad'

# Output directory
OUTPUT_DIR = 'cancer_data/rf_predictive_genes_analysis'

# Classification parameters
COLUMN_NAME = "tissue"
GROUP1 = "metastasis normal"  # Reference group (encoded as 0)
GROUP2 = "metastasis tumor"   # Target group (encoded as 1)

# Analysis parameters
N_GENES = 50  # Number of top predictive genes to analyze

# Random Forest hyperparameters (balanced settings)
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'n_folds': 5,
    'n_repeats': 3,
    'random_state': 42
}

# Plot title
PLOT_TITLE = "Overlap of Phenotypically Predictive Genes:\nMetastasis Tumor vs Normal Classification"

# =============================================================================
# RUN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CANCER DATA RF CLASSIFICATION ANALYSIS")
    print("="*80)
    print(f"\nClassification Task: {GROUP2} vs {GROUP1}")
    print(f"Column: {COLUMN_NAME}")
    print(f"Top N genes: {N_GENES}")
    print(f"\nDatasets:")
    print(f"  Real:      {REAL_PATH}")
    print(f"  CVAE:      {CVAE_PATH}")
    print(f"  Diffusion: {DIFFUSION_PATH}")
    print(f"  GAN:       {GAN_PATH}")
    print(f"\nOutput: {OUTPUT_DIR}")
    print("="*80)

    # Run the complete analysis
    results = compare_real_vs_all_models(
        real_path=REAL_PATH,
        cvae_path=CVAE_PATH,
        diffusion_path=DIFFUSION_PATH,
        gan_path=GAN_PATH,
        output_base_dir=OUTPUT_DIR,
        column_name=COLUMN_NAME,
        group1=GROUP1,
        group2=GROUP2,
        n_genes=N_GENES,
        plot_title=PLOT_TITLE,
        **RF_PARAMS
    )

    # Print final summary
    print("\n" + "="*80)
    print("CANCER DATA RF CLASSIFICATION ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n✓ All results saved to: {OUTPUT_DIR}")
    print("\nKey outputs:")
    print(f"  • Individual comparisons in subdirectories (cvae_comparison/, diffusion_comparison/, gan_comparison/)")
    print(f"  • Combined overlap chart: {OUTPUT_DIR}/all_models_overlap_comparison.png")
    print(f"  • Summary table: {OUTPUT_DIR}/all_models_summary.csv")
    print("\nOverlap Results:")
    for model in ['CVAE', 'Diffusion', 'GAN']:
        count = results['overlap_counts'][model]
        percentage = (count / N_GENES) * 100
        print(f"  {model:10s}: {count:3d}/{N_GENES} genes ({percentage:5.1f}%)")
    print("="*80 + "\n")
