"""
Cancer-specific script to load cancer AnnData, analyze conditions,
generate synthetic data using diffusion model, and save results.
"""

import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import os
import sys
import json
import torch
from typing import List, Dict, Optional
from datetime import datetime


# Add diffusion directory to path to import functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'diffusion'))

from diffusion_model_cfg import ClassifierFreeDiffusionModel
from vae_diffusion import DiffusionIntegrationVAE
from diffusion_generation_utils import load_model_by_checkpoint


def get_condition_combinations_cancer(data, condition_columns: List[str]):
    """
    Get all unique combinations of categorical conditions and their counts for cancer data.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with categorical condition columns (e.g., from adata.obs)
    condition_columns : List[str]
        List of condition column names to use (e.g., ['annotation_major', 'tissue'])

    Returns:
    --------
    tuple : (list_of_condition_dicts, list_of_counts)
    """
    # Work with a copy
    data_copy = data.copy()

    # Ensure all condition columns are strings
    for col in condition_columns:
        if col not in data_copy.columns:
            raise ValueError(f"Column {col} not found in data! Available: {list(data_copy.columns)}")
        data_copy[col] = data_copy[col].astype(str)

    # Get the subset of data with only the condition columns
    condition_data = data_copy[condition_columns]

    # Get value counts for all unique combinations
    combination_counts = condition_data.value_counts().sort_values(ascending=False)

    # Convert to list of dictionaries and counts
    condition_dicts = []
    counts = []

    for combination, count in combination_counts.items():
        # Create dictionary for this combination
        condition_dict = {}
        for i, col in enumerate(condition_columns):
            condition_dict[col] = combination[i]

        condition_dicts.append(condition_dict)
        counts.append(count)

    return condition_dicts, counts


def save_synthetic_cancer_data(
    generated_data_list: List[np.ndarray],
    condition_dicts: List[Dict],
    num_samples_list: List[int],
    output_path: str,
    gene_names: Optional[List[str]] = None
) -> ad.AnnData:
    """
    Save synthetic cancer data as AnnData object with categorical condition columns.
    """
    print("Creating synthetic cancer AnnData object...")

    # Concatenate all generated data
    X_combined = np.vstack(generated_data_list)
    n_cells, n_genes = X_combined.shape

    print(f"Combined data shape: {X_combined.shape} (cells x genes)")

    # Create gene names if not provided
    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(n_genes)]

    # Create obs (cell metadata) from condition dictionaries
    print("Creating cell metadata...")
    obs_data = []

    for condition_dict, data_array in zip(condition_dicts, generated_data_list):
        n_samples = data_array.shape[0]

        # Create metadata for this condition - use the actual condition dict keys
        for _ in range(n_samples):
            cell_metadata = condition_dict.copy()  # This preserves the original column names
            obs_data.append(cell_metadata)

    # Convert to DataFrame
    obs_df = pd.DataFrame(obs_data)
    obs_df.index = [f"Cell_{i}" for i in range(n_cells)]

    print(f"Created obs with shape: {obs_df.shape}")
    print(f"Obs columns: {list(obs_df.columns)}")

    # Create simple var DataFrame
    var_df = pd.DataFrame(index=gene_names)
    var_df['gene_ids'] = gene_names

    print(f"Created var with shape: {var_df.shape}")

    # Create AnnData object
    adata_synthetic = ad.AnnData(
        X=X_combined,
        obs=obs_df,
        var=var_df
    )

    print(f"Created AnnData object: {adata_synthetic}")

    # Save to file
    print(f"Saving AnnData to {output_path}...")
    adata_synthetic.write_h5ad(output_path)
    print(f"Successfully saved to {output_path}")

    return adata_synthetic


def generate_cancer_synthetic_data(
    conditions: Dict,
    num_samples: int,
    vae_model: DiffusionIntegrationVAE,
    generator: ClassifierFreeDiffusionModel,
    guidance_scale: float = 2.0
) -> np.ndarray:
    """
    Generate synthetic cancer data for a single condition.

    Parameters:
    -----------
    conditions : Dict
        Condition dictionary (e.g., {'annotation_major': 'T cells', 'tissue': 'lung'})
    num_samples : int
        Number of samples to generate
    vae_model : DiffusionIntegrationVAE
        VAE model for latent space conversion
    generator : ClassifierFreeDiffusionModel
        Diffusion model for generation
    guidance_scale : float
        Guidance scale for generation

    Returns:
    --------
    np.ndarray
        Generated synthetic data array
    """
    print(f"  Generating {num_samples} samples with conditions: {conditions}")

    # Generate synthetic data
    generated_decoded = generator.generate_and_decode(
        conditions=conditions,
        vae_model=vae_model,
        num_samples=num_samples,
        guidance_scale=guidance_scale
    )

    return generated_decoded


def generate_multiple_conditions_cancer(
    condition_dicts: List[Dict],
    num_samples_list: List[int],
    vae_model: DiffusionIntegrationVAE,
    generator: ClassifierFreeDiffusionModel,
    guidance_scale: float = 2.0
) -> List[np.ndarray]:
    """
    Generate synthetic cancer data for multiple conditions.

    Parameters:
    -----------
    condition_dicts : List[Dict]
        List of condition dictionaries
    num_samples_list : List[int]
        Number of samples for each condition
    vae_model : DiffusionIntegrationVAE
        VAE model
    generator : ClassifierFreeDiffusionModel
        Diffusion model
    guidance_scale : float
        Guidance scale

    Returns:
    --------
    List[np.ndarray]
        List of generated data arrays
    """
    print(f"Generating data for {len(condition_dicts)} different conditions...")

    all_generated_data = []

    for i, (conditions, num_samples) in enumerate(zip(condition_dicts, num_samples_list)):
        print(f"\nCondition {i+1}/{len(condition_dicts)}:")
        data = generate_cancer_synthetic_data(
            conditions=conditions,
            num_samples=num_samples,
            vae_model=vae_model,
            generator=generator,
            guidance_scale=guidance_scale
        )
        all_generated_data.append(data)

    print(f"\nCompleted generation for all {len(condition_dicts)} conditions")

    return all_generated_data


def main():
    """Main execution function"""

    # Configuration
    ADATA_PATH = "cancer_data/cancer_0.997_filtered_hvg_2000_norm.h5ad"
    VAE_MODEL_PATH = "cancer_data/diffusion_model_cancer/diffusion_vae_norm_retry.pt"
    CHECKPOINT_DIR = "cancer_data/diffusion_checkpoints_cancer"
    CHECKPOINT_NUM = 94  # Which checkpoint to load
    CONDITION_COLUMNS = ['annotation_major', 'tissue']
    GUIDANCE_SCALE = 2.0
    SAVE_PREFIX = "diffusion_cancer_synthetic"

    print("🧬 Cancer Diffusion Model Synthetic Data Generation")
    print("=" * 60)

    try:
        # Load the original data
        print(f"\nLoading AnnData from: {ADATA_PATH}")
        adata = sc.read_h5ad(ADATA_PATH)

        print(f"Loaded data shape: {adata.shape}")
        print(f"Observations columns: {list(adata.obs.columns)}")

        # Verify required columns exist
        missing_columns = [col for col in CONDITION_COLUMNS if col not in adata.obs.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Get all unique condition combinations and their counts
        print("\nAnalyzing condition combinations...")
        condition_dicts, counts = get_condition_combinations_cancer(adata.obs, CONDITION_COLUMNS)

        print(f"Found {len(condition_dicts)} unique condition combinations:")
        for i, (condition, count) in enumerate(zip(condition_dicts[:10], counts[:10])):
            print(f"  {i+1}: {condition} - {count} samples")
        if len(condition_dicts) > 10:
            print(f"  ... and {len(condition_dicts) - 10} more combinations")

        # Setup VAE for latent space conversion
        print("\nSetting up VAE...")
        vae_model = DiffusionIntegrationVAE(
            n_genes=adata.n_vars,
            latent_size=128,
            hidden_dims=[1024, 1024],
            model=VAE_MODEL_PATH,
            verbose=True,
            auto_load=True
        )

        # Load diffusion generator
        print(f"\nLoading diffusion model from checkpoint {CHECKPOINT_NUM}...")
        generator = load_model_by_checkpoint(
            directory=CHECKPOINT_DIR,
            checkpoint_num=CHECKPOINT_NUM,
            model_class=ClassifierFreeDiffusionModel
        )

        print("Models loaded successfully!")

        # Generate synthetic data for all conditions
        print(f"\nGenerating synthetic data for all {len(condition_dicts)} conditions...")
        generated_data_list = generate_multiple_conditions_cancer(
            condition_dicts=condition_dicts,
            num_samples_list=counts,
            vae_model=vae_model,
            generator=generator,
            guidance_scale=GUIDANCE_SCALE
        )

        # Save as AnnData
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{SAVE_PREFIX}_{timestamp}.h5ad"

        print("\nSaving synthetic data as AnnData...")
        adata_synthetic = save_synthetic_cancer_data(
            generated_data_list=generated_data_list,
            condition_dicts=condition_dicts,
            num_samples_list=counts,
            output_path=output_path,
            gene_names=adata.var_names.tolist()
        )

        # Convert list to a single array by concatenating
        print("\nCombining generated data for numpy save...")
        all_generated_data = np.concatenate(generated_data_list, axis=0)

        print(f"Final generated data shape: {all_generated_data.shape}")

        # Save the generated array
        npy_output = f"{SAVE_PREFIX}_{timestamp}.npy"
        print(f"\nSaving generated numpy array to {npy_output}...")
        np.save(npy_output, all_generated_data)

        # Also save condition information for reference
        summary_filename = f"{SAVE_PREFIX}_summary_{timestamp}.json"
        summary = {
            'original_data_shape': list(adata.shape),
            'synthetic_data_shape': list(adata_synthetic.shape),
            'condition_columns': CONDITION_COLUMNS,
            'n_condition_combinations': len(condition_dicts),
            'total_samples_generated': all_generated_data.shape[0],
            'n_genes': adata.n_vars,
            'vae_model_path': VAE_MODEL_PATH,
            'checkpoint_dir': CHECKPOINT_DIR,
            'checkpoint_num': CHECKPOINT_NUM,
            'guidance_scale': GUIDANCE_SCALE,
            'timestamp': timestamp,
            'h5ad_output': output_path,
            'npy_output': npy_output,
            'condition_combinations': [
                {'condition': condition_dict, 'count': count}
                for condition_dict, count in zip(condition_dicts, counts)
            ]
        }

        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved summary information to {summary_filename}")

        print("\n✅ Generation completed successfully!")
        print(f"📊 Generated synthetic AnnData: {adata_synthetic}")
        print(f"📁 AnnData saved to: {output_path}")
        print(f"📁 Numpy array saved to: {npy_output}")
        print(f"📋 Summary saved to: {summary_filename}")

        # Display some basic statistics
        print("\n📈 Basic Statistics:")
        print(f"  - Original data: {adata.shape[0]} cells × {adata.shape[1]} genes")
        print(f"  - Synthetic data: {adata_synthetic.shape[0]} cells × {adata_synthetic.shape[1]} genes")

        # Show condition distribution
        print("\n🏷️  Condition Distribution (top 10):")
        condition_summary = adata_synthetic.obs[CONDITION_COLUMNS].value_counts().head(10)
        for i, (conditions, count) in enumerate(condition_summary.items()):
            condition_dict = dict(zip(CONDITION_COLUMNS, conditions))
            print(f"  {i + 1}. {condition_dict}: {count} samples")

        print("\nDone!")

        return adata_synthetic

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
