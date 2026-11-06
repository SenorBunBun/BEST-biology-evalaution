import scanpy as sc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scanpy as sc
import anndata as ad
import time
import json
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional


from src.unbatch_vanilla_cvae_relu_qol_categorical import Enhanced_Vanilla_CVAE_Categorical

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'diffusion'))


class CategoricalConditionEncoder:
    """Handles encoding of categorical conditions to numerical format for CVAE (Cancer data)"""

    def __init__(self, condition_columns: List[str]):
        """
        Initialize encoder with specified condition columns.

        Parameters:
        -----------
        condition_columns : List[str]
            List of categorical condition column names (e.g., ['annotation_major', 'tissue'])
        """
        self.encoders = {}
        self.condition_columns = condition_columns
        print(f"Initialized CategoricalConditionEncoder with columns: {condition_columns}")


    def fit(self, condition_dicts):
        """Fit encoders on the condition data"""
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(condition_dicts)

        # Fit label encoders for all condition columns
        for col in self.condition_columns:
            if col in df.columns:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(df[col].astype(str))
                print(f"Encoder for {col}: {self.encoders[col].classes_}")
            else:
                print(f"WARNING: Column {col} not found in condition data!")


    def encode_conditions(self, condition_dicts):
        """Encode condition dictionaries to numerical arrays"""
        encoded_conditions = []

        for condition_dict in condition_dicts:
            encoded_condition = []

            for col in self.condition_columns:
                if col in condition_dict and col in self.encoders:
                    encoded_val = self.encoders[col].transform([str(condition_dict[col])])[0]
                    encoded_condition.append(encoded_val)
                else:
                    raise ValueError(f"Column {col} missing from condition dict or not fitted!")

            encoded_conditions.append(encoded_condition)

        return np.array(encoded_conditions, dtype=np.float32)

    def get_n_labels(self):
        """Get the total number of label dimensions (number of categorical columns)"""
        return len(self.condition_columns)


def save_synthetic_cancer_data(
    generated_data_list: List[np.ndarray],
    condition_dicts: List[Dict],
    num_samples_list: List[int],
    output_path: str,
    gene_names: Optional[List[str]] = None,
    condition_columns: Optional[List[str]] = None
) -> ad.AnnData:
    """
    Save synthetic cancer data as AnnData object with categorical condition columns.

    Parameters:
    -----------
    generated_data_list : List[np.ndarray]
        List of generated data arrays
    condition_dicts : List[Dict]
        List of condition dictionaries corresponding to each generated dataset
    num_samples_list : List[int]
        Number of samples for each condition
    output_path : str
        Path to save the .h5ad file
    gene_names : Optional[List[str]]
        List of gene names
    condition_columns : Optional[List[str]]
        List of condition column names (e.g., ['annotation_major', 'tissue'])

    Returns:
    --------
    ad.AnnData
        The created AnnData object
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


def get_condition_combinations_cancer(data, condition_columns: List[str]):
    """
    Get all unique combinations of metadata conditions and their counts for cancer data.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with categorical condition columns (e.g., from adata.obs)
    condition_columns : List[str]
        List of condition column names to use (e.g., ['annotation_major', 'tissue'])

    Returns:
    --------
    tuple : (list_of_condition_dicts, list_of_counts)
        - list_of_condition_dicts: List of dictionaries, each containing a unique combination
        - list_of_counts: List of integers representing count for each combination

    Example:
    --------
    >>> condition_dicts, counts = get_condition_combinations_cancer(adata.obs, ['annotation_major', 'tissue'])
    >>> print(condition_dicts[0])
    {'annotation_major': 'T cells', 'tissue': 'lung'}
    >>> print(counts[0])
    150
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


def generate_synthetic_data_with_cvae_cancer(
    adata,
    model_path,
    condition_columns: List[str],
    latent_size=32,
    save_prefix="synthetic_cancer",
    device=None
):
    """
    Generate synthetic cancer data using Enhanced_Vanilla_CVAE_Categorical based on original data conditions.

    Parameters:
    -----------
    adata : AnnData
        Original single-cell cancer data
    model_path : str
        Path to the trained CVAE model
    condition_columns : List[str]
        List of categorical condition column names (e.g., ['annotation_major', 'tissue'])
    latent_size : int
        Latent dimension size (must match trained model)
    save_prefix : str
        Prefix for saved files
    device : torch.device
        Device for computation

    Returns:
    --------
    tuple : (adata_synthetic, summary_filename)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"Original data shape: {adata.shape}")
    print(f"Condition columns: {condition_columns}")

    # Step 1: Get condition combinations from original data
    print("\nStep 1: Analyzing condition combinations...")
    condition_dicts, counts = get_condition_combinations_cancer(adata.obs, condition_columns)

    print(f"Found {len(condition_dicts)} unique condition combinations")
    for i, (condition_dict, count) in enumerate(zip(condition_dicts[:5], counts[:5])):
        print(f"  {i + 1}. {condition_dict} (count: {count})")
    if len(condition_dicts) > 5:
        print(f"  ... and {len(condition_dicts) - 5} more combinations")

    # Step 2: Set up condition encoder
    print("\nStep 2: Setting up condition encoder...")
    encoder = CategoricalConditionEncoder(condition_columns)
    encoder.fit(condition_dicts)
    n_labels = encoder.get_n_labels()
    print(f"Number of label dimensions: {n_labels}")

    # Step 3: Initialize and load the CVAE model
    print("\nStep 3: Loading CVAE model...")
    n_genes = adata.n_vars

    model = Enhanced_Vanilla_CVAE_Categorical(
        n_genes=n_genes,
        n_labels=n_labels,
        latent_size=latent_size,
        device=device,
        condition_cols=condition_columns
    )

    # Load model weights
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully!")

        # Try to load the saved label encoders from the model
        try:
            encoder_path = model_path.replace('.pt', '_label_encoders.pkl')
            if os.path.exists(encoder_path):
                print(f"Loading label encoders from: {encoder_path}")
                import pickle
                with open(encoder_path, 'rb') as f:
                    saved_encoders = pickle.load(f)
                # Use the saved encoders instead of fitting new ones
                encoder.encoders = saved_encoders
                print("Label encoders loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load saved label encoders: {e}")
            print("Using newly fitted encoders instead.")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Step 4: Generate synthetic data organized by condition combinations
    print("\nStep 4: Generating synthetic data...")
    generated_data_list = []
    batch_size = 50  # Smaller batch size for memory management

    for i, (condition_dict, count) in enumerate(zip(condition_dicts, counts)):
        print(f"  Generating condition {i + 1}/{len(condition_dicts)}: {condition_dict} ({count} samples)")

        # Encode the condition
        encoded_condition = encoder.encode_conditions([condition_dict])[0]
        condition_tensor = torch.tensor(encoded_condition, dtype=torch.float32).to(device)

        # Generate samples for this condition in batches
        condition_samples = []

        for j in range(0, count, batch_size):
            current_batch_size = min(batch_size, count - j)

            # Generate batch of samples for this condition
            with torch.no_grad():
                # Expand condition tensor for batch generation
                batch_condition = condition_tensor.unsqueeze(0).repeat(current_batch_size, 1)

                # Generate samples
                synthetic_batch = model.generate(batch_condition[0:1], num_samples=current_batch_size)
                synthetic_batch = synthetic_batch.cpu()

                synthetic_batch = torch.clamp(synthetic_batch, min=0).numpy()  # Clipping to 0 during inference

                condition_samples.append(synthetic_batch)

        # Combine all batches for this condition
        condition_data = np.vstack(condition_samples)
        generated_data_list.append(condition_data)

        print(f"    Generated {condition_data.shape[0]} samples with shape {condition_data.shape[1]} genes")

    print(f"Generated data for {len(generated_data_list)} condition combinations")

    # Step 5: Create AnnData object using cancer-specific saving function
    print("\nStep 5: Creating AnnData object...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{save_prefix}_{timestamp}.h5ad"

    adata_synthetic = save_synthetic_cancer_data(
        generated_data_list=generated_data_list,
        condition_dicts=condition_dicts,
        num_samples_list=counts,
        output_path=output_path,
        gene_names=adata.var_names.tolist(),
        condition_columns=condition_columns
    )

    # Step 6: Save summary statistics
    summary_filename = f"{save_prefix}_generation_summary_{timestamp}.json"
    summary = {
        'original_data_shape': list(adata.shape),
        'synthetic_data_shape': list(adata_synthetic.shape),
        'n_condition_combinations': len(condition_dicts),
        'total_synthetic_samples': sum(counts),
        'n_genes': n_genes,
        'n_labels': n_labels,
        'condition_columns': condition_columns,
        'model_path': model_path,
        'latent_size': latent_size,
        'device': str(device),
        'timestamp': timestamp,
        'output_path': output_path,
        'condition_combinations': [
            {'condition': condition_dict, 'count': count}
            for condition_dict, count in zip(condition_dicts, counts)
        ]
    }

    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved generation summary to: {summary_filename}")

    return adata_synthetic, summary_filename


def main():
    """Main execution function"""

    # Configuration
    ADATA_PATH = "cancer_data/cancer_0.997_filtered_hvg_2000_norm.h5ad"
    MODEL_PATH = "cancer_data/trained_models_cancer/vanilla_cvae_relu_raw_filtered_0.997_800_400_32.pt"
    SAVE_PREFIX = "cvae_cancer_relu_raw_filtered_0.997_800_400_32.pt"
    CONDITION_COLUMNS = ['annotation_major', 'tissue']
    LATENT_SIZE = 32  # Must match the trained model

    print("🧬 Enhanced CVAE Synthetic Cancer Data Generation")
    print("=" * 50)

    try:
        # Load the original data
        print(f"Loading data from: {ADATA_PATH}")
        adata = sc.read_h5ad(ADATA_PATH)
        print(f"Loaded data shape: {adata.shape}")
        print(f"Available metadata columns: {list(adata.obs.columns)}")

        # Verify required columns exist
        missing_columns = [col for col in CONDITION_COLUMNS if col not in adata.obs.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        print(f"Condition columns found: {CONDITION_COLUMNS}")

        # Generate synthetic data
        adata_synthetic, summary_filename = generate_synthetic_data_with_cvae_cancer(
            adata=adata,
            model_path=MODEL_PATH,
            condition_columns=CONDITION_COLUMNS,
            latent_size=LATENT_SIZE,
            save_prefix=SAVE_PREFIX
        )

        print("\n✅ Generation completed successfully!")
        print(f"📊 Generated synthetic AnnData: {adata_synthetic}")
        print(f"📁 AnnData saved to: {adata_synthetic.filename}")
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

        # Show basic info about the synthetic AnnData
        print(f"\n🧬 Synthetic AnnData Info:")
        print(f"  - Observation columns: {list(adata_synthetic.obs.columns)}")
        print(f"  - Variable columns: {list(adata_synthetic.var.columns)}")
        print(f"  - Data type: {type(adata_synthetic.X)}")
        print(f"  - Data shape: {adata_synthetic.X.shape}")

        return adata_synthetic

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
