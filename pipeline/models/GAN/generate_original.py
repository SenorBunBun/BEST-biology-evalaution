#!/usr/bin/env python
"""
Generation script for Multi-Conditional scGAN (Cancer Data)
Generates synthetic cancer single-cell data with annotation_major and tissue conditions
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import scanpy as sc
import anndata as ad
from datetime import datetime
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_cond.utils.condition_processor_cancer import ConditionProcessorCancer
from multi_cond.models import build_generator


class SyntheticCancerDataGenerator:
    """Generate synthetic cancer single-cell data using trained model"""

    def __init__(self, checkpoint_dir, config_path=None):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        print(f"\n{'='*60}")
        print(f"Initializing Cancer Data Generator")
        print(f"{'='*60}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")

        # Load configuration
        if config_path:
            print(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Try to load from checkpoint directory
            state_path = os.path.join(checkpoint_dir, "training_state.json")
            print(f"Looking for config at: {state_path}")
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    self.config = state['config']
                print("✅ Loaded config from checkpoint")
            else:
                raise ValueError("Configuration not found. Please provide config_path")

        # Load condition processor
        print("\nLoading condition processor...")
        self.condition_processor = ConditionProcessorCancer()

        # Check both possible locations
        cond_config_path_1 = os.path.join(os.path.dirname(checkpoint_dir), "condition_config.json")
        cond_config_path_2 = os.path.join(checkpoint_dir, "condition_config.json")

        print(f"  Looking for condition config at:")
        print(f"    1) {cond_config_path_1}")
        print(f"    2) {cond_config_path_2}")

        if os.path.exists(cond_config_path_1):
            print(f"  ✅ Found at location 1")
            self.condition_processor.load_config(cond_config_path_1)
        elif os.path.exists(cond_config_path_2):
            print(f"  ✅ Found at location 2")
            self.condition_processor.load_config(cond_config_path_2)
        else:
            print("⚠️  Warning: Condition config not found at either location!")
            print("    Will need to fit from reference data.")

        if self.condition_processor.is_fitted:
            print(f"✅ ConditionProcessorCancer loaded")
            print(f"  Condition names: {self.condition_processor.condition_names}")
            print(f"  Total dimension: {self.condition_processor.total_condition_dim}")

        self.generator = None
        self.reference_adata = None
        print(f"{'='*60}\n")

    def load_model(self):
        """Load trained generator model"""
        print("\nLoading trained model...")
        print("="*60)

        # Debug info
        print("Configuration:")
        print(f"  Latent dim: {self.config['model']['latent_dim']}")
        print(f"  Gene dim: {self.config['data']['gene_dim']}")
        print(f"  Generator layers: {self.config['model']['gen_layers']}")

        # Check if processor is fitted - if not, we need to fit it from reference data first
        if not self.condition_processor.is_fitted:
            print("\n⚠️  Condition processor not fitted - need to load reference data first!")
            print("    Call load_reference_data() before load_model()")
            raise ValueError("Condition processor is not fitted! Please load reference data first with load_reference_data()")

        print(f"  Condition dim from processor: {self.condition_processor.total_condition_dim}")

        # Show condition details
        print(f"\nCondition Processor Info:")
        print(f"  Condition names: {self.condition_processor.condition_names}")
        print(f"  Condition dims: {self.condition_processor.condition_dims}")
        print(f"  Total condition dim: {self.condition_processor.total_condition_dim}")

        # Build generator
        condition_dim = self.condition_processor.total_condition_dim
        latent_dim = self.config['model']['latent_dim']
        gene_dim = self.config['data']['gene_dim']

        print(f"\nBuilding generator:")
        print(f"  Input: latent_dim={latent_dim} + condition_dim={condition_dim}")
        print(f"  Output: gene_dim={gene_dim}")

        self.generator = build_generator(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            output_dim=gene_dim,
            hidden_layers=self.config['model']['gen_layers']
        )

        # CRITICAL: Call the model once with dummy inputs to build the layers
        # This initializes all the layer variables before loading weights
        print(f"\nInitializing model with dummy forward pass...")
        dummy_noise = tf.random.normal((1, latent_dim))
        dummy_conditions = tf.random.normal((1, condition_dim))
        dummy_output = self.generator((dummy_noise, dummy_conditions), training=False)
        print(f"  Dummy output shape: {dummy_output.shape}")
        print(f"✅ Generator built with {self.generator.count_params():,} parameters")

        # Load weights
        generator_path = os.path.join(self.checkpoint_dir, "generator.weights.h5")
        if not os.path.exists(generator_path):
            generator_path = os.path.join(self.checkpoint_dir, "generator")

        if not os.path.exists(generator_path):
            raise ValueError(f"Generator weights not found at {generator_path}")

        print(f"\nLoading weights from: {generator_path}")
        self.generator.load_weights(generator_path)
        print(f"✅ Model weights loaded successfully")
        print("="*60)

    def load_reference_data(self, data_path=None):
        """Load reference data to match condition distribution"""
        if data_path is None:
            data_path = self.config['data']['input_file']

        print(f"\nLoading reference data from {data_path}...")
        self.reference_adata = sc.read_h5ad(data_path)
        print(f"✅ Reference data shape: {self.reference_adata.shape}")

        # Fit condition processor if not already fitted
        if not self.condition_processor.is_fitted:
            print("Fitting condition processor to reference data...")
            self.condition_processor.fit(self.reference_adata)

        # Get condition distribution
        self.condition_combos, self.condition_counts = \
            self.condition_processor.get_condition_combinations(self.reference_adata.obs)

        print(f"✅ Found {len(self.condition_combos)} unique condition combinations")

        # Show distribution
        print("\nCondition Distribution:")
        for combo, count in zip(self.condition_combos[:10], self.condition_counts[:10]):
            print(f"  {combo['annotation_major']:20s} | {combo['tissue']:20s} | {count:,} cells")
        if len(self.condition_combos) > 10:
            print(f"  ... and {len(self.condition_combos) - 10} more combinations")

    def generate_matching_distribution(self, scale_factor=1.0):
        """
        Generate synthetic data matching reference condition distribution

        Args:
            scale_factor: Multiply the number of cells per condition by this factor
                         1.0 = same number as reference
                         0.5 = half the cells
                         2.0 = double the cells

        Returns:
            AnnData object with synthetic data
        """
        print(f"\n{'='*60}")
        print(f"Generating synthetic data (scale factor: {scale_factor})")
        print(f"{'='*60}")

        all_expression = []
        all_conditions_raw = []

        # Generate for each condition combination
        for i, (combo, count) in enumerate(zip(self.condition_combos, self.condition_counts)):
            # Calculate number of cells to generate
            n_generate = max(1, int(count * scale_factor))

            print(f"[{i+1}/{len(self.condition_combos)}] Generating {n_generate} cells | "
                  f"{combo['annotation_major']:15s} | {combo['tissue']:15s}")

            # Create dataframe with repeated conditions
            condition_df = pd.DataFrame([combo] * n_generate)

            # Encode conditions
            encoded_conditions = self.condition_processor.encode_conditions(condition_df)

            # Generate in batches if needed
            batch_size = self.config['data']['batch_size']
            generated_expression = []

            for start_idx in range(0, n_generate, batch_size):
                end_idx = min(start_idx + batch_size, n_generate)
                batch_conditions = encoded_conditions[start_idx:end_idx]

                # Generate expression
                batch_expression = self.generator.generate(batch_conditions)
                generated_expression.append(batch_expression.numpy())

            # Combine batches
            expression = np.vstack(generated_expression)[:n_generate]
            all_expression.append(expression)

            # Store raw conditions for metadata
            for _ in range(n_generate):
                all_conditions_raw.append(combo)

        # Combine all generated data
        print("\n✅ Combining generated data...")
        full_expression = np.vstack(all_expression)
        print(f"✅ Total generated cells: {full_expression.shape[0]:,}")

        # Create AnnData object
        synthetic_adata = self.create_anndata(full_expression, all_conditions_raw)

        return synthetic_adata

    def generate_specific_conditions(self, annotation_major, tissue, n_cells=1000):
        """
        Generate synthetic data for a specific annotation_major and tissue combination

        Args:
            annotation_major: Specific annotation_major value (e.g., 'T cell')
            tissue: Specific tissue value (e.g., 'lung')
            n_cells: Number of cells to generate

        Returns:
            AnnData object with synthetic data
        """
        print(f"\n{'='*60}")
        print(f"Generating {n_cells} cells for specific conditions:")
        print(f"  annotation_major: {annotation_major}")
        print(f"  tissue: {tissue}")
        print(f"{'='*60}")

        # Use condition processor's method to sample specific conditions
        encoded_conditions = self.condition_processor.sample_specific_conditions(
            annotation_major=annotation_major,
            tissue=tissue,
            n_samples=n_cells
        )

        # Generate in batches
        batch_size = self.config['data']['batch_size']
        generated_expression = []

        for start_idx in range(0, n_cells, batch_size):
            end_idx = min(start_idx + batch_size, n_cells)
            batch_conditions = encoded_conditions[start_idx:end_idx]

            # Generate expression
            batch_expression = self.generator.generate(batch_conditions)
            generated_expression.append(batch_expression.numpy())

        # Combine batches
        full_expression = np.vstack(generated_expression)[:n_cells]
        print(f"✅ Generated {full_expression.shape[0]:,} cells")

        # Create condition list
        conditions_list = [{'annotation_major': annotation_major, 'tissue': tissue}] * n_cells

        # Create AnnData object
        synthetic_adata = self.create_anndata(full_expression, conditions_list)

        return synthetic_adata

    def create_anndata(self, expression_matrix, conditions_list):
        """
        Create AnnData object with expression and metadata

        Args:
            expression_matrix: Generated expression values (n_cells, n_genes)
            conditions_list: List of condition dictionaries

        Returns:
            AnnData object
        """
        # Create basic AnnData object
        adata = ad.AnnData(X=expression_matrix)

        # Add gene names if available from reference
        if self.reference_adata is not None:
            adata.var_names = self.reference_adata.var_names
            adata.var = self.reference_adata.var.copy()
        else:
            # Create generic gene names
            adata.var_names = [f"Gene_{i}" for i in range(expression_matrix.shape[1])]

        # Add observation metadata
        obs_df = pd.DataFrame(conditions_list)
        adata.obs = obs_df

        # Add generation metadata
        adata.uns['generation_info'] = {
            'generator': 'Multi-Conditional scGAN (Cancer)',
            'timestamp': datetime.now().isoformat(),
            'checkpoint': self.checkpoint_dir,
            'n_cells': expression_matrix.shape[0],
            'n_genes': expression_matrix.shape[1],
            'conditions': ['annotation_major', 'tissue']
        }

        # Add cell names
        adata.obs_names = [f"synthetic_cell_{i}" for i in range(adata.n_obs)]

        return adata

    def save_synthetic_data(self, adata, output_path=None):
        """Save synthetic data to h5ad file"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Ensure the generated directory exists first
            generated_dir = self.config['output']['generated_dir']
            Path(generated_dir).mkdir(parents=True, exist_ok=True)

            output_path = os.path.join(
                generated_dir,
                f"synthetic_cancer_data_{timestamp}.h5ad"
            )

        # Create directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and output_dir != '':
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save AnnData
        adata.write(output_path)
        print(f"\n{'='*60}")
        print(f"✅ Synthetic data saved to: {output_path}")
        print(f"{'='*60}")

        # Print summary
        print("\nGeneration Summary:")
        print(f"  Total cells: {adata.n_obs:,}")
        print(f"  Total genes: {adata.n_vars:,}")

        # Condition distribution
        print("\nCondition Distribution:")
        print(f"\n  annotation_major:")
        for val, count in adata.obs['annotation_major'].value_counts().items():
            print(f"    {val:20s}: {count:,} cells ({count/adata.n_obs*100:.1f}%)")

        print(f"\n  tissue:")
        for val, count in adata.obs['tissue'].value_counts().items():
            print(f"    {val:20s}: {count:,} cells ({count/adata.n_obs*100:.1f}%)")

        # Combination counts
        print(f"\n  Unique combinations: {len(adata.obs[['annotation_major', 'tissue']].drop_duplicates())}")

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic cancer single-cell data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data matching reference distribution
  python multi_cond/generate_cancer.py --checkpoint cancer/best_model --reference data.h5ad

  # Generate 2x the reference size
  python multi_cond/generate_cancer.py --checkpoint cancer/best_model --reference data.h5ad --scale 2.0

  # Generate specific condition combination
  python multi_cond/generate_cancer.py --checkpoint cancer/best_model --specific --annotation "T cell" --tissue "lung" --n-cells 5000
        """
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint directory (e.g., cancer/best_model)')
    parser.add_argument('--reference', type=str,
                        help='Path to reference h5ad file (optional, uses config if not provided)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor for number of cells (1.0 = same as reference, 2.0 = double)')
    parser.add_argument('--output', type=str,
                        help='Output path for synthetic data')
    parser.add_argument('--config', type=str,
                        help='Path to config file (optional, will use checkpoint config if not provided)')

    # Specific condition generation
    parser.add_argument('--specific', action='store_true',
                        help='Generate specific annotation_major/tissue combination')
    parser.add_argument('--annotation', type=str,
                        help='Specific annotation_major value (used with --specific)')
    parser.add_argument('--tissue', type=str,
                        help='Specific tissue value (used with --specific)')
    parser.add_argument('--n-cells', type=int, default=1000,
                        help='Number of cells to generate (used with --specific)')

    args = parser.parse_args()

    # Validate arguments
    if args.specific:
        if not args.annotation or not args.tissue:
            parser.error("--specific requires both --annotation and --tissue")

    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    print("="*60)
    print("Multi-Conditional scGAN - Cancer Data Generator")
    print("="*60)

    # Create generator
    generator = SyntheticCancerDataGenerator(args.checkpoint, args.config)

    # Load model
    generator.load_model()

    if args.specific:
        # Generate specific conditions
        synthetic_adata = generator.generate_specific_conditions(
            annotation_major=args.annotation,
            tissue=args.tissue,
            n_cells=args.n_cells
        )
    else:
        # Load reference data and match distribution
        generator.load_reference_data(args.reference)
        synthetic_adata = generator.generate_matching_distribution(scale_factor=args.scale)

    # Save synthetic data
    generator.save_synthetic_data(synthetic_adata, args.output)

    print("\n✅ Generation complete!")


if __name__ == "__main__":
    main()
