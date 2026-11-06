#!/usr/bin/env python
"""
Evaluate how well the trained model learns and respects conditions (Cancer Data)
Tests annotation_major and tissue conditions
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import silhouette_score
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_cond.utils.condition_processor_cancer import ConditionProcessorCancer
from multi_cond.models import build_generator, build_critic


class CancerConditionEvaluator:
    """Evaluate condition learning in trained Multi-Conditional scGAN for cancer data"""

    def __init__(self, checkpoint_dir, reference_data_path=None, config_path=None):
        self.checkpoint_dir = checkpoint_dir

        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Try to load from checkpoint directory
            state_path = os.path.join(checkpoint_dir, "training_state.json")
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    self.config = state['config']
                print("✅ Loaded config from checkpoint")
            else:
                raise ValueError("Configuration not found")

        # Load reference data
        if reference_data_path is None:
            reference_data_path = self.config['data']['input_file']

        print(f"Loading reference data from {reference_data_path}...")
        self.reference_adata = sc.read_h5ad(reference_data_path)
        print(f"✅ Reference data shape: {self.reference_adata.shape}")

        # Load condition processor
        self.condition_processor = ConditionProcessorCancer()
        cond_config_path = os.path.join(
            os.path.dirname(checkpoint_dir),
            "condition_config.json"
        )

        if os.path.exists(cond_config_path):
            self.condition_processor.load_config(cond_config_path)
            print(f"✅ Loaded ConditionProcessorCancer (dim={self.condition_processor.total_condition_dim})")
        else:
            print("Fitting condition processor to reference data...")
            self.condition_processor.fit(self.reference_adata)

        # Load models
        self.generator = None
        self.critic = None
        self.load_models()

    def load_models(self):
        """Load trained generator and critic models"""
        print("\nLoading trained models...")

        # Build generator and critic
        condition_dim = self.condition_processor.total_condition_dim
        latent_dim = self.config['model']['latent_dim']

        self.generator = build_generator(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            output_dim=self.config['data']['gene_dim'],
            hidden_layers=self.config['model']['gen_layers']
        )

        self.critic = build_critic(
            input_dim=self.config['data']['gene_dim'],
            condition_dim=condition_dim,
            hidden_layers=self.config['model']['critic_layers']
        )

        # Build models with dummy inputs
        dummy_noise = tf.random.normal((1, latent_dim))
        dummy_conditions = tf.random.normal((1, condition_dim))
        dummy_expression = tf.random.normal((1, self.config['data']['gene_dim']))
        _ = self.generator((dummy_noise, dummy_conditions), training=False)
        _ = self.critic((dummy_expression, dummy_conditions), training=False)

        # Load weights
        generator_path = os.path.join(self.checkpoint_dir, "generator.weights.h5")
        critic_path = os.path.join(self.checkpoint_dir, "critic.weights.h5")

        self.generator.load_weights(generator_path)
        self.critic.load_weights(critic_path)
        print(f"✅ Models loaded (Generator: {self.generator.count_params():,} params, "
              f"Critic: {self.critic.count_params():,} params)")

    def evaluate_condition_consistency(self, n_samples_per_condition=100, n_conditions=10):
        """
        Test 1: Condition Consistency
        Generate multiple samples with same conditions and measure variance
        Lower variance = better condition consistency
        """
        print("\n" + "="*60)
        print("TEST 1: Condition Consistency")
        print("="*60)

        # Get random condition combinations from reference data
        unique_conditions = self.reference_adata.obs[['annotation_major', 'tissue']].drop_duplicates()
        if len(unique_conditions) > n_conditions:
            unique_conditions = unique_conditions.sample(n_conditions)

        consistency_scores = []

        for idx, row in unique_conditions.iterrows():
            # Create multiple samples with same conditions
            condition_df = pd.DataFrame([row.to_dict()] * n_samples_per_condition)
            encoded_conditions = self.condition_processor.encode_conditions(condition_df)

            # Generate samples
            noise = tf.random.normal((n_samples_per_condition, self.config['model']['latent_dim']))
            generated = self.generator((noise, encoded_conditions), training=False).numpy()

            # Calculate variance across samples (should be low for same conditions)
            variance = np.mean(np.var(generated, axis=0))
            consistency_scores.append(variance)

            print(f"Condition {idx+1}/{len(unique_conditions)} "
                  f"({row['annotation_major']}, {row['tissue']}): Variance = {variance:.4f}")

        mean_consistency = np.mean(consistency_scores)
        std_consistency = np.std(consistency_scores)

        print(f"\n✅ Overall Consistency Score:")
        print(f"  Mean variance: {mean_consistency:.4f}")
        print(f"  Std variance: {std_consistency:.4f}")
        print(f"  (Lower is better - indicates consistent generation)")

        return consistency_scores

    def evaluate_condition_discrimination(self, n_samples=500):
        """
        Test 2: Condition Discrimination
        Check if critic can distinguish between real and fake for specific conditions
        """
        print("\n" + "="*60)
        print("TEST 2: Condition Discrimination")
        print("="*60)

        # Group by conditions in reference data
        condition_groups = self.reference_adata.obs.groupby(
            ['annotation_major', 'tissue']
        ).size()

        discrimination_scores = {}
        conditions_tested = 0
        max_tests = 10

        for (annotation, tissue), count in condition_groups.items():
            if count < 10:  # Skip rare conditions
                continue

            if conditions_tested >= max_tests:
                break

            # Get real samples with these conditions
            mask = ((self.reference_adata.obs['annotation_major'] == annotation) &
                   (self.reference_adata.obs['tissue'] == tissue))
            real_samples = self.reference_adata[mask]

            sample_size = min(n_samples, len(real_samples))
            if sample_size < 10:
                continue

            if len(real_samples) > sample_size:
                indices = np.random.choice(len(real_samples), sample_size, replace=False)
                real_samples = real_samples[indices]

            # Extract expression
            if hasattr(real_samples.X, 'toarray'):
                real_expression = real_samples.X.toarray().astype(np.float32)
            else:
                real_expression = real_samples.X.astype(np.float32)

            # Encode conditions
            encoded_conditions = self.condition_processor.encode_conditions(real_samples.obs)

            # Generate fake samples with same conditions
            noise = tf.random.normal((len(real_samples), self.config['model']['latent_dim']))
            fake_expression = self.generator((noise, encoded_conditions), training=False)

            # Get critic scores
            real_scores = self.critic((real_expression, encoded_conditions), training=False).numpy()
            fake_scores = self.critic((fake_expression, encoded_conditions), training=False).numpy()

            # Calculate discrimination (higher difference = better)
            discrimination = np.mean(real_scores) - np.mean(fake_scores)
            discrimination_scores[f"{annotation}_{tissue}"] = discrimination

            print(f"[{conditions_tested+1}] {annotation:20s} | {tissue:15s}")
            print(f"    Real score: {np.mean(real_scores):7.4f} | Fake score: {np.mean(fake_scores):7.4f} | "
                  f"Discrimination: {discrimination:7.4f}")

            conditions_tested += 1

        mean_discrimination = np.mean(list(discrimination_scores.values()))
        print(f"\n✅ Overall Discrimination Score: {mean_discrimination:.4f}")
        print(f"  (Should be small/near zero for good generation)")

        return discrimination_scores

    def evaluate_condition_specificity(self, n_samples=100):
        """
        Test 3: Condition Specificity
        Generate samples with different conditions and check if they're different
        """
        print("\n" + "="*60)
        print("TEST 3: Condition Specificity")
        print("="*60)

        # Get common annotations and tissues
        top_annotations = self.reference_adata.obs['annotation_major'].value_counts().head(3).index.tolist()
        top_tissues = self.reference_adata.obs['tissue'].value_counts().head(3).index.tolist()

        results = {}

        # Test annotation_major differences (same tissue)
        if len(top_annotations) >= 2 and len(top_tissues) >= 1:
            print(f"\nTesting annotation_major difference (same tissue={top_tissues[0]}):")
            annotation1, annotation2 = top_annotations[0], top_annotations[1]

            # Generate samples for annotation1
            cond1_df = pd.DataFrame([{'annotation_major': annotation1, 'tissue': top_tissues[0]}] * n_samples)
            cond1_encoded = self.condition_processor.encode_conditions(cond1_df)
            noise1 = tf.random.normal((n_samples, self.config['model']['latent_dim']))
            samples1 = self.generator((noise1, cond1_encoded), training=False).numpy()

            # Generate samples for annotation2
            cond2_df = pd.DataFrame([{'annotation_major': annotation2, 'tissue': top_tissues[0]}] * n_samples)
            cond2_encoded = self.condition_processor.encode_conditions(cond2_df)
            noise2 = tf.random.normal((n_samples, self.config['model']['latent_dim']))
            samples2 = self.generator((noise2, cond2_encoded), training=False).numpy()

            # Statistical test
            t_stats, p_values = stats.ttest_ind(samples1, samples2, axis=0)
            n_significant = np.sum(p_values < 0.05)

            print(f"  {annotation1} vs {annotation2}")
            print(f"  Significantly different genes: {n_significant}/{len(p_values)} "
                  f"({n_significant/len(p_values)*100:.1f}%)")
            print(f"  Mean expression difference: {np.mean(np.abs(samples1.mean(axis=0) - samples2.mean(axis=0))):.4f}")

            results['annotation_diff_genes'] = n_significant

        # Test tissue differences (same annotation)
        if len(top_tissues) >= 2 and len(top_annotations) >= 1:
            print(f"\nTesting tissue difference (same annotation_major={top_annotations[0]}):")
            tissue1, tissue2 = top_tissues[0], top_tissues[1]

            # Generate samples for tissue1
            cond1_df = pd.DataFrame([{'annotation_major': top_annotations[0], 'tissue': tissue1}] * n_samples)
            cond1_encoded = self.condition_processor.encode_conditions(cond1_df)
            noise1 = tf.random.normal((n_samples, self.config['model']['latent_dim']))
            samples1 = self.generator((noise1, cond1_encoded), training=False).numpy()

            # Generate samples for tissue2
            cond2_df = pd.DataFrame([{'annotation_major': top_annotations[0], 'tissue': tissue2}] * n_samples)
            cond2_encoded = self.condition_processor.encode_conditions(cond2_df)
            noise2 = tf.random.normal((n_samples, self.config['model']['latent_dim']))
            samples2 = self.generator((noise2, cond2_encoded), training=False).numpy()

            # Statistical test
            t_stats, p_values = stats.ttest_ind(samples1, samples2, axis=0)
            n_significant = np.sum(p_values < 0.05)

            print(f"  {tissue1} vs {tissue2}")
            print(f"  Significantly different genes: {n_significant}/{len(p_values)} "
                  f"({n_significant/len(p_values)*100:.1f}%)")
            print(f"  Mean expression difference: {np.mean(np.abs(samples1.mean(axis=0) - samples2.mean(axis=0))):.4f}")

            results['tissue_diff_genes'] = n_significant

        results['total_genes'] = self.config['data']['gene_dim']

        return results

    def evaluate_condition_distribution(self, n_samples=1000):
        """
        Test 4: Condition Distribution Matching
        Check if generated samples match the expression distribution of real samples
        for each condition combination
        """
        print("\n" + "="*60)
        print("TEST 4: Condition Distribution Matching")
        print("="*60)

        # Get major condition combinations (with enough samples)
        condition_groups = self.reference_adata.obs.groupby(
            ['annotation_major', 'tissue']
        ).size()

        major_conditions = condition_groups[condition_groups > 50].head(5)  # Top 5 conditions

        distribution_scores = {}

        for (annotation, tissue), count in major_conditions.items():
            # Get real samples
            mask = ((self.reference_adata.obs['annotation_major'] == annotation) &
                   (self.reference_adata.obs['tissue'] == tissue))
            real_samples = self.reference_adata[mask]

            if hasattr(real_samples.X, 'toarray'):
                real_expression = real_samples.X.toarray().astype(np.float32)
            else:
                real_expression = real_samples.X.astype(np.float32)

            # Generate fake samples
            n_gen = min(n_samples, len(real_samples))
            condition_df = pd.DataFrame([{'annotation_major': annotation, 'tissue': tissue}] * n_gen)
            encoded_conditions = self.condition_processor.encode_conditions(condition_df)
            noise = tf.random.normal((n_gen, self.config['model']['latent_dim']))
            fake_expression = self.generator((noise, encoded_conditions), training=False).numpy()

            # Compare distributions using KS test
            ks_stats = []
            for gene_idx in range(min(100, real_expression.shape[1])):  # Test first 100 genes
                ks_stat, _ = stats.ks_2samp(
                    real_expression[:, gene_idx],
                    fake_expression[:, gene_idx]
                )
                ks_stats.append(ks_stat)

            mean_ks = np.mean(ks_stats)
            distribution_scores[f"{annotation}_{tissue}"] = mean_ks

            print(f"{annotation:20s} | {tissue:15s} | Mean KS: {mean_ks:.4f}")

        overall_score = np.mean(list(distribution_scores.values()))
        print(f"\n✅ Overall Distribution Matching Score: {overall_score:.4f}")
        print(f"  (Lower is better - indicates similar distributions)")

        return distribution_scores

    def generate_condition_report(self, output_dir=None):
        """
        Generate comprehensive condition learning report
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE CONDITION EVALUATION REPORT (Cancer Data)")
        print("="*80)

        # Run all tests
        consistency_scores = self.evaluate_condition_consistency()
        discrimination_scores = self.evaluate_condition_discrimination()
        specificity_results = self.evaluate_condition_specificity()
        distribution_scores = self.evaluate_condition_distribution()

        # Create summary
        report = {
            'checkpoint': self.checkpoint_dir,
            'timestamp': pd.Timestamp.now().isoformat(),
            'conditions': ['annotation_major', 'tissue'],
            'tests': {
                'consistency': {
                    'mean_variance': float(np.mean(consistency_scores)),
                    'std_variance': float(np.std(consistency_scores)),
                    'interpretation': 'Lower variance indicates better consistency'
                },
                'discrimination': {
                    'mean_score': float(np.mean(list(discrimination_scores.values()))),
                    'interpretation': 'Near zero indicates good generation quality'
                },
                'specificity': {
                    'annotation_responsive_genes': specificity_results.get('annotation_diff_genes', 0),
                    'tissue_responsive_genes': specificity_results.get('tissue_diff_genes', 0),
                    'total_genes': specificity_results['total_genes'],
                    'interpretation': 'Higher percentage indicates better condition specificity'
                },
                'distribution': {
                    'mean_ks_statistic': float(np.mean(list(distribution_scores.values()))),
                    'interpretation': 'Lower KS statistic indicates better distribution matching'
                }
            }
        }

        # Save report
        if output_dir is None:
            output_dir = os.path.dirname(self.checkpoint_dir)

        report_path = os.path.join(output_dir, 'condition_evaluation_report_cancer.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Report saved to: {report_path}")

        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        # Determine overall quality
        quality_score = 0
        quality_max = 0

        # Consistency check (variance < 1.0 is good)
        if report['tests']['consistency']['mean_variance'] < 1.0:
            quality_score += 1
            print("✅ Consistency: GOOD (low variance)")
        else:
            print("⚠️  Consistency: POOR (high variance)")
        quality_max += 1

        # Discrimination check (abs value < 2.0 is good)
        if abs(report['tests']['discrimination']['mean_score']) < 2.0:
            quality_score += 1
            print("✅ Discrimination: GOOD (realistic samples)")
        else:
            print("⚠️  Discrimination: POOR (easily distinguished)")
        quality_max += 1

        # Specificity check (>10% genes responsive is good)
        annotation_pct = report['tests']['specificity']['annotation_responsive_genes'] / report['tests']['specificity']['total_genes']
        tissue_pct = report['tests']['specificity']['tissue_responsive_genes'] / report['tests']['specificity']['total_genes']
        if annotation_pct > 0.1 and tissue_pct > 0.1:
            quality_score += 1
            print(f"✅ Specificity: GOOD (annotation: {annotation_pct*100:.1f}%, tissue: {tissue_pct*100:.1f}%)")
        else:
            print(f"⚠️  Specificity: POOR (annotation: {annotation_pct*100:.1f}%, tissue: {tissue_pct*100:.1f}%)")
        quality_max += 1

        # Distribution check (KS < 0.3 is good)
        if report['tests']['distribution']['mean_ks_statistic'] < 0.3:
            quality_score += 1
            print("✅ Distribution: GOOD (matches real data)")
        else:
            print("⚠️  Distribution: POOR (different from real)")
        quality_max += 1

        print(f"\nOverall Quality: {quality_score}/{quality_max}")

        if quality_score == quality_max:
            print("🎉 Excellent condition learning!")
        elif quality_score >= quality_max * 0.75:
            print("👍 Good condition learning with minor issues")
        elif quality_score >= quality_max * 0.5:
            print("⚠️  Moderate condition learning - may need more training")
        else:
            print("❌ Poor condition learning - consider retraining")

        return report


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate condition learning in trained model (Cancer Data)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python multi_cond/evaluate_conditions_cancer.py --checkpoint cancer/best_model --reference cancer_data.h5ad
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint directory (e.g., cancer/best_model)')
    parser.add_argument('--reference', type=str,
                        help='Path to reference h5ad file (optional, uses config if not provided)')
    parser.add_argument('--config', type=str,
                        help='Path to config file (optional)')
    parser.add_argument('--output', type=str,
                        help='Output directory for report')

    args = parser.parse_args()

    print("="*60)
    print("Multi-Conditional scGAN - Cancer Condition Evaluator")
    print("="*60)

    # Create evaluator
    evaluator = CancerConditionEvaluator(
        checkpoint_dir=args.checkpoint,
        reference_data_path=args.reference,
        config_path=args.config
    )

    # Generate comprehensive report
    report = evaluator.generate_condition_report(output_dir=args.output)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
