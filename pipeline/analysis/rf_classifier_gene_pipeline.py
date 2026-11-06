#!/usr/bin/env python3
"""
Random Forest Classifier Gene Pipeline for Phenotypically Predictive Gene Analysis

This script performs Random Forest classification between two groups in a specified column
(e.g., Flight: 0.0 vs 1.0 for spaceflight data, or tissue: metastasis normal vs metastasis tumor for cancer data)
in two separate AnnData objects, identifies the top N most predictive genes from each, and finds overlaps.

Default behavior preserves spaceflight analysis (Flight column, groups 0.0 vs 1.0).

Uses Repeated Stratified K-Fold Cross-Validation to:
- Obtain robust performance metrics (accuracy, precision, recall, F1)
- Average feature importances across multiple models for stable gene rankings
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# Set plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.linewidth': 1.2
})


class RFClassifierComparison:
    """Class to handle Random Forest classification comparison between two AnnData objects."""

    def __init__(self, adata1, adata2, name1="Dataset_1", name2="Dataset_2", output_dir="rf_results",
                 column_name="Flight", group1="0.0", group2="1.0",
                 group1_display=None, group2_display=None,
                 n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2,
                 n_folds=5, n_repeats=3, random_state=42, show_plots=False):
        """
        Initialize the RF classifier comparison with two AnnData objects.

        Parameters:
        -----------
        adata1, adata2 : AnnData
            AnnData objects to compare
        name1, name2 : str
            Custom names for the datasets (used in plots and outputs)
        output_dir : str
            Directory to save results
        column_name : str
            Name of the column in obs to use for grouping (default: "Flight" for spaceflight data)
        group1 : str
            First group value (will be encoded as 0)
        group2 : str
            Second group value (will be encoded as 1)
        group1_display : str
            Display name for group1 in plots (default: same as group1)
        group2_display : str
            Display name for group2 in plots (default: same as group2)
        n_estimators : int
            Number of trees in random forest
        max_depth : int
            Maximum depth of trees
        min_samples_split : int
            Minimum samples required to split a node
        min_samples_leaf : int
            Minimum samples required at leaf node
        n_folds : int
            Number of folds for cross-validation
        n_repeats : int
            Number of times to repeat cross-validation
        random_state : int
            Random seed for reproducibility
        show_plots : bool
            Whether to display plots with plt.show() (default: False)
        """
        self.adata1 = adata1.copy()
        self.adata2 = adata2.copy()
        self.name1 = name1
        self.name2 = name2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Store column and group information
        self.column_name = column_name
        self.group1 = str(group1)
        self.group2 = str(group2)
        self.group1_display = group1_display if group1_display else str(group1)
        self.group2_display = group2_display if group2_display else str(group2)
        self.show_plots = show_plots

        # Random Forest hyperparameters
        self.rf_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            'n_jobs': -1  # Use all available cores
        }

        # Cross-validation parameters
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.random_state = random_state

        # Validate inputs
        self._validate_data()

        # Results storage
        self.results = {}

    def _validate_data(self):
        """Validate that both AnnData objects have the required column with specified groups."""
        for i, adata in enumerate([self.adata1, self.adata2], 1):
            if self.column_name not in adata.obs.columns:
                raise ValueError(f"AnnData {i} does not have '{self.column_name}' column in obs")

            # Get unique values and handle different data types
            column_values = set(adata.obs[self.column_name].unique())

            # Convert values to strings for comparison, handling NaN values
            column_values_str = set()
            for val in column_values:
                if pd.isna(val):
                    continue  # Skip NaN values
                column_values_str.add(str(val))

            # Check if the specified groups exist in the data
            if self.group1 not in column_values_str:
                raise ValueError(f"AnnData {i} '{self.column_name}' column does not contain group '{self.group1}'. Found: {column_values_str}")
            if self.group2 not in column_values_str:
                raise ValueError(f"AnnData {i} '{self.column_name}' column does not contain group '{self.group2}'. Found: {column_values_str}")

            # Convert to string for consistency
            adata.obs[self.column_name] = adata.obs[self.column_name].astype(str)

        print("✓ Data validation passed")
        print(f"{self.name1}: {self.adata1.n_obs} cells, {self.adata1.n_vars} genes")
        print(f"{self.name2}: {self.adata2.n_obs} cells, {self.adata2.n_vars} genes")
        print(f"Classifying '{self.group2}' (1) vs '{self.group1}' (0) in column '{self.column_name}'")

    def _prepare_adata_for_classification(self, adata, name):
        """
        Prepare AnnData for RF classification by filtering to specified groups.

        Returns:
        --------
        X : np.ndarray
            Gene expression matrix (cells × genes)
        y : np.ndarray
            Binary labels (0 for group1, 1 for group2)
        gene_names : np.ndarray
            Gene names corresponding to columns in X
        """
        print(f"\nPreparing {name} for RF classification...")

        # Filter to only the two groups of interest
        mask = adata.obs[self.column_name].isin([self.group1, self.group2])
        adata_filtered = adata[mask].copy()

        # Print group distribution
        group_counts = adata_filtered.obs[self.column_name].value_counts()
        print(f"{name} {self.column_name} distribution after filtering:")
        for group, count in group_counts.items():
            print(f"  Group {group}: {count} cells")

        # Extract gene expression matrix
        if sparse.issparse(adata_filtered.X):
            X = adata_filtered.X.toarray()
        else:
            X = adata_filtered.X

        # Create binary labels (0 for group1, 1 for group2)
        y = (adata_filtered.obs[self.column_name] == self.group2).astype(int).values

        # Get gene names
        gene_names = adata_filtered.var_names.values

        print(f"Prepared data shape: {X.shape}")
        print(f"Class distribution: {self.group1}={np.sum(y == 0)}, {self.group2}={np.sum(y == 1)}")

        return X, y, gene_names

    def perform_rf_classification(self, adata, name, n_genes=50):
        """
        Perform Random Forest classification with repeated stratified k-fold cross-validation.

        Parameters:
        -----------
        adata : AnnData
            AnnData object to analyze
        name : str
            Name identifier for the dataset
        n_genes : int
            Number of top genes to return

        Returns:
        --------
        pd.DataFrame
            DataFrame with top N genes ranked by mean feature importance across CV folds
        """
        print(f"\n{'='*80}")
        print(f"Performing RF Classification for {name}")
        print(f"{'='*80}")

        # Prepare data
        X, y, gene_names = self._prepare_adata_for_classification(adata, name)

        # Setup repeated stratified k-fold cross-validation
        rskf = RepeatedStratifiedKFold(
            n_splits=self.n_folds,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )

        # Storage for results across all folds
        all_importances = []
        all_metrics = []

        print(f"\nRunning {self.n_folds}-fold CV with {self.n_repeats} repeats ({self.n_folds * self.n_repeats} total models)...")

        fold_num = 0
        for train_idx, test_idx in rskf.split(X, y):
            fold_num += 1

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train Random Forest
            rf = RandomForestClassifier(**self.rf_params)
            rf.fit(X_train, y_train)

            # Predict on test set
            y_pred = rf.predict(X_test)

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            all_metrics.append(metrics)

            # Extract feature importances
            importances = rf.feature_importances_
            all_importances.append(importances)

            if fold_num % 5 == 0 or fold_num == 1:
                print(f"  Fold {fold_num}/{self.n_folds * self.n_repeats}: "
                      f"Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")

        # Convert to numpy arrays
        all_importances = np.array(all_importances)  # Shape: (n_folds * n_repeats, n_genes)

        # Calculate mean and std of importances across all folds
        mean_importances = np.mean(all_importances, axis=0)
        std_importances = np.std(all_importances, axis=0)

        # Calculate mean and std of metrics across all folds
        metrics_df = pd.DataFrame(all_metrics)
        mean_metrics = metrics_df.mean()
        std_metrics = metrics_df.std()

        print(f"\n{'='*80}")
        print(f"Cross-Validation Results for {name}")
        print(f"{'='*80}")
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            print(f"{metric_name.capitalize():12s}: {mean_metrics[metric_name]:.4f} ± {std_metrics[metric_name]:.4f}")

        # Get top N genes by mean importance
        top_idx = np.argsort(mean_importances)[::-1][:n_genes]
        top_genes = gene_names[top_idx]
        top_mean_importance = mean_importances[top_idx]
        top_std_importance = std_importances[top_idx]

        # Create results DataFrame
        result_df = pd.DataFrame({
            'gene': top_genes,
            'mean_importance': top_mean_importance,
            'std_importance': top_std_importance,
            'rank': np.arange(1, n_genes + 1),
            'dataset': name
        })

        print(f"\nTop 10 Most Predictive Genes for {name}:")
        for idx, row in result_df.head(10).iterrows():
            print(f"  {row['rank']:2d}. {row['gene']:20s} - Importance: {row['mean_importance']:.6f} ± {row['std_importance']:.6f}")

        # Store metrics separately
        self.results[f'{name}_metrics'] = {
            'mean': mean_metrics.to_dict(),
            'std': std_metrics.to_dict(),
            'all_folds': metrics_df
        }

        return result_df

    def find_overlaps(self, rf_results1, rf_results2, top_n=50):
        """
        Find overlapping genes between two RF classification result sets.

        Parameters:
        -----------
        rf_results1, rf_results2 : pd.DataFrame
            RF results from the two datasets
        top_n : int
            Number of top genes to consider for overlap

        Returns:
        --------
        dict
            Dictionary containing overlap analysis results
        """
        print(f"\n{'='*80}")
        print(f"Analyzing overlaps between top {top_n} predictive genes...")
        print(f"{'='*80}")

        # Get top N genes from each dataset
        genes1 = set(rf_results1.head(top_n)['gene'])
        genes2 = set(rf_results2.head(top_n)['gene'])

        # Calculate overlaps
        overlap_genes = genes1.intersection(genes2)
        unique_to_1 = genes1 - genes2
        unique_to_2 = genes2 - genes1

        # Calculate statistics
        overlap_count = len(overlap_genes)
        jaccard_index = overlap_count / len(genes1.union(genes2)) if len(genes1.union(genes2)) > 0 else 0
        overlap_percentage_1 = (overlap_count / len(genes1)) * 100 if len(genes1) > 0 else 0
        overlap_percentage_2 = (overlap_count / len(genes2)) * 100 if len(genes2) > 0 else 0

        overlap_results = {
            'top_n': top_n,
            'dataset1_genes': len(genes1),
            'dataset2_genes': len(genes2),
            'overlap_count': overlap_count,
            'overlap_genes': sorted(list(overlap_genes)),
            'unique_to_dataset1': sorted(list(unique_to_1)),
            'unique_to_dataset2': sorted(list(unique_to_2)),
            'jaccard_index': jaccard_index,
            'overlap_percentage_dataset1': overlap_percentage_1,
            'overlap_percentage_dataset2': overlap_percentage_2
        }

        print(f"Overlap count: {overlap_count}/{top_n} genes")
        print(f"Jaccard Index: {jaccard_index:.4f}")
        print(f"Overlap percentage ({self.name1}): {overlap_percentage_1:.2f}%")
        print(f"Overlap percentage ({self.name2}): {overlap_percentage_2:.2f}%")

        return overlap_results

    def save_results(self, rf_results1, rf_results2, overlap_results):
        """Save all results to files."""
        print(f"\nSaving results to {self.output_dir}...")

        # Save RF results with mean importances
        rf_results1.to_csv(self.output_dir / f'rf_results_{self.name1}.csv', index=False)
        rf_results2.to_csv(self.output_dir / f'rf_results_{self.name2}.csv', index=False)

        # Save metrics
        for name in [self.name1, self.name2]:
            if f'{name}_metrics' in self.results:
                metrics_data = self.results[f'{name}_metrics']
                metrics_summary = pd.DataFrame({
                    'metric': ['accuracy', 'precision', 'recall', 'f1'],
                    'mean': [metrics_data['mean'][m] for m in ['accuracy', 'precision', 'recall', 'f1']],
                    'std': [metrics_data['std'][m] for m in ['accuracy', 'precision', 'recall', 'f1']]
                })
                metrics_summary.to_csv(self.output_dir / f'rf_metrics_{name}.csv', index=False)

        # Save overlap results
        overlap_df = pd.DataFrame([
            {'Metric': 'Top N genes analyzed', 'Value': overlap_results['top_n']},
            {'Metric': f'{self.name1} genes', 'Value': overlap_results['dataset1_genes']},
            {'Metric': f'{self.name2} genes', 'Value': overlap_results['dataset2_genes']},
            {'Metric': 'Overlapping genes', 'Value': overlap_results['overlap_count']},
            {'Metric': 'Jaccard Index', 'Value': f"{overlap_results['jaccard_index']:.4f}"},
            {'Metric': f'Overlap % ({self.name1})', 'Value': f"{overlap_results['overlap_percentage_dataset1']:.2f}%"},
            {'Metric': f'Overlap % ({self.name2})', 'Value': f"{overlap_results['overlap_percentage_dataset2']:.2f}%"}
        ])
        overlap_df.to_csv(self.output_dir / 'overlap_summary.csv', index=False)

        # Save gene lists
        pd.DataFrame({'Gene': overlap_results['overlap_genes']}).to_csv(
            self.output_dir / 'overlapping_genes.csv', index=False)
        pd.DataFrame({'Gene': overlap_results['unique_to_dataset1']}).to_csv(
            self.output_dir / f'unique_to_{self.name1}.csv', index=False)
        pd.DataFrame({'Gene': overlap_results['unique_to_dataset2']}).to_csv(
            self.output_dir / f'unique_to_{self.name2}.csv', index=False)

        print("✓ All results saved successfully")

    def print_summary(self, overlap_results):
        """Print a comprehensive summary of the analysis."""
        print("\n" + "="*80)
        print("RANDOM FOREST CLASSIFICATION COMPARISON SUMMARY")
        print("="*80)

        print(f"\n📊 ANALYSIS OVERVIEW:")
        print(f"   • Classification task: {self.group2} (1) vs {self.group1} (0)")
        print(f"   • Column used: {self.column_name}")
        print(f"   • Top {overlap_results['top_n']} predictive genes analyzed from each dataset")
        print(f"   • CV Strategy: {self.n_folds}-fold × {self.n_repeats} repeats = {self.n_folds * self.n_repeats} models")

        print(f"\n🔄 OVERLAP RESULTS:")
        print(f"   • Overlapping genes: {overlap_results['overlap_count']}")
        print(f"   • Jaccard Index: {overlap_results['jaccard_index']:.4f}")
        print(f"   • Overlap percentage ({self.name1}): {overlap_results['overlap_percentage_dataset1']:.2f}%")
        print(f"   • Overlap percentage ({self.name2}): {overlap_results['overlap_percentage_dataset2']:.2f}%")

        if overlap_results['overlap_count'] > 0:
            print(f"\n🧬 TOP 10 OVERLAPPING GENES:")
            for i, gene in enumerate(overlap_results['overlap_genes'][:10], 1):
                print(f"   {i:2d}. {gene}")
            if len(overlap_results['overlap_genes']) > 10:
                print(f"   ... and {len(overlap_results['overlap_genes']) - 10} more")

        print(f"\n📈 INTERPRETATION:")
        jaccard = overlap_results['jaccard_index']
        if jaccard > 0.5:
            interpretation = "High similarity"
        elif jaccard > 0.3:
            interpretation = "Moderate similarity"
        elif jaccard > 0.1:
            interpretation = "Low similarity"
        else:
            interpretation = "Very low similarity"

        print(f"   • {interpretation} in predictive gene signatures (Jaccard = {jaccard:.4f})")
        print(f"   • Unique to {self.name1}: {len(overlap_results['unique_to_dataset1'])} genes")
        print(f"   • Unique to {self.name2}: {len(overlap_results['unique_to_dataset2'])} genes")

        print("\n" + "="*80)

    def run_analysis(self, n_genes=50):
        """
        Run the complete RF classification comparison analysis.

        Parameters:
        -----------
        n_genes : int
            Number of top genes to analyze

        Returns:
        --------
        dict
            Dictionary containing all results
        """
        print("\n" + "="*80)
        print("Starting RF Classification Comparison Analysis...")
        print("="*80)
        print(f"Top N genes: {n_genes}")
        print(f"RF Parameters: n_estimators={self.rf_params['n_estimators']}, "
              f"max_depth={self.rf_params['max_depth']}")

        # Perform RF classification on both datasets
        rf_results1 = self.perform_rf_classification(self.adata1, self.name1, n_genes)
        rf_results2 = self.perform_rf_classification(self.adata2, self.name2, n_genes)

        # Find overlaps
        overlap_results = self.find_overlaps(rf_results1, rf_results2, n_genes)

        # Save results
        self.save_results(rf_results1, rf_results2, overlap_results)

        # Print summary
        self.print_summary(overlap_results)

        # Store results in the object
        self.results.update({
            'rf_results1': rf_results1,
            'rf_results2': rf_results2,
            'overlap_results': overlap_results
        })

        return self.results


def _run_single_rf_classification(adata, name, column_name, group1, group2, output_dir, n_genes, rf_params, cv_params):
    """
    Helper function to run RF classification on a single dataset.

    Parameters:
    -----------
    adata : AnnData
        Dataset to analyze
    name : str
        Dataset name
    column_name : str
        Column for grouping
    group1, group2 : str
        Groups to classify
    output_dir : Path
        Directory to save results
    n_genes : int
        Number of top genes to return
    rf_params : dict
        Random Forest hyperparameters
    cv_params : dict
        Cross-validation parameters

    Returns:
    --------
    tuple
        (gene_list, results_df, metrics_dict)
    """
    # Validate and prepare data
    if column_name not in adata.obs.columns:
        raise ValueError(f"'{column_name}' column not found in {name}")

    adata.obs[column_name] = adata.obs[column_name].astype(str)

    # Filter to specified groups
    mask = adata.obs[column_name].isin([str(group1), str(group2)])
    adata_filtered = adata[mask].copy()

    # Extract data
    if sparse.issparse(adata_filtered.X):
        X = adata_filtered.X.toarray()
    else:
        X = adata_filtered.X

    y = (adata_filtered.obs[column_name] == str(group2)).astype(int).values
    gene_names = adata_filtered.var_names.values

    print(f"\n{name}: {X.shape[0]} cells, {X.shape[1]} genes")
    print(f"  {group1}: {np.sum(y == 0)} cells, {group2}: {np.sum(y == 1)} cells")

    # Setup CV
    rskf = RepeatedStratifiedKFold(**cv_params)

    # Storage for results
    all_importances = []
    all_metrics = []

    print(f"  Running {cv_params['n_splits']}-fold CV × {cv_params['n_repeats']} repeats...")

    # Run cross-validation
    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        all_metrics.append(metrics)
        all_importances.append(rf.feature_importances_)

    # Calculate mean importances and metrics
    all_importances = np.array(all_importances)
    mean_importances = np.mean(all_importances, axis=0)
    std_importances = np.std(all_importances, axis=0)

    metrics_df = pd.DataFrame(all_metrics)
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()

    # Get top N genes
    top_idx = np.argsort(mean_importances)[::-1][:n_genes]
    top_genes = gene_names[top_idx]

    # Create results DataFrame
    results_df = pd.DataFrame({
        'gene': top_genes,
        'mean_importance': mean_importances[top_idx],
        'std_importance': std_importances[top_idx],
        'rank': np.arange(1, n_genes + 1),
        'dataset': name
    })

    # Save results
    output_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(output_dir / f'rf_results_{name}.csv', index=False)

    metrics_summary = pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1'],
        'mean': [mean_metrics[m] for m in ['accuracy', 'precision', 'recall', 'f1']],
        'std': [std_metrics[m] for m in ['accuracy', 'precision', 'recall', 'f1']]
    })
    metrics_summary.to_csv(output_dir / f'rf_metrics_{name}.csv', index=False)

    print(f"  Metrics: Acc={mean_metrics['accuracy']:.3f}±{std_metrics['accuracy']:.3f}, "
          f"F1={mean_metrics['f1']:.3f}±{std_metrics['f1']:.3f}")

    return set(top_genes), results_df, mean_metrics.to_dict()


def compare_real_vs_all_models(
    real_path,
    cvae_path,
    diffusion_path,
    gan_path,
    output_base_dir,
    column_name="Flight",
    group1="0.0",
    group2="1.0",
    n_genes=50,
    plot_title=None,
    **rf_kwargs
):
    """
    Compare real data against all three synthetic models (CVAE, Diffusion, GAN)
    and generate a combined overlap comparison visualization.

    OPTIMIZED: Runs RF classification on real data only ONCE, then compares against each synthetic model.

    Parameters:
    -----------
    real_path : str
        Path to real data h5ad file
    cvae_path : str
        Path to CVAE synthetic data h5ad file
    diffusion_path : str
        Path to Diffusion synthetic data h5ad file
    gan_path : str
        Path to GAN synthetic data h5ad file
    output_base_dir : str
        Base directory for saving all results
    column_name : str
        Column name for classification groups
    group1, group2 : str
        Group values to classify
    n_genes : int
        Number of top genes to analyze
    plot_title : str, optional
        Title for the combined bar chart. Auto-generated if None.
    **rf_kwargs : dict
        Additional keyword arguments (n_estimators, max_depth, etc.)

    Returns:
    --------
    dict
        Dictionary containing all results and gene lists
    """
    print("\n" + "="*80)
    print("COMPARING REAL DATA VS ALL SYNTHETIC MODELS (OPTIMIZED)")
    print("="*80)
    print(f"Real data: {real_path}")
    print(f"CVAE data: {cvae_path}")
    print(f"Diffusion data: {diffusion_path}")
    print(f"GAN data: {gan_path}")
    print(f"Output directory: {output_base_dir}")
    print("="*80)

    # Load all datasets
    print("\nLoading datasets...")
    adata_real = sc.read_h5ad(real_path)
    adata_cvae = sc.read_h5ad(cvae_path)
    adata_diffusion = sc.read_h5ad(diffusion_path)
    adata_gan = sc.read_h5ad(gan_path)
    print("All datasets loaded successfully")

    # Create output directories
    base_dir = Path(output_base_dir)
    base_dir.mkdir(exist_ok=True, parents=True)

    # Prepare RF and CV parameters
    rf_params = {
        'n_estimators': rf_kwargs.get('n_estimators', 100),
        'max_depth': rf_kwargs.get('max_depth', 10),
        'min_samples_split': rf_kwargs.get('min_samples_split', 5),
        'min_samples_leaf': rf_kwargs.get('min_samples_leaf', 2),
        'random_state': rf_kwargs.get('random_state', 42),
        'n_jobs': -1
    }

    cv_params = {
        'n_splits': rf_kwargs.get('n_folds', 5),
        'n_repeats': rf_kwargs.get('n_repeats', 3),
        'random_state': rf_kwargs.get('random_state', 42)
    }

    print(f"\nRF Parameters: n_estimators={rf_params['n_estimators']}, max_depth={rf_params['max_depth']}")
    print(f"CV Strategy: {cv_params['n_splits']}-fold × {cv_params['n_repeats']} repeats")

    # Run RF classification on each dataset ONCE
    print(f"\n{'='*80}")
    print("STEP 1: Running RF Classification on Each Dataset")
    print(f"{'='*80}")

    datasets = {
        'Real': (adata_real, base_dir / 'real_analysis'),
        'CVAE': (adata_cvae, base_dir / 'cvae_analysis'),
        'Diffusion': (adata_diffusion, base_dir / 'diffusion_analysis'),
        'GAN': (adata_gan, base_dir / 'gan_analysis')
    }

    gene_sets = {}
    results_dfs = {}
    metrics = {}

    for name, (adata, output_dir) in datasets.items():
        print(f"\nRunning RF classification for {name}...")
        genes, results_df, model_metrics = _run_single_rf_classification(
            adata, name, column_name, group1, group2, output_dir, n_genes, rf_params, cv_params
        )
        gene_sets[name] = genes
        results_dfs[name] = results_df
        metrics[name] = model_metrics

    # Calculate overlaps between real and each synthetic model
    print(f"\n{'='*80}")
    print("STEP 2: Calculating Overlaps with Real Data")
    print(f"{'='*80}")

    overlap_results = {}
    overlap_counts = {}

    for model_name in ['CVAE', 'Diffusion', 'GAN']:
        overlap_genes = gene_sets['Real'].intersection(gene_sets[model_name])
        unique_to_real = gene_sets['Real'] - gene_sets[model_name]
        unique_to_model = gene_sets[model_name] - gene_sets['Real']

        overlap_count = len(overlap_genes)
        jaccard_index = overlap_count / len(gene_sets['Real'].union(gene_sets[model_name]))
        overlap_pct_real = (overlap_count / n_genes) * 100
        overlap_pct_model = (overlap_count / n_genes) * 100

        overlap_results[model_name] = {
            'overlap_genes': sorted(list(overlap_genes)),
            'overlap_count': overlap_count,
            'unique_to_real': sorted(list(unique_to_real)),
            'unique_to_model': sorted(list(unique_to_model)),
            'jaccard_index': jaccard_index,
            'overlap_percentage_real': overlap_pct_real,
            'overlap_percentage_model': overlap_pct_model
        }
        overlap_counts[model_name] = overlap_count

        # Save comparison results
        comparison_dir = base_dir / f'{model_name.lower()}_comparison'
        comparison_dir.mkdir(exist_ok=True, parents=True)

        pd.DataFrame({'Gene': overlap_results[model_name]['overlap_genes']}).to_csv(
            comparison_dir / 'overlapping_genes.csv', index=False)
        pd.DataFrame({'Gene': overlap_results[model_name]['unique_to_real']}).to_csv(
            comparison_dir / 'unique_to_Real.csv', index=False)
        pd.DataFrame({'Gene': overlap_results[model_name]['unique_to_model']}).to_csv(
            comparison_dir / f'unique_to_{model_name}.csv', index=False)

        overlap_summary = pd.DataFrame([
            {'Metric': 'Overlap count', 'Value': overlap_count},
            {'Metric': 'Jaccard Index', 'Value': f"{jaccard_index:.4f}"},
            {'Metric': 'Overlap % (Real)', 'Value': f"{overlap_pct_real:.2f}%"},
            {'Metric': f'Overlap % ({model_name})', 'Value': f"{overlap_pct_model:.2f}%"}
        ])
        overlap_summary.to_csv(comparison_dir / 'overlap_summary.csv', index=False)

        print(f"\n{model_name}: {overlap_count}/{n_genes} genes overlap (Jaccard={jaccard_index:.3f})")

    # Generate combined bar chart
    print(f"\n{'='*80}")
    print("STEP 3: Generating Combined Overlap Chart")
    print(f"{'='*80}")

    colors = {
        'Diffusion': 'lightcoral',
        'CVAE': '#8E44AD',
        'GAN': '#E69F00'
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = ['CVAE', 'Diffusion', 'GAN']
    counts = [overlap_counts[name] for name in model_names]
    bar_colors = [colors[name] for name in model_names]

    bars = ax.bar(model_names, counts, color=bar_colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5, width=0.5)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + n_genes * 0.02,
               f'{count}', ha='center', va='bottom',
               fontweight='bold', fontsize=16)

    if plot_title is None:
        plot_title = f"Overlap of Phenotypically Predictive Genes:\n{group2} vs {group1} Classification"
    ax.set_title(plot_title, fontsize=18, fontweight='bold', pad=20)

    ax.set_xlabel('Synthetic Data Model', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Number of Overlapping Genes (out of {n_genes})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, n_genes)

    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#FAFAFA')

    fig_path = base_dir / 'all_models_overlap_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {fig_path}")
    plt.close()

    # Save combined summary
    summary_df = pd.DataFrame({
        'Model': model_names,
        'Overlap_Count': counts,
        'Overlap_Percentage': [(c / n_genes) * 100 for c in counts],
        'Jaccard_Index': [overlap_results[name]['jaccard_index'] for name in model_names]
    })
    summary_path = base_dir / 'all_models_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    # Print final summary
    print(f"\n{'='*80}")
    print("ALL MODELS COMPARISON COMPLETE")
    print(f"{'='*80}")
    print("\nOverlap Summary:")
    for model_name in model_names:
        count = overlap_counts[model_name]
        percentage = (count / n_genes) * 100
        print(f"  {model_name:10s}: {count:3d}/{n_genes} genes ({percentage:.1f}%)")
    print(f"\nAll results saved to: {output_base_dir}")
    print(f"{'='*80}\n")

    return {
        'gene_sets': gene_sets,
        'results_dfs': results_dfs,
        'metrics': metrics,
        'overlap_results': overlap_results,
        'overlap_counts': overlap_counts,
        'summary': summary_df
    }


if __name__ == "__main__":
    print("Random Forest Classifier Gene Pipeline")
    print("=" * 80)
    print("\nThis module provides:")
    print("1. RFClassifierComparison - Pairwise RF classification comparison")
    print("2. compare_real_vs_all_models() - Compare real vs 3 synthetic models")
    print("\nImport this module to use these functions in your analysis scripts.")
    print("=" * 80)
