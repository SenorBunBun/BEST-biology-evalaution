#!/usr/bin/env python3
"""
Compute RF Classification for a Single Dataset

Runs Random Forest classification with Repeated Stratified K-Fold CV on one
h5ad file and saves the top N gene importances and metrics. Used by the driver
to compute real and synthetic classifications independently before comparison.

Usage:
    python compute_classifier.py --data path/to/data.h5ad --output-dir path/to/output \
        --name Real --column-name Flight --group1 0.0 --group2 1.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
)
import warnings

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute RF classification for a single dataset'
    )

    parser.add_argument('--data', required=True,
                        help='Path to h5ad file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for rf_results.csv and rf_metrics.csv')
    parser.add_argument('--name', default='Dataset',
                        help='Name identifier for the dataset (default: Dataset)')
    parser.add_argument('--column-name', required=True,
                        help='Column name for grouping (e.g., tissue, Flight)')
    parser.add_argument('--group1', required=True,
                        help='First group value (encoded as 0)')
    parser.add_argument('--group2', required=True,
                        help='Second group value (encoded as 1)')
    parser.add_argument('--n-genes', type=int, default=50,
                        help='Number of top genes to return (default: 50)')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of RF estimators (default: 100)')
    parser.add_argument('--max-depth', type=int, default=10,
                        help='Max tree depth; use <=0 for None/unbounded (default: 10)')
    parser.add_argument('--min-samples-split', type=int, default=5,
                        help='Min samples to split an internal node (default: 5)')
    parser.add_argument('--min-samples-leaf', type=int, default=2,
                        help='Min samples at a leaf node (default: 2)')
    parser.add_argument('--class-weight', default=None,
                        help="RF class_weight: 'balanced', 'balanced_subsample', or "
                             "'none'/unset for no weighting (default: none)")
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--n-repeats', type=int, default=3,
                        help='Number of CV repeats (default: 3)')

    return parser.parse_args()


def parse_class_weight(value):
    """Normalize a class_weight CLI/config value to what sklearn expects.

    '', 'none', 'null' (any case) -> None; otherwise the string is passed
    through (e.g. 'balanced', 'balanced_subsample').
    """
    if value is None:
        return None
    v = str(value).strip()
    if v.lower() in ('', 'none', 'null'):
        return None
    return v


def compute_rf_classification(adata, column_name, group1, group2, n_genes=50,
                               n_estimators=100, n_folds=5, n_repeats=3,
                               max_depth=10, min_samples_split=5, min_samples_leaf=2,
                               class_weight=None, random_state=42):
    """
    Run RF classification with repeated stratified k-fold CV on a single dataset.

    Reuses the core logic from RFClassifierComparison.perform_rf_classification().

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    column_name : str
        obs column used for grouping.
    group1 : str
        First group value (encoded as 0).
    group2 : str
        Second group value (encoded as 1).
    n_genes : int
        Number of top genes to return.
    n_estimators : int
        Number of trees in the forest.
    n_folds : int
        Number of CV folds.
    n_repeats : int
        Number of CV repeats.
    random_state : int
        Random seed.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (rf_results with top genes + importances, rf_metrics with accuracy/precision/recall/F1)
    """
    # Validate column exists
    if column_name not in adata.obs.columns:
        raise ValueError(f"Column '{column_name}' not found in obs. Available: {list(adata.obs.columns)}")

    # Convert column to string for consistent comparison
    adata.obs[column_name] = adata.obs[column_name].astype(str)
    group1_str = str(group1)
    group2_str = str(group2)

    # Validate groups exist
    unique_vals = set(adata.obs[column_name].unique())
    for g in [group1_str, group2_str]:
        if g not in unique_vals:
            raise ValueError(f"Group '{g}' not found in column '{column_name}'. Found: {unique_vals}")

    # Filter to specified groups
    mask = adata.obs[column_name].isin([group1_str, group2_str])
    adata_filtered = adata[mask].copy()

    # Print group distribution
    group_counts = adata_filtered.obs[column_name].value_counts()
    print(f"  Group distribution after filtering:")
    for group, count in group_counts.items():
        print(f"    {group}: {count} cells")

    # Extract data matrix
    if sparse.issparse(adata_filtered.X):
        X = adata_filtered.X.toarray()
    else:
        X = np.asarray(adata_filtered.X)

    # Binary labels: group1=0, group2=1
    y = (adata_filtered.obs[column_name] == group2_str).astype(int).values
    gene_names = adata_filtered.var_names.values

    print(f"  Data shape: {X.shape}")
    print(f"  Class distribution: {group1_str}={np.sum(y == 0)}, {group2_str}={np.sum(y == 1)}")

    # RF parameters (all configurable; <=0 max_depth means unbounded)
    class_weight = parse_class_weight(class_weight)
    rf_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth if (max_depth is not None and max_depth > 0) else None,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'class_weight': class_weight,
        'random_state': random_state,
        'n_jobs': -1
    }
    print(f"  RF params: n_estimators={rf_params['n_estimators']}, "
          f"max_depth={rf_params['max_depth']}, "
          f"min_samples_split={rf_params['min_samples_split']}, "
          f"min_samples_leaf={rf_params['min_samples_leaf']}, "
          f"class_weight={rf_params['class_weight']}")

    # Cross-validation
    rskf = RepeatedStratifiedKFold(
        n_splits=n_folds,
        n_repeats=n_repeats,
        random_state=random_state
    )

    all_importances = []
    all_metrics = []
    total_folds = n_folds * n_repeats

    print(f"  Running {n_folds}-fold CV x {n_repeats} repeats ({total_folds} total models)...")

    fold_num = 0
    for train_idx, test_idx in rskf.split(X, y):
        fold_num += 1

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        # Probability of the positive class (group2=1) for threshold-free metrics
        y_proba = rf.predict_proba(X_test)[:, 1]

        # ROC-AUC / PR-AUC are undefined if a test fold has a single class;
        # stratified CV avoids this, but guard anyway.
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
        else:
            roc_auc = np.nan
            pr_auc = np.nan

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
        }
        all_metrics.append(metrics)
        all_importances.append(rf.feature_importances_)

        if fold_num % 5 == 0 or fold_num == 1:
            print(f"    Fold {fold_num}/{total_folds}: "
                  f"Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")

    # Aggregate results
    all_importances = np.array(all_importances)
    mean_importances = np.mean(all_importances, axis=0)
    std_importances = np.std(all_importances, axis=0)

    metric_order = ['accuracy', 'precision', 'recall', 'f1',
                    'f1_macro', 'balanced_accuracy', 'roc_auc', 'pr_auc']

    metrics_df = pd.DataFrame(all_metrics)
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()

    print(f"\n  CV Results:")
    for metric_name in metric_order:
        print(f"    {metric_name.replace('_', ' ').capitalize():18s}: "
              f"{mean_metrics[metric_name]:.4f} +/- {std_metrics[metric_name]:.4f}")

    # Top N genes by mean importance
    top_idx = np.argsort(mean_importances)[::-1][:n_genes]

    results_df = pd.DataFrame({
        'gene': gene_names[top_idx],
        'mean_importance': mean_importances[top_idx],
        'std_importance': std_importances[top_idx],
        'rank': np.arange(1, n_genes + 1)
    })

    rf_metrics = pd.DataFrame({
        'metric': metric_order,
        'mean': [mean_metrics[m] for m in metric_order],
        'std': [std_metrics[m] for m in metric_order]
    })

    print(f"\n  Top 5 most predictive genes:")
    for _, row in results_df.head(5).iterrows():
        print(f"    {int(row['rank']):2d}. {row['gene']:20s} importance={row['mean_importance']:.6f}")

    return results_df, rf_metrics


def main():
    args = parse_args()

    print("=" * 60)
    print(f"COMPUTING RF CLASSIFICATION: {args.name}")
    print("=" * 60)
    print(f"  Data: {args.data}")
    print(f"  Column: {args.column_name}")
    print(f"  Groups: {args.group2} (1) vs {args.group1} (0)")
    print(f"  N genes: {args.n_genes}")
    print(f"  N estimators: {args.n_estimators}")
    print(f"  Max depth: {args.max_depth}  Min split: {args.min_samples_split}  "
          f"Min leaf: {args.min_samples_leaf}")
    print(f"  Class weight: {parse_class_weight(args.class_weight)}")
    print(f"  CV: {args.n_folds}-fold x {args.n_repeats} repeats")

    # Load data
    print("\nLoading data...")
    try:
        adata = sc.read_h5ad(args.data)
        print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)

    # Compute RF classification
    print("\nRunning RF classification...")
    try:
        results_df, metrics_df = compute_rf_classification(
            adata,
            column_name=args.column_name,
            group1=args.group1,
            group2=args.group2,
            n_genes=args.n_genes,
            n_estimators=args.n_estimators,
            n_folds=args.n_folds,
            n_repeats=args.n_repeats,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=args.class_weight,
            random_state=args.random_state
        )
    except Exception as e:
        print(f"[ERROR] RF classification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "rf_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[OK] Saved {len(results_df)} gene importances to: {results_path}")

    metrics_path = output_dir / "rf_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[OK] Saved metrics to: {metrics_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
