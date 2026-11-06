#!/usr/bin/env python3
"""
Compute DEGs for a Single Dataset

Runs differential gene expression analysis on one h5ad file and saves
the top N DEG results to a CSV. Used by the driver to compute real and
synthetic DEGs independently before comparison.

Usage:
    python compute_deg.py --data path/to/data.h5ad --output-dir path/to/output \
        --name Real --column-name Flight --group1 0.0 --group2 1.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import warnings

warnings.filterwarnings('ignore')

sc.settings.verbosity = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute DEGs for a single dataset'
    )

    parser.add_argument('--data', required=True,
                        help='Path to h5ad file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for deg_results.csv')
    parser.add_argument('--name', default='Dataset',
                        help='Name identifier for the dataset (default: Dataset)')
    parser.add_argument('--column-name', required=True,
                        help='Column name for grouping (e.g., tissue, Flight)')
    parser.add_argument('--group1', required=True,
                        help='Reference group value')
    parser.add_argument('--group2', required=True,
                        help='Target group value')
    parser.add_argument('--method', default='wilcoxon',
                        choices=['wilcoxon', 't-test', 't-test_overestim_var', 'logreg'],
                        help='DEG method (default: wilcoxon)')
    parser.add_argument('--n-genes', type=int, default=50,
                        help='Number of top DEGs to return (default: 50)')

    return parser.parse_args()


def compute_deg(adata, column_name, group1, group2, method='wilcoxon', n_genes=50):
    """
    Run DEG analysis on a single AnnData object and return top N results.

    Reuses the core logic from DEGCrossComparison.perform_deg_analysis().

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    column_name : str
        obs column used for grouping.
    group1 : str
        Reference group value.
    group2 : str
        Target group value.
    method : str
        Statistical test method.
    n_genes : int
        Number of top genes to return.

    Returns
    -------
    pd.DataFrame
        Top N DEG results sorted by adjusted p-value then absolute log-fold-change.
    """
    # Validate column exists
    if column_name not in adata.obs.columns:
        raise ValueError(f"Column '{column_name}' not found in obs. Available: {list(adata.obs.columns)}")

    # Convert column to string for consistent comparison
    adata.obs[column_name] = adata.obs[column_name].astype(str)

    # Validate groups exist
    unique_vals = set(adata.obs[column_name].unique())
    for g in [str(group1), str(group2)]:
        if g not in unique_vals:
            raise ValueError(f"Group '{g}' not found in column '{column_name}'. Found: {unique_vals}")

    # Print group distribution
    group_counts = adata.obs[column_name].value_counts()
    print(f"  Group distribution in '{column_name}':")
    for group, count in group_counts.items():
        print(f"    {group}: {count} cells")

    # Ensure sparse matrix for scanpy
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)

    # Perform DEG analysis
    reference_group = str(group1)
    target_group = str(group2)
    print(f"  Comparing '{target_group}' vs reference '{reference_group}' using {method}...")

    sc.tl.rank_genes_groups(
        adata,
        column_name,
        method=method,
        reference=reference_group,
        use_raw=False
    )

    # Extract results for the target group
    result_df = sc.get.rank_genes_groups_df(adata, group=target_group)

    # Add absolute log fold change
    result_df['abs_logfoldchanges'] = np.abs(result_df['logfoldchanges'])

    # Sort by adjusted p-value and absolute log fold change
    result_df = result_df.sort_values(
        ['pvals_adj', 'abs_logfoldchanges'],
        ascending=[True, False]
    )

    top_results = result_df.head(n_genes)

    print(f"  Found {len(result_df)} total DEGs, returning top {len(top_results)}")
    if len(top_results) > 0:
        print(f"  Top gene: {top_results.iloc[0]['names']} "
              f"(adj p-val: {top_results.iloc[0]['pvals_adj']:.2e})")

    return top_results


def main():
    args = parse_args()

    print("=" * 60)
    print(f"COMPUTING DEGs: {args.name}")
    print("=" * 60)
    print(f"  Data: {args.data}")
    print(f"  Column: {args.column_name}")
    print(f"  Groups: {args.group2} vs {args.group1} (reference)")
    print(f"  Method: {args.method}")
    print(f"  N genes: {args.n_genes}")

    # Load data
    print("\nLoading data...")
    try:
        adata = sc.read_h5ad(args.data)
        print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)

    # Compute DEGs
    print("\nRunning DEG analysis...")
    try:
        result_df = compute_deg(
            adata,
            column_name=args.column_name,
            group1=args.group1,
            group2=args.group2,
            method=args.method,
            n_genes=args.n_genes
        )
    except Exception as e:
        print(f"[ERROR] DEG analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "deg_results.csv"
    result_df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved {len(result_df)} DEGs to: {output_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
