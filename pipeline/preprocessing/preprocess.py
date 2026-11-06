#!/usr/bin/env python3
"""
Preprocessing script for scRNA-seq data.

Performs:
1. Gene filtering by minimum cell expression
2. Sparsity-based gene filtering
3. Highly variable gene selection (optional)
4. Normalization (optional)

Usage:
    python preprocess.py --input raw_data.h5ad --output processed_data.h5ad [options]
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.sparse import issparse


def filter_genes_by_min_cells(adata, min_cells: int = 3, verbose: bool = True):
    """
    Filter genes expressed in fewer than min_cells.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    min_cells : int
        Minimum number of cells where gene must be expressed
    verbose : bool
        Print filtering statistics

    Returns
    -------
    None (modifies adata in place)
    """
    import scanpy as sc

    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    n_genes_after = adata.n_vars

    if verbose:
        print(f"Gene filtering (min_cells={min_cells}):")
        print(f"  Before: {n_genes_before} genes")
        print(f"  After: {n_genes_after} genes")
        print(f"  Removed: {n_genes_before - n_genes_after} genes")


def analyze_gene_sparsity(adata, verbose: bool = True) -> np.ndarray:
    """
    Analyze and return sparsity (zero percentage) for each gene.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    verbose : bool
        Print sparsity summary

    Returns
    -------
    np.ndarray
        Zero percentage for each gene
    """
    X = adata.X
    n_cells = X.shape[0]

    if issparse(X):
        non_zero_counts = np.array((X > 0).sum(axis=0)).flatten()
    else:
        non_zero_counts = np.sum(X > 0, axis=0)

    zero_percentages = (n_cells - non_zero_counts) / n_cells

    if verbose:
        print("\nSparsity Summary:")
        print(f"  Mean sparsity: {zero_percentages.mean()*100:.1f}%")
        print(f"  Median sparsity: {np.median(zero_percentages)*100:.1f}%")
        print(f"  Genes with >90% zeros: {np.sum(zero_percentages > 0.9)}")
        print(f"  Genes with >95% zeros: {np.sum(zero_percentages > 0.95)}")
        print(f"  Genes with >99% zeros: {np.sum(zero_percentages > 0.99)}")
        print(f"  Genes with >99.7% zeros: {np.sum(zero_percentages > 0.997)}")

    return zero_percentages


def filter_genes_by_sparsity(adata, max_zero_percentage: float = 0.997, verbose: bool = True):
    """
    Filter genes with sparsity above threshold.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    max_zero_percentage : float
        Maximum allowed zero percentage (0-1). Genes above this are removed.
    verbose : bool
        Print filtering statistics

    Returns
    -------
    None (modifies adata in place)
    """
    n_genes_before = adata.n_vars

    X = adata.X
    n_cells = X.shape[0]

    if issparse(X):
        non_zero_counts = np.array((X > 0).sum(axis=0)).flatten()
    else:
        non_zero_counts = np.sum(X > 0, axis=0)

    zero_percentages = (n_cells - non_zero_counts) / n_cells
    genes_to_keep = zero_percentages <= max_zero_percentage

    adata._inplace_subset_var(genes_to_keep)

    n_genes_after = adata.n_vars

    if verbose:
        print(f"\nSparsity filtering (max_zero={max_zero_percentage*100:.1f}%):")
        print(f"  Before: {n_genes_before} genes")
        print(f"  After: {n_genes_after} genes")
        print(f"  Removed: {n_genes_before - n_genes_after} genes ({(n_genes_before - n_genes_after)/n_genes_before*100:.1f}%)")

        # Report new sparsity
        if issparse(adata.X):
            sparsity_after = 1 - (adata.X.nnz / (adata.n_obs * adata.n_vars))
        else:
            sparsity_after = np.sum(adata.X == 0) / (adata.n_obs * adata.n_vars)
        print(f"  Matrix sparsity after filtering: {sparsity_after*100:.1f}%")


def select_highly_variable_genes(adata, n_top_genes: int = 2000, verbose: bool = True):
    """
    Select highly variable genes while preserving raw counts.

    Uses normalized data to identify HVGs, but returns raw counts for those genes.

    Parameters
    ----------
    adata : AnnData
        Raw count data (will be modified in place)
    n_top_genes : int
        Number of highly variable genes to select
    verbose : bool
        Print HVG selection statistics

    Returns
    -------
    None (modifies adata in place)
    """
    import scanpy as sc

    if verbose:
        print(f"\nHVG Selection (n_top_genes={n_top_genes}):")
        print(f"  Starting genes: {adata.n_vars}")

    # Store raw counts
    adata.raw = adata

    # Create copy for normalization
    adata_norm = adata.copy()

    # Normalize for HVG selection
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)

    # Select HVGs
    sc.pp.highly_variable_genes(adata_norm, n_top_genes=n_top_genes)
    hvg_genes = adata_norm.var['highly_variable']
    selected_gene_names = adata_norm.var_names[hvg_genes]

    # Apply to original data
    raw_gene_mask = adata.var_names.isin(selected_gene_names)
    adata._inplace_subset_var(raw_gene_mask)

    # Mark as highly variable
    adata.var['highly_variable'] = True

    if verbose:
        print(f"  Selected: {adata.n_vars} highly variable genes")


def normalize_data(adata, target_sum: float = 1e4, verbose: bool = True):
    """
    Normalize data using standard scanpy pipeline (normalize_total + log1p).

    Parameters
    ----------
    adata : AnnData
        Raw count data (will be modified in place)
    target_sum : float
        Target sum for normalization
    verbose : bool
        Print normalization info

    Returns
    -------
    None (modifies adata in place)
    """
    import scanpy as sc

    if verbose:
        print(f"\nNormalization (normalize_total + log1p):")
        print(f"  Target sum: {target_sum}")

    # Store raw if not already stored
    if adata.raw is None:
        adata.raw = adata

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    if verbose:
        X = adata.X
        if issparse(X):
            X = X.toarray()
        print(f"  Value range after: [{X.min():.4f}, {X.max():.4f}]")


def preprocess(
    input_path: str,
    output_path: str,
    min_cells: int = 3,
    sparsity_threshold: float = 0.997,
    n_hvg: Optional[int] = 2000,
    normalize: bool = True,
    target_sum: float = 1e4,
    verbose: bool = True
) -> bool:
    """
    Main preprocessing function.

    Parameters
    ----------
    input_path : str
        Path to input h5ad file
    output_path : str
        Path to save processed h5ad file
    min_cells : int
        Minimum cells for gene filtering
    sparsity_threshold : float
        Maximum zero percentage for sparsity filtering
    n_hvg : int, optional
        Number of highly variable genes (None to skip HVG selection)
    normalize : bool
        Whether to normalize data (normalize_total + log1p)
    target_sum : float
        Target sum for normalization
    verbose : bool
        Print progress

    Returns
    -------
    bool
        True if successful
    """
    import scanpy as sc

    print("=" * 60)
    print("  scRNA-seq Data Preprocessing")
    print("=" * 60)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")

    # Load data
    print(f"\nLoading data...")
    adata = sc.read_h5ad(input_path)
    print(f"  Shape: {adata.n_obs} cells x {adata.n_vars} genes")

    # Step 1: Filter genes by min cells
    filter_genes_by_min_cells(adata, min_cells=min_cells, verbose=verbose)

    # Step 2: Analyze and filter by sparsity
    analyze_gene_sparsity(adata, verbose=verbose)
    filter_genes_by_sparsity(adata, max_zero_percentage=sparsity_threshold, verbose=verbose)

    # Step 3: HVG selection (optional)
    if n_hvg is not None and n_hvg > 0:
        select_highly_variable_genes(adata, n_top_genes=n_hvg, verbose=verbose)

    # Step 4: Normalization (optional) - normalize_total + log1p
    if normalize:
        normalize_data(adata, target_sum=target_sum, verbose=verbose)

    # Save
    print(f"\nSaving processed data...")
    print(f"  Final shape: {adata.n_obs} cells x {adata.n_vars} genes")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)

    print(f"  Saved to: {output_path}")
    print("\n" + "=" * 60)
    print("  Preprocessing Complete")
    print("=" * 60)

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess scRNA-seq data for synthetic data generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing with defaults
  python preprocess.py --input raw.h5ad --output processed.h5ad

  # Custom sparsity threshold
  python preprocess.py --input raw.h5ad --output processed.h5ad --sparsity-threshold 0.99

  # Skip HVG selection
  python preprocess.py --input raw.h5ad --output processed.h5ad --n-hvg 0

  # Skip normalization (keep raw counts)
  python preprocess.py --input raw.h5ad --output processed.h5ad --no-normalize
        """
    )

    parser.add_argument('--input', required=True,
                        help='Path to input h5ad file (raw data)')
    parser.add_argument('--output', required=True,
                        help='Path to save processed h5ad file')

    # Gene filtering
    parser.add_argument('--min-cells', type=int, default=3,
                        help='Minimum cells for gene filtering (default: 3)')
    parser.add_argument('--sparsity-threshold', type=float, default=0.997,
                        help='Max zero percentage for genes (default: 0.997)')

    # HVG selection
    parser.add_argument('--n-hvg', type=int, default=2000,
                        help='Number of highly variable genes (0 to skip, default: 2000)')

    # Normalization
    parser.add_argument('--no-normalize', action='store_true',
                        help='Skip normalization (keep raw counts)')
    parser.add_argument('--target-sum', type=float, default=1e4,
                        help='Target sum for normalization (default: 10000)')

    # Output options
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    success = preprocess(
        input_path=args.input,
        output_path=args.output,
        min_cells=args.min_cells,
        sparsity_threshold=args.sparsity_threshold,
        n_hvg=args.n_hvg if args.n_hvg > 0 else None,
        normalize=not args.no_normalize,
        target_sum=args.target_sum,
        verbose=not args.quiet
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
