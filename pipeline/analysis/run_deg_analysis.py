#!/usr/bin/env python3
"""
DEG Analysis Runner for Pipeline Integration

Runs differential expression analysis comparing real vs synthetic data.

Usage:
    python run_deg_analysis.py --real path/to/real.h5ad --synthetic path/to/synthetic.h5ad \
        --output-dir path/to/output --column-name tissue --group1 "metastasis normal" --group2 "metastasis tumor"
"""

import argparse
import sys
from pathlib import Path
import scanpy as sc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from diff_exp_gene_pipeline_v2 import main as run_deg_main


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run DEG analysis comparing real vs synthetic data'
    )

    parser.add_argument('--real', required=True,
                        help='Path to real data h5ad file')
    parser.add_argument('--synthetic', required=True,
                        help='Path to synthetic data h5ad file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--column-name', required=True,
                        help='Column name for grouping (e.g., tissue, Flight)')
    parser.add_argument('--group1', required=True,
                        help='Reference group value')
    parser.add_argument('--group2', required=True,
                        help='Target group value')
    parser.add_argument('--group1-display', default=None,
                        help='Display name for group1 in plots (default: same as group1)')
    parser.add_argument('--group2-display', default=None,
                        help='Display name for group2 in plots (default: same as group2)')
    parser.add_argument('--name1', default='Real',
                        help='Name for real dataset (default: Real)')
    parser.add_argument('--name2', default='Synthetic',
                        help='Name for synthetic dataset (default: Synthetic)')
    parser.add_argument('--method', default='wilcoxon',
                        choices=['wilcoxon', 't-test', 't-test_overestim_var', 'logreg'],
                        help='DEG method (default: wilcoxon)')
    parser.add_argument('--n-genes', type=int, default=50,
                        help='Number of top DEGs to analyze (default: 50)')
    parser.add_argument('--show-plots', action='store_true',
                        help='Display plots with plt.show() (default: False)')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("DEG Analysis Pipeline")
    print("=" * 60)

    # Use display names if provided, otherwise use raw values
    group1_display = args.group1_display if args.group1_display else args.group1
    group2_display = args.group2_display if args.group2_display else args.group2

    print(f"\nConfiguration:")
    print(f"  Real data: {args.real}")
    print(f"  Synthetic data: {args.synthetic}")
    print(f"  Output: {args.output_dir}")
    print(f"  Column: {args.column_name}")
    print(f"  Groups: {args.group1} vs {args.group2}")
    print(f"  Display: {group1_display} vs {group2_display}")
    print(f"  Method: {args.method}")
    print(f"  N genes: {args.n_genes}")

    # Load data
    print("\n" + "-" * 40)
    print("Loading data...")

    try:
        adata_real = sc.read_h5ad(args.real)
        print(f"[OK] Real: {adata_real.n_obs} cells x {adata_real.n_vars} genes")

        adata_synthetic = sc.read_h5ad(args.synthetic)
        print(f"[OK] Synthetic: {adata_synthetic.n_obs} cells x {adata_synthetic.n_vars} genes")

    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)

    # Run analysis
    print("\n" + "-" * 40)
    print("Running DEG analysis...")

    try:
        analyzer = run_deg_main(
            adata1=adata_real,
            adata2=adata_synthetic,
            name1=args.name1,
            name2=args.name2,
            output_dir=args.output_dir,
            column_name=args.column_name,
            group1=args.group1,
            group2=args.group2,
            group1_display=group1_display,
            group2_display=group2_display,
            method=args.method,
            n_genes=args.n_genes,
            show_plots=args.show_plots
        )

        print("\n" + "=" * 60)
        print("DEG ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {args.output_dir}")

        sys.exit(0)

    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
