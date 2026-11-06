#!/usr/bin/env python3
"""
Combined RF Classifier Analysis Runner for Pipeline Integration

Runs RF classification comparing real data against all synthetic models (CVAE, Diffusion, GAN)
and generates a combined overlap bar chart.

Usage:
    python run_combined_classifier_analysis.py --real path/to/real.h5ad \
        --cvae path/to/cvae.h5ad --diffusion path/to/diffusion.h5ad --gan path/to/gan.h5ad \
        --output-dir path/to/output --column-name tissue --group1 "normal" --group2 "tumor"
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rf_classifier_gene_pipeline import compare_real_vs_all_models


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run combined RF classifier analysis for all models'
    )

    parser.add_argument('--real', required=True,
                        help='Path to real data h5ad file')
    parser.add_argument('--cvae', required=True,
                        help='Path to CVAE synthetic data h5ad file')
    parser.add_argument('--diffusion', required=True,
                        help='Path to Diffusion synthetic data h5ad file')
    parser.add_argument('--gan', required=True,
                        help='Path to GAN synthetic data h5ad file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--column-name', required=True,
                        help='Column name for grouping (e.g., tissue, Flight)')
    parser.add_argument('--group1', required=True,
                        help='First group value')
    parser.add_argument('--group2', required=True,
                        help='Second group value')
    parser.add_argument('--n-genes', type=int, default=50,
                        help='Number of top genes to analyze (default: 50)')
    parser.add_argument('--plot-title', default=None,
                        help='Custom title for the bar chart')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of RF estimators (default: 100)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--n-repeats', type=int, default=3,
                        help='Number of CV repeats (default: 3)')
    parser.add_argument('--core-output-dir', default=None,
                        help='Directory for core plots (defaults to output-dir)')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Combined RF Classifier Analysis Pipeline")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Real data: {args.real}")
    print(f"  CVAE data: {args.cvae}")
    print(f"  Diffusion data: {args.diffusion}")
    print(f"  GAN data: {args.gan}")
    print(f"  Output: {args.output_dir}")
    print(f"  Column: {args.column_name}")
    print(f"  Groups: {args.group1} vs {args.group2}")
    print(f"  N genes: {args.n_genes}")

    # Verify all files exist
    for name, path in [('Real', args.real), ('CVAE', args.cvae),
                        ('Diffusion', args.diffusion), ('GAN', args.gan)]:
        if not Path(path).exists():
            print(f"[ERROR] {name} data file not found: {path}")
            sys.exit(1)

    try:
        results = compare_real_vs_all_models(
            real_path=args.real,
            cvae_path=args.cvae,
            diffusion_path=args.diffusion,
            gan_path=args.gan,
            output_base_dir=args.output_dir,
            column_name=args.column_name,
            group1=args.group1,
            group2=args.group2,
            n_genes=args.n_genes,
            plot_title=args.plot_title,
            n_estimators=args.n_estimators,
            n_folds=args.n_folds,
            n_repeats=args.n_repeats
        )

        # Copy core plot to main analysis directory if specified
        if args.core_output_dir:
            import shutil
            core_dir = Path(args.core_output_dir)
            core_dir.mkdir(parents=True, exist_ok=True)
            source_plot = Path(args.output_dir) / "all_models_overlap_comparison.png"
            if source_plot.exists():
                core_plot = core_dir / "classifier_overlap_comparison.png"
                shutil.copy2(source_plot, core_plot)
                print(f"Core plot: {core_plot}")

        print("\n" + "=" * 60)
        print("COMBINED RF CLASSIFIER ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {args.output_dir}")
        print(f"Combined plot: {args.output_dir}/all_models_overlap_comparison.png")
        print(f"Summary: {args.output_dir}/all_models_summary.csv")

        sys.exit(0)

    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
