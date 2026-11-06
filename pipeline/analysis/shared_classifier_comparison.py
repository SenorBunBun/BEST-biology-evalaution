#!/usr/bin/env python3
"""
Shared Classifier (RF) Comparison
Publication-ready bar chart comparing all models (CVAE, Diffusion, GAN)

Computes overlaps from per-dataset rf_results.csv files (produced by
compute_classifier.py), saving per-model comparison outputs and a combined
bar chart.

Usage:
    python shared_classifier_comparison.py --clf-dir path/to/classifier \
        --real-results path/to/classifier/real/rf_results.csv \
        --output-dir path/to/classifier \
        --dataset-name "OSDR-402" --group1-display "SpaceFlight" --group2-display "Ground Control" \
        --n-genes 50
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# PLOTTING PARAMETERS FOR PRESENTATION QUALITY
# =============================================================================
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'Arial',
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 26,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.linewidth': 1.5,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True
})

# Color scheme matching the pipeline visualization config
DEFAULT_COLORS = {
    'cvae': '#8E44AD',      # Purple
    'diffusion': '#E57373', # Light coral/red
    'gan': '#E69F00',       # Orange
}

MODEL_ORDER = ['cvae', 'diffusion', 'gan']
MODEL_DISPLAY_NAMES = {
    'cvae': 'CVAE',
    'diffusion': 'Diffusion',
    'gan': 'GAN'
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create combined RF classifier comparison plot for all models'
    )

    parser.add_argument('--clf-dir', required=True,
                        help='Path to classifier results directory (containing cvae/, diffusion/, gan/ subdirs)')
    parser.add_argument('--real-results', required=True,
                        help='Path to real rf_results.csv')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for the comparison plot')
    parser.add_argument('--core-output-dir', default=None,
                        help='Directory for core plots (defaults to output-dir)')
    parser.add_argument('--dataset-name', default='Dataset',
                        help='Dataset name for plot title (default: Dataset)')
    parser.add_argument('--group1-display', default='Group 1',
                        help='Display name for group 1 (default: Group 1)')
    parser.add_argument('--group2-display', default='Group 2',
                        help='Display name for group 2 (default: Group 2)')
    parser.add_argument('--n-genes', type=int, default=50,
                        help='Number of top genes analyzed (default: 50)')
    parser.add_argument('--colors', default=None,
                        help='JSON string with color overrides, e.g., \'{"cvae": "#8E44AD"}\'')
    parser.add_argument('--show-plots', action='store_true',
                        help='Display plots with plt.show() (default: False)')

    return parser.parse_args()


# =============================================================================
# OVERLAP COMPUTATION
# =============================================================================

def compute_overlaps_from_csvs(real_results_path, clf_dir, n_genes):
    """
    Load real and per-model rf_results.csv files, compute set overlaps.

    Args:
        real_results_path: Path to real/rf_results.csv
        clf_dir: Path to classifier directory containing model subdirectories
        n_genes: Number of top genes to consider

    Returns:
        Dictionary mapping model name to shared gene count.
        Also saves per-model comparison files.
    """
    real_path = Path(real_results_path)
    clf_dir = Path(clf_dir)

    if not real_path.exists():
        print(f"[ERROR] Real RF results not found: {real_path}")
        return {}

    real_df = pd.read_csv(real_path)
    real_genes = set(real_df['gene'].head(n_genes).tolist())
    print(f"[OK] Real: {len(real_genes)} top genes loaded from {real_path.name}")

    shared_counts = {}

    for model in MODEL_ORDER:
        model_dir = clf_dir / model
        model_results_path = model_dir / "rf_results.csv"

        if not model_results_path.exists():
            print(f"[WARN] RF results not found for {model}: {model_results_path}")
            continue

        model_df = pd.read_csv(model_results_path)
        model_genes = set(model_df['gene'].head(n_genes).tolist())

        # Compute overlap
        overlap_genes = sorted(real_genes.intersection(model_genes))
        unique_to_real = sorted(real_genes - model_genes)
        unique_to_model = sorted(model_genes - real_genes)

        overlap_count = len(overlap_genes)
        union_size = len(real_genes.union(model_genes))
        jaccard_index = overlap_count / union_size if union_size > 0 else 0
        overlap_pct_real = (overlap_count / len(real_genes)) * 100 if real_genes else 0
        overlap_pct_model = (overlap_count / len(model_genes)) * 100 if model_genes else 0

        shared_counts[model] = overlap_count

        # Save per-model comparison outputs
        model_dir.mkdir(parents=True, exist_ok=True)

        # overlap_summary.csv (same format as RFClassifierComparison.save_results)
        overlap_summary = pd.DataFrame([
            {'Metric': 'Top N genes analyzed', 'Value': n_genes},
            {'Metric': 'Real genes', 'Value': len(real_genes)},
            {'Metric': f'Synthetic_{model.upper()} genes', 'Value': len(model_genes)},
            {'Metric': 'Overlapping genes', 'Value': overlap_count},
            {'Metric': 'Jaccard Index', 'Value': f"{jaccard_index:.4f}"},
            {'Metric': 'Overlap % (Real)', 'Value': f"{overlap_pct_real:.2f}%"},
            {'Metric': f'Overlap % (Synthetic_{model.upper()})', 'Value': f"{overlap_pct_model:.2f}%"}
        ])
        overlap_summary.to_csv(model_dir / 'overlap_summary.csv', index=False)

        # Gene lists
        pd.DataFrame({'Gene': overlap_genes}).to_csv(
            model_dir / 'overlapping_genes.csv', index=False)
        pd.DataFrame({'Gene': unique_to_real}).to_csv(
            model_dir / 'unique_to_Real.csv', index=False)
        pd.DataFrame({'Gene': unique_to_model}).to_csv(
            model_dir / f'unique_to_Synthetic_{model.upper()}.csv', index=False)

        # Print per-model summary
        print(f"[OK] {MODEL_DISPLAY_NAMES[model]}: {overlap_count} shared genes "
              f"(Jaccard={jaccard_index:.4f})")

        # Print metrics if available
        metrics_path = model_dir / "rf_metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            acc_row = metrics_df[metrics_df['metric'] == 'accuracy']
            f1_row = metrics_df[metrics_df['metric'] == 'f1']
            if not acc_row.empty and not f1_row.empty:
                print(f"     Metrics: Acc={acc_row['mean'].values[0]:.3f}, "
                      f"F1={f1_row['mean'].values[0]:.3f}")

    return shared_counts


# =============================================================================
# PLOTTING
# =============================================================================

def create_classifier_comparison_plot(shared_counts, output_dir, dataset_name,
                                       group1_display, group2_display,
                                       n_genes=50, colors=None,
                                       show_plots=False,
                                       core_output_dir=None):
    """
    Create publication-ready shared classifier comparison plot.

    Returns:
        Path to the saved plot file
    """
    if not shared_counts:
        print("[ERROR] No shared gene counts available to plot")
        return None

    # Use default colors with optional overrides
    plot_colors = DEFAULT_COLORS.copy()
    if colors:
        plot_colors.update(colors)

    # Prepare data in model order
    models = []
    gene_counts = []
    bar_colors = []

    for model in MODEL_ORDER:
        if model in shared_counts:
            models.append(MODEL_DISPLAY_NAMES[model])
            gene_counts.append(shared_counts[model])
            bar_colors.append(plot_colors.get(model, '#808080'))

    if not models:
        print("[ERROR] No valid model data to plot")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 9))

    # Create bar plot
    bars = ax.bar(models, gene_counts, color=bar_colors, alpha=0.9,
                  edgecolor='black', linewidth=2, width=0.5)

    # Customize plot - presentation quality styling
    ax.set_title(
        f'Overlap of Phenotypically Predictive Genes\n'
        f'Real vs Synthetic Data for {dataset_name}',
        fontsize=24, fontweight='bold', pad=25
    )
    ax.set_xlabel('Synthetic Data Model', fontsize=22, fontweight='bold')
    ax.set_ylabel(f'Number of Overlapping Genes (out of {n_genes})', fontsize=22, fontweight='bold')
    ax.set_ylim(0, n_genes)

    # Add grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Background color
    ax.set_facecolor('#FAFAFA')

    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='x', which='major', pad=12, labelsize=20)

    # Add value labels on top of bars
    for bar, value in zip(bars, gene_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + n_genes * 0.02,
                f'{value}', ha='center', va='bottom',
                fontweight='bold', fontsize=20)

    plt.tight_layout()

    # Save plot
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"shared_classifier_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Plot saved: {filename}")

    # Also save without timestamp for easy reference
    latest_filename = output_dir / "shared_classifier_comparison.png"
    plt.savefig(latest_filename, dpi=300, bbox_inches='tight', facecolor='white')

    # Save core plot to main analysis directory if specified
    if core_output_dir:
        core_dir = Path(core_output_dir)
        core_dir.mkdir(parents=True, exist_ok=True)
        core_filename = core_dir / "shared_classifier_comparison.png"
        plt.savefig(core_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[OK] Core plot: {core_filename}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    return str(filename)


def save_summary(shared_counts, output_dir, dataset_name):
    """Save summary of shared gene counts as JSON and CSV."""
    output_dir = Path(output_dir)

    # JSON summary
    summary = {
        'dataset_name': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'shared_gene_counts': {
            MODEL_DISPLAY_NAMES.get(k, k): v for k, v in shared_counts.items()
        }
    }

    # Find best model
    if shared_counts:
        best_model = max(shared_counts.keys(), key=lambda k: shared_counts[k])
        summary['best_model'] = MODEL_DISPLAY_NAMES.get(best_model, best_model)
        summary['best_model_count'] = shared_counts[best_model]

    json_path = output_dir / "shared_classifier_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Summary saved: {json_path}")

    # CSV summary
    csv_data = []
    for model in MODEL_ORDER:
        if model in shared_counts:
            csv_data.append({
                'Model': MODEL_DISPLAY_NAMES[model],
                'Shared_Genes': shared_counts[model]
            })

    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = output_dir / "shared_classifier_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"[OK] CSV saved: {csv_path}")


def print_summary(shared_counts, dataset_name):
    """Print summary of shared gene counts."""
    print("\n" + "=" * 60)
    print("CLASSIFIER OVERLAP SUMMARY")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print("-" * 40)

    for model in MODEL_ORDER:
        if model in shared_counts:
            print(f"  {MODEL_DISPLAY_NAMES[model]}: {shared_counts[model]} shared genes")

    if shared_counts:
        best_model = max(shared_counts.keys(), key=lambda k: shared_counts[k])
        print("-" * 40)
        print(f"Best performance: {MODEL_DISPLAY_NAMES[best_model]} "
              f"({shared_counts[best_model]} shared genes)")
    print("=" * 60)


def main():
    args = parse_args()

    print("=" * 60)
    print("SHARED CLASSIFIER COMPARISON")
    print("=" * 60)

    clf_dir = Path(args.clf_dir)
    output_dir = Path(args.output_dir)

    print(f"\nConfiguration:")
    print(f"  Classifier results: {clf_dir}")
    print(f"  Real results: {args.real_results}")
    print(f"  Output: {output_dir}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  N genes: {args.n_genes}")

    # Parse color overrides if provided
    colors = None
    if args.colors:
        try:
            colors = json.loads(args.colors)
        except json.JSONDecodeError:
            print(f"[WARN] Could not parse colors JSON: {args.colors}")

    # Compute overlaps
    print("\n" + "-" * 40)
    print("Computing overlaps from RF results...")
    shared_counts = compute_overlaps_from_csvs(args.real_results, clf_dir, args.n_genes)

    if not shared_counts:
        print("[ERROR] No classifier results found for any model")
        sys.exit(1)

    # Print summary
    print_summary(shared_counts, args.dataset_name)

    # Create plot
    print("\n" + "-" * 40)
    print("Creating comparison plot...")
    plot_path = create_classifier_comparison_plot(
        shared_counts=shared_counts,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        group1_display=args.group1_display,
        group2_display=args.group2_display,
        n_genes=args.n_genes,
        colors=colors,
        show_plots=args.show_plots,
        core_output_dir=args.core_output_dir
    )

    # Save summary
    save_summary(shared_counts, output_dir, args.dataset_name)

    print("\n" + "=" * 60)
    print("SHARED CLASSIFIER COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Plot: {plot_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
