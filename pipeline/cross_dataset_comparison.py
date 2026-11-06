#!/usr/bin/env python3
"""
Cross-Dataset Comparison Bar Graphs

Creates bar graphs comparing:
1. Phenotypically Predictive Genes (from RF classifier overlap)
2. Differentially Expressed Genes (from DEG overlap)

across all datasets for CVAE, Diffusion, and GAN models.

Usage:
    python cross_dataset_comparison.py                    # Auto-discover from outputs/
    python cross_dataset_comparison.py --output-dir plots # Save to custom directory
    python cross_dataset_comparison.py --show             # Show plots interactively
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Script directory
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

# ============================================================================
# DATASET LIST - (folder_name, display_name) tuples
# Comment out datasets that don't have full data yet.
# If this list is empty/None, all datasets with analysis results are
# auto-discovered (display name will default to the folder name).
# ============================================================================
ENABLED_DATASETS = [
    ('OSDR-402',              'OSDR-402'),
    ('OSDR-403',              'OSDR-403'),
    ('OSDR-404',              'OSDR-404'),
    ('OSDR-405',                  'OSDR-405'),
    #('CRC_IMM_10X',                'CRC IMM 10X'),
]

# ============================================================================
# MANUAL DATA - Datasets not in the pipeline outputs
# ============================================================================
# Format: dataset_name -> {model: overlap_count}
# These are merged into the results keyed by display name.

MANUAL_CLASSIFIER_DATA = {
    'OSDR-612,613,352': {
        'cvae': 0,
        'diffusion': 5,
        'gan': 23,
    },
    'CANCER-IMMU 10x': {
        'cvae': 5,
        'diffusion': 16,
        'gan': 34,
    },
}

MANUAL_DEG_DATA = {
    'OSDR-612/613/352': {
        'cvae': 17,
        'diffusion': 20,
        'gan': 23,
    },
    'CANCER-IMMU 10x': {
        'cvae': 6,
        'diffusion': 17,
        'gan': 37,
    },
}

# Dataset display order (manual datasets will be appended after auto-discovered)
DATASET_ORDER = [
    'OSDR-402',
    'OSDR-403',
    'OSDR-404',
    'OSDR-405',
    'OSDR-612,613,352',
    'CANCER-IMMU 10x',
]

# ============================================================================

# Model order and colors (consistent with pipeline)
MODEL_ORDER = ['cvae', 'diffusion', 'gan']
COLORS = {
    'cvae': '#8E44AD',          # Purple
    'diffusion': 'lightcoral',  # Light coral
    'gan': '#E69F00',           # Orange
}

# Display names
MODEL_DISPLAY = {
    'cvae': 'CVAE',
    'diffusion': 'Diffusion',
    'gan': 'GAN'
}


def order_datasets(data: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """Order datasets according to DATASET_ORDER, with unknown datasets at the end."""
    ordered = {}

    # First add datasets in the specified order
    for dataset in DATASET_ORDER:
        if dataset in data:
            ordered[dataset] = data[dataset]

    # Then add any remaining datasets not in DATASET_ORDER
    for dataset in sorted(data.keys()):
        if dataset not in ordered:
            ordered[dataset] = data[dataset]

    return ordered


def discover_datasets(outputs_dir: Path) -> List[Tuple[str, str]]:
    """Return list of (folder_name, display_name) tuples.

    Uses ENABLED_DATASETS if set, otherwise auto-discovers (display name = folder name).
    """
    if ENABLED_DATASETS:
        datasets = []
        for folder, display in ENABLED_DATASETS:
            d = outputs_dir / folder
            if d.exists():
                datasets.append((folder, display))
            else:
                print(f"  [WARN] Dataset directory not found, skipping: {d}")
        return datasets

    # Auto-discover all datasets with analysis results
    datasets = []
    if not outputs_dir.exists():
        print(f"[WARN] Outputs directory not found: {outputs_dir}")
        return datasets

    for item in sorted(outputs_dir.iterdir()):
        if item.is_dir():
            analysis_dir = item / "analysis"
            if analysis_dir.exists():
                datasets.append((item.name, item.name))

    return datasets


def get_classifier_overlap(dataset_dir: Path) -> Optional[Dict[str, int]]:
    """Get classifier (phenotype predictive) overlap counts for a dataset."""
    # Try combined summary first
    summary_file = dataset_dir / "analysis" / "classifier" / "all_models_summary.csv"

    if summary_file.exists():
        try:
            df = pd.read_csv(summary_file)
            result = {}
            for _, row in df.iterrows():
                model = row['Model'].lower()
                if model in MODEL_ORDER:
                    result[model] = int(row['Overlap_Count'])
            return result
        except Exception as e:
            print(f"  [WARN] Error reading {summary_file}: {e}")

    # Fall back to individual files
    result = {}
    for model in MODEL_ORDER:
        overlap_file = dataset_dir / "analysis" / "classifier" / model / "overlap_summary.csv"
        if overlap_file.exists():
            try:
                df = pd.read_csv(overlap_file)
                overlap_row = df[df['Metric'] == 'Overlapping genes']
                if not overlap_row.empty:
                    result[model] = int(overlap_row['Value'].values[0])
            except Exception as e:
                print(f"  [WARN] Error reading {overlap_file}: {e}")

    return result if result else None


def get_deg_overlap(dataset_dir: Path) -> Optional[Dict[str, int]]:
    """Get DEG overlap counts for a dataset."""
    # Try combined summary first
    summary_file = dataset_dir / "analysis" / "deg" / "shared_deg_summary.json"

    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)

            shared_counts = data.get('shared_gene_counts', {})
            result = {}
            for model in MODEL_ORDER:
                # Try different key formats
                for key in [model, model.upper(), model.capitalize()]:
                    if key in shared_counts:
                        result[model] = int(shared_counts[key])
                        break
            return result
        except Exception as e:
            print(f"  [WARN] Error reading {summary_file}: {e}")

    # Fall back to individual files
    result = {}
    for model in MODEL_ORDER:
        overlap_file = dataset_dir / "analysis" / "deg" / model / "overlap_summary.csv"
        if overlap_file.exists():
            try:
                df = pd.read_csv(overlap_file)
                overlap_row = df[df['Metric'] == 'Overlapping genes']
                if not overlap_row.empty:
                    result[model] = int(overlap_row['Value'].values[0])
            except Exception as e:
                print(f"  [WARN] Error reading {overlap_file}: {e}")

    return result if result else None


def collect_data(outputs_dir: Path) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """Collect classifier and DEG overlap data for all datasets.

    Returns dicts keyed by display name.
    """
    datasets = discover_datasets(outputs_dir)
    print(f"\nDiscovered {len(datasets)} datasets: {', '.join(d for _, d in datasets)}")

    classifier_data = {}
    deg_data = {}

    for folder, display in datasets:
        dataset_dir = outputs_dir / folder
        print(f"\n  {display} ({folder})...")

        # Get classifier overlap
        clf_overlap = get_classifier_overlap(dataset_dir)
        if clf_overlap:
            classifier_data[display] = clf_overlap
            print(f"    Classifier: {clf_overlap}")
        else:
            print(f"    [SKIP] No classifier data found")

        # Get DEG overlap
        deg_overlap = get_deg_overlap(dataset_dir)
        if deg_overlap:
            deg_data[display] = deg_overlap
            print(f"    DEG: {deg_overlap}")
        else:
            print(f"    [SKIP] No DEG data found")

    # Add manual data (keyed by display name, won't overwrite auto-discovered)
    if MANUAL_CLASSIFIER_DATA:
        for dataset, values in MANUAL_CLASSIFIER_DATA.items():
            if dataset not in classifier_data:
                classifier_data[dataset] = values
                print(f"  [MANUAL] Added classifier data for {dataset}: {values}")

    if MANUAL_DEG_DATA:
        for dataset, values in MANUAL_DEG_DATA.items():
            if dataset not in deg_data:
                deg_data[dataset] = values
                print(f"  [MANUAL] Added DEG data for {dataset}: {values}")

    # Order datasets
    classifier_data = order_datasets(classifier_data)
    deg_data = order_datasets(deg_data)

    return classifier_data, deg_data


def create_bar_graph(data: Dict[str, Dict[str, int]],
                     title: str,
                     ylabel: str,
                     output_path: Path,
                     n_genes: int = 50,
                     show_percentage: bool = True,
                     show: bool = False):
    """Create a grouped bar graph comparing models across datasets."""

    if not data:
        print(f"[SKIP] No data for {title}")
        return

    datasets = list(data.keys())
    n_datasets = len(datasets)

    # Prepare data arrays
    cvae_values = []
    diffusion_values = []
    gan_values = []

    for dataset in datasets:
        model_data = data[dataset]
        cvae_values.append(model_data.get('cvae', 0))
        diffusion_values.append(model_data.get('diffusion', 0))
        gan_values.append(model_data.get('gan', 0))

    # Convert to percentages if requested
    if show_percentage:
        cvae_pct = [v / n_genes * 100 for v in cvae_values]
        diffusion_pct = [v / n_genes * 100 for v in diffusion_values]
        gan_pct = [v / n_genes * 100 for v in gan_values]
        display_values = (cvae_pct, diffusion_pct, gan_pct)
        value_suffix = '%'
    else:
        display_values = (cvae_values, diffusion_values, gan_values)
        value_suffix = ''

    # Set up the bar positions
    x = np.arange(n_datasets)
    width = 0.25

    # Create the figure
    fig, ax = plt.subplots(figsize=(max(10, n_datasets * 2), 6))

    # Create bars (order: CVAE, Diffusion, GAN from left to right)
    bars1 = ax.bar(x - width, display_values[0], width,
                   label='CVAE', color=COLORS['cvae'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, display_values[1], width,
                   label='Diffusion', color=COLORS['diffusion'], edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, display_values[2], width,
                   label='GAN', color=COLORS['gan'], edgecolor='black', linewidth=0.5)

    # Customize the plot
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11, rotation=15 if n_datasets > 4 else 0, ha='right' if n_datasets > 4 else 'center')
    ax.legend(fontsize=11, loc='upper left')

    # Set y-axis limit
    max_val = max(max(display_values[0]), max(display_values[1]), max(display_values[2]))
    ax.set_ylim(0, max(max_val * 1.15, 10))

    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bars, values in [(bars1, display_values[0]), (bars2, display_values[1]), (bars3, display_values[2])]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if show_percentage:
                label = f'{val:.0f}%'
            else:
                label = f'{int(val)}'
            ax.text(bar.get_x() + bar.get_width()/2., height + max_val * 0.01,
                    label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")

    if show:
        plt.show()
    plt.close()


def create_combined_figure(classifier_data: Dict[str, Dict[str, int]],
                           deg_data: Dict[str, Dict[str, int]],
                           output_path: Path,
                           n_genes: int = 50,
                           show: bool = False):
    """Create a combined figure with both bar graphs."""

    # Get union of datasets
    all_datasets = sorted(set(list(classifier_data.keys()) + list(deg_data.keys())))

    if not all_datasets:
        print("[SKIP] No data for combined figure")
        return

    n_datasets = len(all_datasets)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, n_datasets * 3), 6))

    # Helper function to plot bars
    def plot_bars(ax, data, datasets, title, ylabel):
        if not data:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')
            return

        # Filter to datasets that have data
        valid_datasets = [d for d in datasets if d in data]
        if not valid_datasets:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')
            return

        n = len(valid_datasets)
        x = np.arange(n)
        width = 0.25

        cvae_pct = [(data[d].get('cvae', 0) / n_genes * 100) for d in valid_datasets]
        diff_pct = [(data[d].get('diffusion', 0) / n_genes * 100) for d in valid_datasets]
        gan_pct = [(data[d].get('gan', 0) / n_genes * 100) for d in valid_datasets]

        bars1 = ax.bar(x - width, cvae_pct, width, label='CVAE', color=COLORS['cvae'], edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x, diff_pct, width, label='Diffusion', color=COLORS['diffusion'], edgecolor='black', linewidth=0.5)
        bars3 = ax.bar(x + width, gan_pct, width, label='GAN', color=COLORS['gan'], edgecolor='black', linewidth=0.5)

        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_datasets, fontsize=10, rotation=20 if n > 3 else 0, ha='right' if n > 3 else 'center')
        ax.legend(fontsize=9, loc='upper left')

        max_val = max(max(cvae_pct), max(diff_pct), max(gan_pct))
        ax.set_ylim(0, max(max_val * 1.2, 10))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add value labels
        for bars, values in [(bars1, cvae_pct), (bars2, diff_pct), (bars3, gan_pct)]:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max_val * 0.02,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Plot classifier data
    plot_bars(ax1, classifier_data, all_datasets,
              'A) Phenotypically Predictive Genes', 'Overlap Percentage (%)')

    # Plot DEG data
    plot_bars(ax2, deg_data, all_datasets,
              'B) Differentially Expressed Genes', 'Overlap Percentage (%)')

    plt.suptitle('Cross-Dataset Model Comparison (100 Epochs, Best Loss)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved combined figure: {output_path}")

    if show:
        plt.show()
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create cross-dataset comparison bar graphs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cross_dataset_comparison.py                    # Auto-discover and generate
  python cross_dataset_comparison.py --output-dir plots # Save to custom directory
  python cross_dataset_comparison.py --show             # Show plots interactively
  python cross_dataset_comparison.py --n-genes 100      # Use different gene count for percentage
        """
    )

    parser.add_argument('--outputs-dir', default=None,
                        help='Path to outputs directory (default: ./outputs)')
    parser.add_argument('--output-dir', default='cross_dataset_plots',
                        help='Directory to save plots (default: cross_dataset_plots)')
    parser.add_argument('--n-genes', type=int, default=50,
                        help='Number of genes used in analysis (for percentage calculation)')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    parser.add_argument('--no-percentage', action='store_true',
                        help='Show raw counts instead of percentages')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Cross-Dataset Comparison")
    print("=" * 60)

    # Set directories
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else OUTPUTS_DIR
    output_dir = SCRIPT_DIR / args.output_dir

    print(f"\nOutputs directory: {outputs_dir}")
    print(f"Save to: {output_dir}")
    print(f"N genes: {args.n_genes}")

    # Collect data
    classifier_data, deg_data = collect_data(outputs_dir)

    if not classifier_data and not deg_data:
        print("\n[ERROR] No data found!")
        return 1

    # Create individual plots
    print("\n" + "-" * 40)
    print("Creating plots...")

    if classifier_data:
        create_bar_graph(
            data=classifier_data,
            title='Phenotypically Predictive Genes Across Datasets (100 Epochs, Best Loss)',
            ylabel='Overlap Percentage (%)' if not args.no_percentage else 'Overlap Count',
            output_path=output_dir / 'phenotype_predictive_genes_comparison.png',
            n_genes=args.n_genes,
            show_percentage=not args.no_percentage,
            show=args.show
        )

    if deg_data:
        create_bar_graph(
            data=deg_data,
            title='Shared DEGs Across Datasets (100 Epochs, Best Loss)',
            ylabel='Overlap Percentage (%)' if not args.no_percentage else 'Overlap Count',
            output_path=output_dir / 'deg_overlap_comparison.png',
            n_genes=args.n_genes,
            show_percentage=not args.no_percentage,
            show=args.show
        )

    # Create combined figure
    if classifier_data or deg_data:
        create_combined_figure(
            classifier_data=classifier_data,
            deg_data=deg_data,
            output_path=output_dir / 'cross_dataset_combined_comparison.png',
            n_genes=args.n_genes,
            show=args.show
        )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nClassifier (Phenotype Predictive) Overlap:")
    for dataset, values in classifier_data.items():
        print(f"  {dataset}: CVAE={values.get('cvae', 'N/A')}, Diffusion={values.get('diffusion', 'N/A')}, GAN={values.get('gan', 'N/A')}")

    print("\nDEG Overlap:")
    for dataset, values in deg_data.items():
        print(f"  {dataset}: CVAE={values.get('cvae', 'N/A')}, Diffusion={values.get('diffusion', 'N/A')}, GAN={values.get('gan', 'N/A')}")

    print(f"\nPlots saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
