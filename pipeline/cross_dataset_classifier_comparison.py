#!/usr/bin/env python3
"""
Cross-Dataset Classifier (Phenotypically Predictive Genes) Comparison

Auto-discovers all datasets from an outputs directory and creates a bar graph
comparing classifier overlap counts across CVAE, Diffusion, and GAN models.
Optionally annotates each dataset group with the MMD spread from matched
checkpoint generation (use --mmd flag).

Usage:
    python cross_dataset_classifier_comparison.py
    python cross_dataset_classifier_comparison.py --outputs-dir D:/NASAResearch/AI4LS/pipeline/outputs
    python cross_dataset_classifier_comparison.py --mmd          # include MMD spread annotations
    python cross_dataset_classifier_comparison.py --plot-name my_clf_plot
    python cross_dataset_classifier_comparison.py --show
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Defaults
DEFAULT_OUTPUTS_DIR = Path(r"D:\NASAResearch\AI4LS\pipeline\outputs")
SCRIPT_DIR = Path(__file__).parent.resolve()

# ============================================================================
# DATASET LIST - (folder_name, display_name) tuples
# Comment out datasets that don't have full data yet.
# If this list is empty/None, all datasets with classifier results are
# auto-discovered (display name will default to the folder name).
# ============================================================================
ENABLED_DATASETS = [
    ('OSDR-402-mmd-complete',    'OSDR-402'),
    ('OSDR-403-mmd-complete',    'OSDR-403'),
    ('OSDR-404-mmd-complete',    'OSDR-404'),
    ('OSDR-405-mmd',        'OSDR-405'),
    ('OSDR-352-612-613-mmd-complete',  'OSDR-352/612/613'),
    ('CRC_IMM_10X-mmd-complete',       'CRC IMM 10X'),
]

# Model config
MODEL_ORDER = ['cvae', 'diffusion', 'gan']
COLORS = {
    'cvae': '#8E44AD',
    'diffusion': 'lightcoral',
    'gan': '#E69F00',
}
MODEL_DISPLAY = {
    'cvae': 'CVAE',
    'diffusion': 'Diffusion',
    'gan': 'GAN',
}


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

    # Auto-discover all datasets with classifier results
    datasets = []
    if not outputs_dir.exists():
        print(f"[WARN] Outputs directory not found: {outputs_dir}")
        return datasets

    for item in sorted(outputs_dir.iterdir()):
        if item.is_dir():
            clf_dir = item / "analysis" / "classifier"
            if clf_dir.exists():
                datasets.append((item.name, item.name))

    return datasets


def get_classifier_overlap(dataset_dir: Path) -> Optional[Dict[str, int]]:
    """Get classifier overlap counts from all_models_summary.csv or individual files."""
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
            return result if result else None
        except Exception as e:
            print(f"  [WARN] Error reading {summary_file}: {e}")

    # Fallback to individual overlap_summary.csv files
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


def get_mmd_spread(dataset_dir: Path) -> Optional[float]:
    """Get MMD spread from the most recent matched_mmd_info JSON in synthetic_data/."""
    synth_dir = dataset_dir / "synthetic_data"
    if not synth_dir.exists():
        return None

    info_files = sorted(synth_dir.glob("matched_mmd_info_*.json"))
    if not info_files:
        return None

    # Use the most recent one (sorted alphabetically by timestamp)
    latest = info_files[-1]
    try:
        with open(latest, 'r') as f:
            data = json.load(f)
        return data.get('mmd_spread')
    except Exception as e:
        print(f"  [WARN] Error reading {latest}: {e}")
        return None


def collect_data(outputs_dir: Path) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Optional[float]]]:
    """Collect classifier overlap and MMD spread for all datasets.

    Returns dicts keyed by display name.
    """
    datasets = discover_datasets(outputs_dir)
    print(f"\nDiscovered {len(datasets)} datasets: {', '.join(d for _, d in datasets)}")

    clf_data = {}
    mmd_spreads = {}

    for folder, display in datasets:
        dataset_dir = outputs_dir / folder
        print(f"\n  {display} ({folder})...")

        clf_overlap = get_classifier_overlap(dataset_dir)
        if clf_overlap:
            clf_data[display] = clf_overlap
            print(f"    Classifier: {clf_overlap}")
        else:
            print(f"    [SKIP] No classifier data")

        spread = get_mmd_spread(dataset_dir)
        if spread is not None:
            mmd_spreads[display] = spread
            print(f"    MMD spread: {spread:.4f}")
        else:
            mmd_spreads[display] = None
            print(f"    MMD spread: N/A")

    return clf_data, mmd_spreads


def _build_classifier_figure(clf_data: Dict[str, Dict[str, int]],
                              mmd_spreads: Optional[Dict[str, Optional[float]]],
                              n_genes: int = 50,
                              show_percentage: bool = True,
                              show_mmd: bool = False,
                              title: str = 'Phenotypically Predictive Genes Across Datasets'):
    """Build and return a figure for classifier bar graph, optionally with MMD annotations."""
    datasets = list(clf_data.keys())
    n_datasets = len(datasets)

    # Gather values per model
    model_values = {}
    for model in MODEL_ORDER:
        model_values[model] = [clf_data[d].get(model, 0) for d in datasets]

    # Convert to percentages
    if show_percentage:
        display = {m: [v / n_genes * 100 for v in vals] for m, vals in model_values.items()}
        value_fmt = lambda v: f'{v:.0f}%'
        ylabel = 'Overlap Percentage (%)'
    else:
        display = model_values
        value_fmt = lambda v: f'{int(v)}'
        ylabel = 'Overlap Count'

    # Bar positions
    x = np.arange(n_datasets)
    width = 0.25

    fig_height = 8 if show_mmd else 7
    fig, ax = plt.subplots(figsize=(max(10, n_datasets * 2.2), fig_height))

    # Draw bars
    bar_groups = []
    offsets = [-width, 0, width]
    for model, offset in zip(MODEL_ORDER, offsets):
        bars = ax.bar(x + offset, display[model], width,
                      label=MODEL_DISPLAY[model], color=COLORS[model],
                      edgecolor='black', linewidth=0.5)
        bar_groups.append((bars, display[model]))

    # Value labels on bars
    max_val = max(max(vals) for vals in display.values()) if any(display.values()) else 10
    for bars, vals in bar_groups:
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + max_val * 0.01,
                    value_fmt(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # MMD spread annotation above each dataset group
    if show_mmd and mmd_spreads:
        annotation_y = max_val * 1.18
        for i, dataset in enumerate(datasets):
            spread = mmd_spreads.get(dataset)
            if spread is not None:
                ax.text(x[i], annotation_y, f'MMD spread: {spread:.4f}',
                        ha='center', va='bottom', fontsize=8, fontstyle='italic',
                        color='#555555',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                                  edgecolor='#cccccc', alpha=0.8))

    # Axes
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11,
                       rotation=15 if n_datasets > 4 else 0,
                       ha='right' if n_datasets > 4 else 'center')

    # Place legend in upper right to avoid MMD annotations
    ax.legend(fontsize=11, loc='upper right')

    # Y limit with room for annotations
    has_any_spread = show_mmd and mmd_spreads and any(v is not None for v in mmd_spreads.values())
    top_margin = 1.45 if has_any_spread else 1.15
    ax.set_ylim(0, max(max_val * top_margin, 10))

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def save_classifier_bar_graphs(clf_data: Dict[str, Dict[str, int]],
                                mmd_spreads: Dict[str, Optional[float]],
                                output_dir: Path,
                                n_genes: int = 50,
                                show_percentage: bool = True,
                                show: bool = False,
                                include_mmd: bool = False,
                                plot_name: str = 'classifier_overlap_comparison'):
    """Create and save classifier bar graphs.

    When include_mmd is True, saves two plots: plain and with MMD annotations.
    When include_mmd is False, saves only the plain plot.
    """
    if not clf_data:
        print("[SKIP] No classifier data to plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize plot_name for use as a filename (replace path-unsafe characters)
    safe_name = plot_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    for ch in '<>"|?*':
        safe_name = safe_name.replace(ch, '_')

    # Graph without MMD (always saved)
    fig = _build_classifier_figure(clf_data, mmd_spreads, n_genes, show_percentage,
                                    show_mmd=False, title=plot_name)
    path_plain = output_dir / f'{safe_name}.png'
    fig.savefig(path_plain, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {path_plain}")
    if show:
        plt.show()
    plt.close(fig)

    # Graph with MMD (only when requested)
    if include_mmd:
        fig = _build_classifier_figure(clf_data, mmd_spreads, n_genes, show_percentage,
                                        show_mmd=True, title=plot_name)
        path_mmd = output_dir / f'{safe_name}_with_mmd.png'
        fig.savefig(path_mmd, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {path_mmd}")
        if show:
            plt.show()
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Cross-dataset classifier comparison (optionally with MMD spread annotations)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cross_dataset_classifier_comparison.py
  python cross_dataset_classifier_comparison.py --outputs-dir D:/NASAResearch/AI4LS/pipeline/outputs
  python cross_dataset_classifier_comparison.py --mmd          # include MMD spread annotations
  python cross_dataset_classifier_comparison.py --plot-name my_custom_plot
  python cross_dataset_classifier_comparison.py --output-dir my_plots --show
  python cross_dataset_classifier_comparison.py --no-percentage
        """
    )

    parser.add_argument('--outputs-dir', default=None,
                        help=f'Path to outputs directory (default: {DEFAULT_OUTPUTS_DIR})')
    parser.add_argument('--output-dir', default='cross_dataset_plots',
                        help='Directory to save plots (default: cross_dataset_plots)')
    parser.add_argument('--n-genes', type=int, default=50,
                        help='Number of genes used in analysis (for percentage calc)')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    parser.add_argument('--no-percentage', action='store_true',
                        help='Show raw counts instead of percentages')
    parser.add_argument('--mmd', action='store_true',
                        help='Include MMD spread annotations (generates additional _with_mmd plot)')
    parser.add_argument('--plot-name', default='classifier_overlap_comparison',
                        help='Base filename for saved plots (default: classifier_overlap_comparison)')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Cross-Dataset Classifier Comparison")
    print("=" * 60)

    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else DEFAULT_OUTPUTS_DIR
    output_dir = SCRIPT_DIR / args.output_dir

    print(f"\nOutputs directory: {outputs_dir}")
    print(f"Save to: {output_dir}")
    print(f"N genes: {args.n_genes}")
    print(f"MMD annotations: {'enabled' if args.mmd else 'disabled'}")
    print(f"Plot name: {args.plot_name}")

    clf_data, mmd_spreads = collect_data(outputs_dir)

    if not clf_data:
        print("\n[ERROR] No classifier data found!")
        return 1

    print("\n" + "-" * 40)
    print("Creating plots...")

    save_classifier_bar_graphs(
        clf_data=clf_data,
        mmd_spreads=mmd_spreads,
        output_dir=output_dir,
        n_genes=args.n_genes,
        show_percentage=not args.no_percentage,
        show=args.show,
        include_mmd=args.mmd,
        plot_name=args.plot_name,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for dataset, values in clf_data.items():
        line = (f"  {dataset}: CVAE={values.get('cvae', 'N/A')}, "
                f"Diffusion={values.get('diffusion', 'N/A')}, "
                f"GAN={values.get('gan', 'N/A')}")
        if args.mmd:
            spread = mmd_spreads.get(dataset)
            spread_str = f"{spread:.4f}" if spread is not None else "N/A"
            line += f"  (MMD spread: {spread_str})"
        print(line)

    print(f"\nPlots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
