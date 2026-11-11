#!/usr/bin/env python3
"""
Directional DEG-overlap analysis for OSDR-352-612-613 (standoff / cross-analysis tool).

Compares, between Real and each synthetic model (CVAE, Diffusion, GAN), the overlap of
the top-50 UPREGULATED genes and (separately) the top-50 DOWNREGULATED genes.

DEG method mirrors pipeline/analysis/compute_deg.py EXACTLY:
    groupby column = "Flight"; reference group = "1.0" (group1); target group = "0.0" (group2)
    sc.tl.rank_genes_groups(..., method="wilcoxon", reference="1.0", use_raw=False)
    df = sc.get.rank_genes_groups_df(adata, group="0.0")
    abs_logfoldchanges = |logfoldchanges|
    sort by [pvals_adj asc, abs_logfoldchanges desc]
We keep the FULL ranked table (no head(50) truncation, no significance filter), then split
by direction of logfoldchanges and take the top 50 per direction.

MEMORY: loads ONE h5ad at a time; del + gc.collect() before loading the next. Never holds two
of these 214k-cell datasets in memory at once.

Does NOT modify anything under pipeline/.
"""

import os
import gc
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sc.settings.verbosity = 1

# =============================================================================
# CONFIG
# =============================================================================
COLUMN_NAME = "Flight"
GROUP1 = "1.0"   # reference
GROUP2 = "0.0"   # target
METHOD = "wilcoxon"
TOP_N = 50
DATASET_NAME = "OSDR-352-612-613"

OUT_DIR = Path("/Users/ljeytl/Arsh/Research/NASA/AI4LS/BEST_bioeval/"
               "cross_analysis_scripts/results/OSDR-352-612-613_deg_updown")

REAL_PATH = ("/Users/ljeytl/Arsh/Research/NASA/AI4LS/BEST_bioeval/pipeline/"
             "real_datasets/osdr_352_612_613_raw_0.997_filtered_hvg_2000_norm.h5ad")

SSD_SYNTH_DIR = ("/Volumes/Ljeytl's SSD/NASAResearch/AI4LS/pipeline/outputs/"
                 "OSDR-352-612-613-mmd-complete-pca/synthetic_data")

DATASETS = {
    "Real": REAL_PATH,
    "cvae": os.path.join(SSD_SYNTH_DIR,
                         "cvae_bestmmd_OSDR-352-612-613_20260403_071100.h5ad"),
    "diffusion": os.path.join(SSD_SYNTH_DIR,
                              "diffusion_bestmmd_OSDR-352-612-613_20260403_071100.h5ad"),
    "gan": os.path.join(SSD_SYNTH_DIR,
                        "gan_bestmmd_OSDR-352-612-613_20260403_071100.h5ad"),
}

MODEL_ORDER = ['cvae', 'diffusion', 'gan']
MODEL_DISPLAY_NAMES = {'cvae': 'CVAE', 'diffusion': 'Diffusion', 'gan': 'GAN'}
DEFAULT_COLORS = {'cvae': '#8E44AD', 'diffusion': '#E57373', 'gan': '#E69F00'}

# Plot styling matching shared_classifier_comparison.py
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
    'legend.shadow': True,
})


# =============================================================================
# DEG COMPUTATION (full ranked table, matches compute_deg.py logic)
# =============================================================================
def compute_full_deg(adata):
    """Return the FULL ranked DEG table for target group GROUP2 vs reference GROUP1."""
    if COLUMN_NAME not in adata.obs.columns:
        raise ValueError(f"Column '{COLUMN_NAME}' not found in obs. "
                         f"Available: {list(adata.obs.columns)}")

    # Convert to string and filter to the two groups (as compute_deg does)
    adata.obs[COLUMN_NAME] = adata.obs[COLUMN_NAME].astype(str)
    mask = adata.obs[COLUMN_NAME].isin([GROUP1, GROUP2])
    adata = adata[mask].copy()

    unique_vals = set(adata.obs[COLUMN_NAME].unique())
    for g in [GROUP1, GROUP2]:
        if g not in unique_vals:
            raise ValueError(f"Group '{g}' not found in '{COLUMN_NAME}'. Found: {unique_vals}")

    counts = adata.obs[COLUMN_NAME].value_counts()
    print(f"  Group distribution in '{COLUMN_NAME}':")
    for group, count in counts.items():
        print(f"    {group}: {count} cells")

    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)

    print(f"  Comparing '{GROUP2}' vs reference '{GROUP1}' using {METHOD}...")
    sc.tl.rank_genes_groups(
        adata,
        COLUMN_NAME,
        method=METHOD,
        reference=GROUP1,
        use_raw=False,
    )

    df = sc.get.rank_genes_groups_df(adata, group=GROUP2)
    df['abs_logfoldchanges'] = np.abs(df['logfoldchanges'])
    # Full composite sort (no truncation)
    df = df.sort_values(['pvals_adj', 'abs_logfoldchanges'],
                        ascending=[True, False]).reset_index(drop=True)
    print(f"  Full ranked table: {len(df)} genes")
    return df


def directional_split(df, top_n=TOP_N):
    """Return (up_genes, down_genes) lists of top-N names per direction."""
    up = df[df['logfoldchanges'] > 0].sort_values(
        ['pvals_adj', 'abs_logfoldchanges'], ascending=[True, False])
    down = df[df['logfoldchanges'] < 0].sort_values(
        ['pvals_adj', 'abs_logfoldchanges'], ascending=[True, False])
    up_genes = up['names'].head(top_n).tolist()
    down_genes = down['names'].head(top_n).tolist()
    return up_genes, down_genes, len(up), len(down)


# =============================================================================
# OVERLAP
# =============================================================================
def compute_overlap(real_genes, model_genes):
    real_set = set(real_genes)
    model_set = set(model_genes)
    inter = sorted(real_set & model_set)
    union = real_set | model_set
    overlap_count = len(inter)
    jaccard = overlap_count / len(union) if union else 0.0
    overlap_pct_real = (100.0 * overlap_count / len(real_set)) if real_set else 0.0
    unique_real = sorted(real_set - model_set)
    unique_model = sorted(model_set - real_set)
    return inter, unique_real, unique_model, overlap_count, jaccard, overlap_pct_real


def write_overlap_outputs(direction, model, real_genes, model_genes):
    """Write per-model overlap files into results/<direction>/<model>/ ; return summary dict."""
    (inter, unique_real, unique_model,
     overlap_count, jaccard, overlap_pct_real) = compute_overlap(real_genes, model_genes)

    mdir = OUT_DIR / direction / model
    mdir.mkdir(parents=True, exist_ok=True)
    muc = model.upper()

    overlap_pct_model = (100.0 * overlap_count / len(set(model_genes))) if model_genes else 0.0
    summary = pd.DataFrame([
        {'Metric': 'Direction', 'Value': direction.upper()},
        {'Metric': 'Top N genes analyzed', 'Value': TOP_N},
        {'Metric': 'Real genes', 'Value': len(set(real_genes))},
        {'Metric': f'Synthetic_{muc} genes', 'Value': len(set(model_genes))},
        {'Metric': 'Overlapping genes', 'Value': overlap_count},
        {'Metric': 'Jaccard Index', 'Value': f"{jaccard:.4f}"},
        {'Metric': 'Overlap % (Real)', 'Value': f"{overlap_pct_real:.2f}%"},
        {'Metric': f'Overlap % (Synthetic_{muc})', 'Value': f"{overlap_pct_model:.2f}%"},
    ])
    summary.to_csv(mdir / 'overlap_summary.csv', index=False)
    pd.DataFrame({'Gene': inter}).to_csv(mdir / 'overlapping_genes.csv', index=False)
    pd.DataFrame({'Gene': unique_real}).to_csv(mdir / 'unique_to_Real.csv', index=False)
    pd.DataFrame({'Gene': unique_model}).to_csv(
        mdir / f'unique_to_{muc}.csv', index=False)

    print(f"  [{direction.upper()}] {MODEL_DISPLAY_NAMES[model]}: "
          f"{overlap_count}/{TOP_N} overlap (Jaccard={jaccard:.4f})")
    return {
        'model': model,
        'direction': direction,
        'overlap_count': overlap_count,
        'jaccard': round(jaccard, 4),
        'overlap_pct_of_real': round(overlap_pct_real, 2),
    }


# =============================================================================
# PLOTS
# =============================================================================
def _save_fig(fig, basename):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stamped = OUT_DIR / f"{basename}_{ts}.png"
    latest = OUT_DIR / f"{basename}.png"
    fig.savefig(stamped, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(latest, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Saved: {latest}")


def plot_single_direction(direction, counts_by_model):
    """Bar chart of overlap counts per model for one direction."""
    models, counts, colors = [], [], []
    for m in MODEL_ORDER:
        if m in counts_by_model:
            models.append(MODEL_DISPLAY_NAMES[m])
            counts.append(counts_by_model[m])
            colors.append(DEFAULT_COLORS[m])

    fig, ax = plt.subplots(figsize=(14, 9))
    bars = ax.bar(models, counts, color=colors, alpha=0.9,
                  edgecolor='black', linewidth=2, width=0.5)
    ax.set_title(
        f'Overlap of Top-{TOP_N} {direction.upper()}regulated DEGs\n'
        f'Real vs Synthetic Data for {DATASET_NAME}',
        fontsize=24, fontweight='bold', pad=25)
    ax.set_xlabel('Synthetic Data Model', fontsize=22, fontweight='bold')
    ax.set_ylabel(f'Number of Overlapping Genes (out of {TOP_N})',
                  fontsize=22, fontweight='bold')
    ax.set_ylim(0, TOP_N)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#FAFAFA')
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='x', which='major', pad=12, labelsize=20)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + TOP_N * 0.02,
                f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=20)
    plt.tight_layout()
    _save_fig(fig, f"deg_overlap_{direction.upper()}")


def plot_combined(up_by_model, down_by_model):
    """Grouped bars: UP vs DOWN side by side per model."""
    models = [MODEL_DISPLAY_NAMES[m] for m in MODEL_ORDER]
    up_vals = [up_by_model.get(m, 0) for m in MODEL_ORDER]
    down_vals = [down_by_model.get(m, 0) for m in MODEL_ORDER]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 9))
    bars_up = ax.bar(x - width / 2, up_vals, width, label='UP',
                     color='#2E86C1', alpha=0.9, edgecolor='black', linewidth=2)
    bars_down = ax.bar(x + width / 2, down_vals, width, label='DOWN',
                       color='#CB4335', alpha=0.9, edgecolor='black', linewidth=2)
    ax.set_title(
        f'Directional DEG Overlap (Top-{TOP_N} per direction)\n'
        f'Real vs Synthetic Data for {DATASET_NAME}',
        fontsize=24, fontweight='bold', pad=25)
    ax.set_xlabel('Synthetic Data Model', fontsize=22, fontweight='bold')
    ax.set_ylabel(f'Number of Overlapping Genes (out of {TOP_N})',
                  fontsize=22, fontweight='bold')
    ax.set_ylim(0, TOP_N)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#FAFAFA')
    ax.legend(title='Direction')
    for bars in (bars_up, bars_down):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + TOP_N * 0.02,
                    f'{int(h)}', ha='center', va='bottom',
                    fontweight='bold', fontsize=18)
    plt.tight_layout()
    _save_fig(fig, "deg_overlap_updown_combined")


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"DIRECTIONAL DEG OVERLAP: {DATASET_NAME}")
    print("=" * 70)

    up_lists, down_lists = {}, {}
    counts_up, counts_down = {}, {}  # of up/down genes Real/each actually had

    # Process ONE dataset at a time (memory discipline)
    for label, path in DATASETS.items():
        print(f"\n--- {label} ---")
        print(f"  Path: {path}")
        if not os.path.exists(path):
            print(f"  [ERROR] File not found: {path}")
            sys.exit(1)
        adata = sc.read_h5ad(path)
        print(f"  Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

        df = compute_full_deg(adata)
        df.to_csv(OUT_DIR / f"{label}_deg_full.csv", index=False)

        up_genes, down_genes, n_up, n_down = directional_split(df)
        up_lists[label] = up_genes
        down_lists[label] = down_genes
        counts_up[label] = n_up
        counts_down[label] = n_down

        pd.DataFrame({'Gene': up_genes}).to_csv(OUT_DIR / f"{label}_up50.csv", index=False)
        pd.DataFrame({'Gene': down_genes}).to_csv(OUT_DIR / f"{label}_down50.csv", index=False)
        print(f"  Direction totals -> UP genes: {n_up}, DOWN genes: {n_down} "
              f"(top-{TOP_N} saved)")

        # Free memory before next dataset
        del adata, df
        gc.collect()

    # Overlaps: Real vs each model, UP and DOWN
    print("\n" + "=" * 70)
    print("OVERLAPS (Real vs each model)")
    print("=" * 70)
    real_up = up_lists['Real']
    real_down = down_lists['Real']

    combined_rows = []
    up_overlap_counts, down_overlap_counts = {}, {}
    for model in MODEL_ORDER:
        row_up = write_overlap_outputs('up', model, real_up, up_lists[model])
        row_down = write_overlap_outputs('down', model, real_down, down_lists[model])
        up_overlap_counts[model] = row_up['overlap_count']
        down_overlap_counts[model] = row_down['overlap_count']
        combined_rows.append(row_up)
        combined_rows.append(row_down)

    combined_df = pd.DataFrame(combined_rows,
                               columns=['model', 'direction', 'overlap_count',
                                        'jaccard', 'overlap_pct_of_real'])
    combined_df.to_csv(OUT_DIR / "overlap_summary_updown.csv", index=False)
    print(f"\n[OK] Combined summary: {OUT_DIR / 'overlap_summary_updown.csv'}")

    # Record how many up/down genes each dataset actually had
    dir_totals = pd.DataFrame([
        {'dataset': k, 'n_up_genes': counts_up[k], 'n_down_genes': counts_down[k]}
        for k in DATASETS
    ])
    dir_totals.to_csv(OUT_DIR / "direction_totals.csv", index=False)

    # Plots
    print("\n" + "=" * 70)
    print("PLOTS")
    print("=" * 70)
    plot_single_direction('up', up_overlap_counts)
    plot_single_direction('down', down_overlap_counts)
    plot_combined(up_overlap_counts, down_overlap_counts)

    # JSON summary for convenience
    summary = {
        'dataset_name': DATASET_NAME,
        'timestamp': datetime.now().isoformat(),
        'top_n': TOP_N,
        'real_n_up_genes': counts_up['Real'],
        'real_n_down_genes': counts_down['Real'],
        'up_overlap_counts': {MODEL_DISPLAY_NAMES[m]: up_overlap_counts[m] for m in MODEL_ORDER},
        'down_overlap_counts': {MODEL_DISPLAY_NAMES[m]: down_overlap_counts[m] for m in MODEL_ORDER},
    }
    with open(OUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Real UP genes available: {counts_up['Real']}, "
          f"Real DOWN genes available: {counts_down['Real']}")
    print(f"UP overlaps  : " +
          ", ".join(f"{MODEL_DISPLAY_NAMES[m]}={up_overlap_counts[m]}" for m in MODEL_ORDER))
    print(f"DOWN overlaps: " +
          ", ".join(f"{MODEL_DISPLAY_NAMES[m]}={down_overlap_counts[m]}" for m in MODEL_ORDER))
    print(f"Output dir: {OUT_DIR}")


def self_test():
    """Validate the pure helpers without loading any real h5ad file."""
    print("Running self-test (no real-data load)...")
    # Fake ranked DEG table: 4 up, 3 down genes with known ordering
    df = pd.DataFrame({
        'names': ['gU1', 'gU2', 'gU3', 'gU4', 'gD1', 'gD2', 'gD3'],
        'scores': [9, 8, 7, 6, -9, -8, -7],
        'logfoldchanges': [3.0, 2.0, 1.5, 0.5, -3.0, -2.0, -1.0],
        'pvals': [0.0] * 7,
        'pvals_adj': [0.0] * 7,
    })
    df['abs_logfoldchanges'] = np.abs(df['logfoldchanges'])
    df = df.sort_values(['pvals_adj', 'abs_logfoldchanges'],
                        ascending=[True, False]).reset_index(drop=True)

    up_genes, down_genes, n_up, n_down = directional_split(df, top_n=50)
    assert up_genes == ['gU1', 'gU2', 'gU3', 'gU4'], up_genes
    assert down_genes == ['gD1', 'gD2', 'gD3'], down_genes
    assert (n_up, n_down) == (4, 3), (n_up, n_down)

    # top_n truncation
    up2, down2, _, _ = directional_split(df, top_n=2)
    assert up2 == ['gU1', 'gU2'], up2
    assert down2 == ['gD1', 'gD2'], down2

    # Overlap: real {A,B,C} vs model {B,C,D}
    inter, uniq_r, uniq_m, cnt, jac, pct = compute_overlap(['A', 'B', 'C'], ['B', 'C', 'D'])
    assert inter == ['B', 'C'], inter
    assert uniq_r == ['A'], uniq_r
    assert uniq_m == ['D'], uniq_m
    assert cnt == 2, cnt
    assert abs(jac - 2 / 4) < 1e-9, jac
    assert abs(pct - 100 * 2 / 3) < 1e-9, pct

    # Disjoint and identical edge cases
    _, _, _, cnt0, jac0, _ = compute_overlap(['A'], ['B'])
    assert cnt0 == 0 and jac0 == 0.0
    _, _, _, cntI, jacI, pctI = compute_overlap(['A', 'B'], ['A', 'B'])
    assert cntI == 2 and jacI == 1.0 and pctI == 100.0

    print("[OK] Self-test passed: directional split + overlap/Jaccard verified.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--self-test", "--dry-run"):
        self_test()
    else:
        main()
