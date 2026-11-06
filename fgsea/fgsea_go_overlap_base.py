#!/usr/bin/env python3
"""
fgsea-style GO gene-set overlap analysis for the BASE (100-epoch) training runs.

Same analysis as fgsea_go_overlap.py (top-N RF genes -> GO over-representation ->
top-N GO sets -> overlap with Real), but pointed at the base/100-epoch dataset
folders on the SSD, which use a MIX of two classifier output layouts:

  NEW layout  (OSDR-402-jan26, OSDR-404-jan26, CRC_IMM_10X, OSDR-352-612-613):
      analysis/classifier/real/rf_results.csv
      analysis/classifier/<model>/rf_results.csv

  OLD layout  (OSDR-405-2, OSDR-403-jan26):
      analysis/classifier/<model>/rf_results_Real.csv            (Real, duplicated per pair)
      analysis/classifier/<model>/rf_results_Synthetic_<MODEL>.csv

read_base_genes() auto-detects which layout a given model folder uses, so all six
datasets run through one pass. Everything else (GO loading, enrichment, FDR filter,
overlap metric, colours, plot styling) is imported from fgsea_go_overlap so the two
scripts never drift. Results go to their OWN folder so they never touch the headline
run:  NASA/fgsea/base_100epoch/

Run:
    /Users/ljeytl/Arsh/Research/NASA/fgsea/.venv_fgsea/bin/python fgsea_go_overlap_base.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fgsea_go_overlap as fg  # pure helpers + constants, shared cache

# ==========================================================================
# CONFIG  -- edit here only
# ==========================================================================

# Same SSD outputs root as the headline run (reuses fg's value so it stays in sync).
BASE_OUTPUTS_DIR = fg.GENERIC_OUTPUTS_DIR

# The base / 100-epoch run lives in its OWN output folder, never overwriting the
# headline (-mmd-complete-pca) results that share these display names.
OUT_ROOT = os.path.join(fg.NASA_DIR, "base_100epoch")

# base folder on disk  ->  publication display name
DATASETS = {
    "OSDR-405-2":       "OSDR-405",
    "OSDR-404-jan26":   "OSDR-404",
    "OSDR-403-jan26":   "OSDR-403",
    "OSDR-402-jan26":   "OSDR-402",
    "CRC_IMM_10X":      "CRC_IMM_10X",
    "OSDR-352-612-613": "OSDR-352-612-613",
}

# Reused verbatim from the headline script (single source of truth):
MODELS = fg.MODELS                # {cvae,diffusion,gan} -> label
COLORS = fg.COLORS                # label -> colour
TOP_N_GENES = fg.TOP_N_GENES
TOP_N_GO = fg.TOP_N_GO
FDR_Q_THRESHOLD = fg.FDR_Q_THRESHOLD
PLOT_DPI = fg.PLOT_DPI

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ==========================================================================
# end CONFIG
# ==========================================================================


def _classifier_dir(dataset_folder):
    return os.path.join(BASE_OUTPUTS_DIR, dataset_folder, "analysis", "classifier")


def _read_rf(path):
    """Top-TOP_N_GENES gene symbols (by RF rank) from an rf_results csv, or None."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "rank" in df.columns:
        df = df.sort_values("rank")
    elif "mean_importance" in df.columns:
        df = df.sort_values("mean_importance", ascending=False)
    return df["gene"].astype(str).head(TOP_N_GENES).tolist()


def read_base_genes(dataset_folder, model):
    """Genes for (Real, model) handling BOTH classifier layouts.

    Returns (real_genes, model_genes); either may be None if that file is absent.
    model is the lowercase folder name (cvae/diffusion/gan); label is MODELS[model].
    """
    label = MODELS[model]
    cdir = _classifier_dir(dataset_folder)
    mdir = os.path.join(cdir, model)

    # ---- model genes ----
    new_model = os.path.join(mdir, "rf_results.csv")                      # NEW layout
    old_model = os.path.join(mdir, f"rf_results_Synthetic_{label.upper()}.csv")  # OLD layout
    model_genes = _read_rf(new_model) if os.path.exists(new_model) else _read_rf(old_model)

    # ---- real genes ----
    new_real = os.path.join(cdir, "real", "rf_results.csv")  # NEW layout: shared real/ folder
    old_real = os.path.join(mdir, "rf_results_Real.csv")     # OLD layout: Real duplicated per pair
    real_genes = _read_rf(new_real) if os.path.exists(new_real) else _read_rf(old_real)

    return real_genes, model_genes


def read_real_once(dataset_folder):
    """Real gene list for the dataset (NEW real/ folder, else any pair's rf_results_Real.csv)."""
    cdir = _classifier_dir(dataset_folder)
    new_real = os.path.join(cdir, "real", "rf_results.csv")
    genes = _read_rf(new_real)
    if genes is not None:
        return genes
    for model in MODELS:  # fall back to whichever OLD-layout pair exists
        genes = _read_rf(os.path.join(cdir, model, "rf_results_Real.csv"))
        if genes is not None:
            return genes
    return None


def process_dataset(dataset_folder, disp):
    out_root = os.path.join(OUT_ROOT, disp)
    os.makedirs(out_root, exist_ok=True)
    print(f"\n=== {disp}  ({dataset_folder}) [base/100-epoch] ===")

    real_genes = read_real_once(dataset_folder)
    if real_genes is None:
        print(f"[skip] {disp}: no Real rf_results in either layout")
        return None

    organism = fg.ORGANISM_OVERRIDE.get(dataset_folder) or fg.detect_organism(real_genes)
    go_sets = fg.load_go_gene_sets(organism)
    background = fg.BACKGROUND_OVERRIDE or fg.MSIGDB[organism]["universe"]
    print(f"    organism: {organism}  (universe N={background})")

    real_go = fg.enrich_top_go(real_genes, go_sets, background)
    real_terms = real_go["go_term"].tolist()
    print(f"    Real: {len(real_terms)} significant GO groups (FDR q<{FDR_Q_THRESHOLD})")

    real_dir = os.path.join(out_root, "real")
    os.makedirs(real_dir, exist_ok=True)
    real_go.to_csv(os.path.join(real_dir, "go_results.csv"), index=False)

    overlap_counts: dict[str, int] = {}
    overlap_pcts: dict[str, float] = {}

    for model, label in MODELS.items():
        _, genes = read_base_genes(dataset_folder, model)
        if genes is None:
            print(f"[skip] {disp}/{model}: no rf_results in either layout")
            continue
        model_go = fg.enrich_top_go(genes, go_sets, background)
        terms = model_go["go_term"].tolist()

        shared = [t for t in terms if t in set(real_terms)]
        uniq_real = [t for t in real_terms if t not in set(terms)]
        uniq_model = [t for t in terms if t not in set(real_terms)]
        pct_real = fg.overlap_percent_of_real(terms, real_terms)
        pct_model = fg.overlap_percent_of_model(terms, real_terms)
        jac = fg.jaccard_percent(terms, real_terms)

        overlap_counts[label] = len(shared)
        overlap_pcts[label] = pct_real

        mdir = os.path.join(out_root, model)
        os.makedirs(mdir, exist_ok=True)
        model_go.to_csv(os.path.join(mdir, "go_results.csv"), index=False)
        fg.write_series(os.path.join(mdir, "overlapping_go.csv"), shared, "GO_Term")
        fg.write_series(os.path.join(mdir, "unique_to_Real.csv"), uniq_real, "GO_Term")
        fg.write_series(os.path.join(mdir, f"unique_to_Synthetic_{label.upper()}.csv"),
                        uniq_model, "GO_Term")

        pd.DataFrame({
            "Metric": [
                "Max GO analyzed", "FDR q-value threshold",
                "Real significant GO", f"Synthetic_{label.upper()} significant GO",
                "Overlapping GO", "Jaccard Index",
                "Overlap % (of Real)", f"Overlap % (of Synthetic_{label.upper()})",
            ],
            "Value": [
                TOP_N_GO, FDR_Q_THRESHOLD,
                len(real_terms), len(terms), len(shared),
                round(jac / 100.0, 4),
                f"{pct_real:.2f}%", f"{pct_model:.2f}%",
            ],
        }).to_csv(os.path.join(mdir, "overlap_summary.csv"), index=False)

        print(f"    {label:9s}: {len(shared)}/{len(real_terms)} of Real's sig GO "
              f"recovered ({pct_real:.1f}%)  [model sig GO: {len(terms)}]")

    if not overlap_counts:
        print(f"[skip] {disp}: no synthetic models produced results")
        return None

    best = max(overlap_counts, key=overlap_counts.get)
    summary = {
        "dataset_name": disp,
        "run": "base_100epoch",
        "timestamp": datetime.now().isoformat(),
        "organism": organism,
        "top_n_genes": TOP_N_GENES,
        "max_go_per_model": TOP_N_GO,
        "fdr_q_threshold": FDR_Q_THRESHOLD,
        "real_significant_go": len(real_terms),
        "msigdb_collections": fg.MSIGDB[organism]["collections"],
        "msigdb_version": fg.MSIGDB[organism]["version"],
        "universe_N": background,
        "go_overlap_counts": overlap_counts,
        "go_overlap_pct_of_real": {k: round(v, 2) for k, v in overlap_pcts.items()},
        "best_model": best,
        "best_model_count": overlap_counts[best],
    }
    with open(os.path.join(out_root, "shared_fgsea_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    pd.DataFrame({
        "Model": list(overlap_counts.keys()),
        "Overlapping_GO": list(overlap_counts.values()),
        "Real_Significant_GO": [len(real_terms)] * len(overlap_counts),
        "Overlap_Pct_of_Real": [round(overlap_pcts[m], 2) for m in overlap_counts],
    }).to_csv(os.path.join(out_root, "shared_fgsea_summary.csv"), index=False)

    _plot_dataset_bar(disp, overlap_pcts, out_root)
    return overlap_pcts


# --------------------------------------------------------------------------
# Plots (reuse the headline styling + colours, write into base_100epoch/)
# --------------------------------------------------------------------------
def _plot_dataset_bar(disp, overlap_pcts, out_root):
    fg._style()
    labels = [m for m in MODELS.values() if m in overlap_pcts]
    vals = [overlap_pcts[m] for m in labels]
    cols = [COLORS[m] for m in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, vals, color=cols, edgecolor="black", linewidth=0.6, width=0.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Real ontology groups recovered (%)")
    ax.set_title(f"{disp} (100-epoch): M5/C5 ontology overlap with Real\n"
                 f"(top-{TOP_N_GO} sets, FDR q<{FDR_Q_THRESHOLD}, top-{TOP_N_GENES} genes)",
                 fontsize=13, pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    fig.tight_layout()
    stable = os.path.join(out_root, "fgsea_overlap_comparison.png")
    fig.savefig(stable, bbox_inches="tight")
    fig.savefig(os.path.join(out_root, f"fgsea_overlap_comparison_{TIMESTAMP}.png"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"    plot -> {stable}")


def _plot_cross_dataset(results):
    fg._style()
    datasets = list(results.keys())
    labels = list(MODELS.values())
    n_ds, n_m = len(datasets), len(labels)
    x = np.arange(n_ds)
    width = 0.8 / n_m

    fig, ax = plt.subplots(figsize=(max(8, 1.7 * n_ds), 5.5))
    for i, label in enumerate(labels):
        vals = [results[d].get(label, 0.0) for d in datasets]
        offset = (i - (n_m - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=COLORS[label],
                      edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Real ontology groups recovered (%)")
    ax.set_title(f"Base / 100-epoch cross-dataset: M5/C5 ontology overlap with Real "
                 f"(top-{TOP_N_GO} sets, FDR q<{FDR_Q_THRESHOLD}, top-{TOP_N_GENES} genes)",
                 fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(title="Model", frameon=False)
    fig.tight_layout()
    stable = os.path.join(OUT_ROOT, "cross_dataset_fgsea_overlap.png")
    fig.savefig(stable, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_ROOT, f"cross_dataset_fgsea_overlap_{TIMESTAMP}.png"),
                bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(
        [[results[d].get(m, np.nan) for m in labels] for d in datasets],
        index=datasets, columns=labels,
    ).to_csv(os.path.join(OUT_ROOT, "cross_dataset_fgsea_overlap.csv"))
    print(f"\ncross-dataset plot -> {stable}")


def main():
    only = sys.argv[1:] or None
    os.makedirs(OUT_ROOT, exist_ok=True)
    results: dict[str, dict[str, float]] = {}
    for folder, disp in DATASETS.items():
        if only and folder not in only and disp not in only:
            continue
        res = process_dataset(folder, disp)
        if res:
            results[disp] = res
    if results:
        _plot_cross_dataset(results)
    print(f"\nDone. {len(results)} base dataset(s) processed. Output under {OUT_ROOT}/")


if __name__ == "__main__":
    main()
