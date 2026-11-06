#!/usr/bin/env python3
"""
fgsea-style GO gene-set overlap analysis for synthetic vs real single-cell data.

For each dataset, each data source (real, CVAE, Diffusion, GAN) has a Random-Forest
ranked gene list in <dataset>/analysis/classifier/<model>/rf_results.csv. This script:

  1. takes the top-N RF genes per model,
  2. runs GO over-representation against the MSigDB mouse C5 GO collections
     (GO:BP + GO:CC + GO:MF, via gseapy  ->  matches MSigDB `compute_overlaps`),
  3. keeps the top-10 enriched GO groups per model,
  4. computes, for each synthetic model, the overlap of its top-10 GO groups with
     Real's top-10 (overlap is ONLY ever computed against Real),
  5. writes per-model results mirroring the classifier/ folder layout,
  6. draws a per-dataset bar graph (overlap % vs Real, y-axis fixed 0-100),
  7. draws a cross-dataset grouped bar graph across all datasets.

Everything is driven by the CONFIG block below -- edit that, not the body.

Run:
    /Users/ljeytl/Arsh/Research/NASA/.venv_fgsea/bin/python fgsea_go_overlap.py
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

# ==========================================================================
# CONFIG  -- edit here only
# ==========================================================================

GENERIC_OUTPUTS_DIR = "/Volumes/Ljeytl's SSD/NASAResearch/AI4LS/pipeline/outputs"
NASA_DIR = "/Users/ljeytl/Arsh/Research/NASA/fgsea"

# folder name on disk  ->  publication display name
DATASETS = {
    "CRC_IMM_10X-mmd-complete-pca":      "CRC_IMM_10X",
    "OSDR-352-612-613-mmd-complete-pca": "OSDR-352-612-613",
    "OSDR-402-mmd-complete-pca":         "OSDR-402",
    "OSDR-403-mmd-complete-pca":         "OSDR-403",
    "OSDR-404-mmd-complete-pca":         "OSDR-404",
    "OSDR-405-mmd-pca":                  "OSDR-405",
}

# model folder (lowercase, as on disk)  ->  publication label
MODELS = {
    "cvae":      "CVAE",
    "diffusion": "Diffusion",
    "gan":       "GAN",
}

# publication label  ->  colour (unified with shared_deg_comparison.py)
COLORS = {
    "CVAE":      "#8E44AD",  # Purple
    "Diffusion": "#E57373",  # Light coral/red
    "GAN":       "#E69F00",  # Orange
}

# RF genes fed into enrichment per model. Override at runtime without editing the
# file:  FGSEA_TOP_N_GENES=25 python fgsea_go_overlap.py
# When != 50 the output folder + plots get a "_top<N>genes" suffix so runs don't clash.
TOP_N_GENES = int(os.environ.get("FGSEA_TOP_N_GENES", "50"))
# Max ontology sets kept per model (after the FDR filter). Override at runtime:
#   FGSEA_TOP_N_GO=25 python fgsea_go_overlap.py
TOP_N_GO = int(os.environ.get("FGSEA_TOP_N_GO", "10"))
FDR_Q_THRESHOLD = 0.05    # keep only GO groups with FDR q-value (BH adj. p) < this
# Gene universe (N) for the hypergeometric test is taken per-organism from MSIGDB[...]["universe"]
# below. Set FGSEA_BACKGROUND to force one universe for all organisms (overrides the dict).
BACKGROUND_OVERRIDE = int(os.environ["FGSEA_BACKGROUND"]) if os.environ.get("FGSEA_BACKGROUND") else None

# MSigDB ontology collection per organism: the FULL M5 set for mouse (GO:BP + GO:CC +
# GO:MF + Mammalian Phenotype Ontology), and its MSigDB equivalent C5 for human
# (GO:BP + GO:CC + GO:MF + Human Phenotype Ontology). `*.all` is the whole collection.
# Organism is auto-detected per dataset from gene-symbol casing (mouse = title-case
# e.g. Zmpste24; human = all-caps e.g. NKG7). Both keep results MSigDB-aligned.
# Per organism: MSigDB release, ontology collection, and the gene universe size N used
# for the hypergeometric test (copied from the MSigDB compute_overlaps stat line so results
# line up exactly). "universe" is MSigDB's "# Genes in Universe (N)".
MSIGDB = {
    "mouse": {"version": "2026.2.Mm", "collections": ["m5.all"], "universe": 46339},  # 17,636 M5 sets
    "human": {"version": "2026.2.Hs", "collections": ["c5.all"], "universe": 43841},  # 15,849 C5 sets
}
# Optional: force an organism for specific dataset folders, e.g. {"CRC_IMM_10X-mmd-complete-pca": "human"}
ORGANISM_OVERRIDE: dict[str, str] = {}

PLOT_DPI = 300

# ==========================================================================
# end CONFIG
# ==========================================================================

CACHE_DIR = os.path.join(NASA_DIR, "_cache")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Suffix that keeps non-default runs (gene count and/or ontology-set cap) in their
# own folders/filenames so they never overwrite the headline top-50-genes / top-10-sets run.
RUN_SUFFIX = ("" if TOP_N_GENES == 50 else f"_top{TOP_N_GENES}genes") + \
             ("" if TOP_N_GO == 10 else f"_{TOP_N_GO}sets")
OVERLAP_DIRNAME = "fgsea_overlap" + RUN_SUFFIX


# --------------------------------------------------------------------------
# GO gene-set loading (MSigDB mouse C5 GO, cached locally after first fetch)
# --------------------------------------------------------------------------
_GO_CACHE: dict[str, dict] = {}  # organism -> merged GO sets (in-memory, per run)


def detect_organism(genes):
    """Mouse vs human from symbol casing: human symbols are all-caps (NKG7),
    mouse symbols are title-case (Zmpste24, mt-Nd3)."""
    alpha = [g for g in genes if any(c.isalpha() for c in g)]
    if not alpha:
        return "mouse"
    all_caps = sum(1 for g in alpha if g.upper() == g)
    return "human" if all_caps / len(alpha) > 0.6 else "mouse"


def load_go_gene_sets(organism):
    """Return a merged {GO_term: [genes]} dict for `organism`'s GO collections.

    Cached in memory per run and to NASA_DIR/_cache/go_gmt_<version>_<cols>.json
    so reruns are offline and fast.
    """
    if organism in _GO_CACHE:
        return _GO_CACHE[organism]

    spec = MSIGDB[organism]
    version, collections = spec["version"], spec["collections"]
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = version + "_" + "-".join(c.split(".")[-1] for c in collections)
    cache_path = os.path.join(CACHE_DIR, f"go_gmt_{tag}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            merged = json.load(fh)
        print(f"[go] ({organism}) loaded {len(merged)} GO sets from cache")
        _GO_CACHE[organism] = merged
        return merged

    from gseapy import Msigdb

    print(f"[go] ({organism}) fetching MSigDB {version} GO: {collections}")
    msig = Msigdb()
    merged: dict[str, list[str]] = {}
    for cat in collections:
        gmt = msig.get_gmt(category=cat, dbver=version)
        if not gmt:
            print(f"[go]   WARNING: {cat} returned no sets")
            continue
        merged.update(gmt)
        print(f"[go]   {cat}: {len(gmt)} sets")

    if not merged:
        sys.exit(f"[go] ERROR: no GO gene sets fetched from MSigDB for {organism}.")

    with open(cache_path, "w") as fh:
        json.dump(merged, fh)
    print(f"[go] ({organism}) merged {len(merged)} GO sets -> cached")
    _GO_CACHE[organism] = merged
    return merged


# --------------------------------------------------------------------------
# RF gene list per model
# --------------------------------------------------------------------------
def read_top_genes(dataset_folder, model):
    """Top-TOP_N_GENES gene symbols (by RF rank) for one model, or None if missing."""
    path = os.path.join(
        GENERIC_OUTPUTS_DIR, dataset_folder, "analysis", "classifier", model, "rf_results.csv"
    )
    if not os.path.exists(path):
        print(f"[warn] missing rf_results: {path}")
        return None
    df = pd.read_csv(path)
    if "rank" in df.columns:
        df = df.sort_values("rank")
    elif "mean_importance" in df.columns:
        df = df.sort_values("mean_importance", ascending=False)
    return df["gene"].astype(str).head(TOP_N_GENES).tolist()


def display_name(dataset_folder, default):
    """Prefer the canonical name stored by the classifier stage; fall back to the dict value."""
    summ = os.path.join(
        GENERIC_OUTPUTS_DIR, dataset_folder, "analysis", "classifier",
        "shared_classifier_summary.json",
    )
    if os.path.exists(summ):
        try:
            with open(summ) as fh:
                name = json.load(fh).get("dataset_name")
            if name:
                return name
        except Exception:
            pass
    return default


# --------------------------------------------------------------------------
# GO enrichment -> top-N GO terms for a gene list
# --------------------------------------------------------------------------
def enrich_top_go(gene_list, go_sets, background):
    """Return a DataFrame of the significant top GO terms for `gene_list`.

    Hypergeometric over-representation via gseapy.enrich (same statistic as the
    MSigDB compute_overlaps tool). Keeps only GO groups with FDR q-value
    (BH-adjusted p-value) < FDR_Q_THRESHOLD, ranked by q-value, capped at TOP_N_GO.
    May return fewer than TOP_N_GO rows (including zero) when few/none are significant.
    Columns: go_term, adj_pvalue, pvalue, overlap_genes, rank.
    """
    import gseapy as gp

    empty = pd.DataFrame(columns=["go_term", "adj_pvalue", "pvalue", "overlap_genes", "rank"])
    try:
        enr = gp.enrich(
            gene_list=list(gene_list),
            gene_sets=go_sets,
            background=background,
            outdir=None,
            verbose=False,
        )
    except Exception as e:
        print(f"[warn] enrichment failed: {e}")
        return empty

    res = getattr(enr, "results", None)
    if res is None or not isinstance(res, pd.DataFrame) or res.empty:
        return empty
    res = res.copy()

    res = res.sort_values(["Adjusted P-value", "P-value"])
    res = res[res["Adjusted P-value"].astype(float) < FDR_Q_THRESHOLD]
    res = res.head(TOP_N_GO).reset_index(drop=True)
    if res.empty:
        return empty
    out = pd.DataFrame({
        "go_term": res["Term"].astype(str),
        "adj_pvalue": res["Adjusted P-value"].astype(float),
        "pvalue": res["P-value"].astype(float),
        "overlap_genes": res["Genes"].astype(str),
        "rank": np.arange(1, len(res) + 1),
    })
    return out


# --------------------------------------------------------------------------
# Overlap metric (vs Real). Swap this one function for a different definition.
# --------------------------------------------------------------------------
def overlap_percent_of_real(model_terms, real_terms):
    """% of Real's significant GO groups that the model also recovers (0-100).

    This is the 'overlap with Real' value plotted on the bar graphs. Denominator is
    Real's set, so it reads as 'how much of Real's significant GO biology the model
    reproduces'. Returns 0 when Real has no significant groups.
    """
    if not real_terms:
        return 0.0
    shared = set(model_terms) & set(real_terms)
    return 100.0 * len(shared) / len(real_terms)


def overlap_percent_of_model(model_terms, real_terms):
    """% of the model's significant GO groups that are also in Real's (0-100)."""
    if not model_terms:
        return 0.0
    shared = set(model_terms) & set(real_terms)
    return 100.0 * len(shared) / len(model_terms)


def jaccard_percent(model_terms, real_terms):
    """Alternative metric: Jaccard index of the two top-N GO sets, scaled to %."""
    a, b = set(model_terms), set(real_terms)
    union = a | b
    if not union:
        return 0.0
    return 100.0 * len(a & b) / len(union)


# --------------------------------------------------------------------------
# Writers (mirror the classifier/ folder format)
# --------------------------------------------------------------------------
def write_series(path, values, header):
    pd.DataFrame({header: list(values)}).to_csv(path, index=False)


def process_dataset(dataset_folder, disp_default):
    """Run the full GO-overlap analysis for one dataset. Returns per-model overlap %."""
    disp = display_name(dataset_folder, disp_default)
    out_root = os.path.join(NASA_DIR, disp, OVERLAP_DIRNAME)
    os.makedirs(out_root, exist_ok=True)
    print(f"\n=== {disp}  ({dataset_folder}) ===")

    # ---- enrich Real first (the only reference) ----
    real_genes = read_top_genes(dataset_folder, "real")
    if real_genes is None:
        print(f"[skip] {disp}: no real rf_results")
        return None

    organism = ORGANISM_OVERRIDE.get(dataset_folder) or detect_organism(real_genes)
    go_sets = load_go_gene_sets(organism)
    background = BACKGROUND_OVERRIDE or MSIGDB[organism]["universe"]
    print(f"    organism: {organism}  (universe N={background})")

    real_go = enrich_top_go(real_genes, go_sets, background)
    real_terms = real_go["go_term"].tolist()
    print(f"    Real: {len(real_terms)} significant GO groups (FDR q<{FDR_Q_THRESHOLD})")
    if not real_terms:
        print(f"[note] {disp}: Real has no GO groups below FDR q<{FDR_Q_THRESHOLD}; "
              f"all overlaps will be 0%")

    real_dir = os.path.join(out_root, "real")
    os.makedirs(real_dir, exist_ok=True)
    real_go.to_csv(os.path.join(real_dir, "go_results.csv"), index=False)

    # ---- each synthetic model vs Real ----
    overlap_counts: dict[str, int] = {}
    overlap_pcts: dict[str, float] = {}

    for model, label in MODELS.items():
        genes = read_top_genes(dataset_folder, model)
        if genes is None:
            print(f"[skip] {disp}/{model}: no rf_results")
            continue
        model_go = enrich_top_go(genes, go_sets, background)
        terms = model_go["go_term"].tolist()

        shared = [t for t in terms if t in set(real_terms)]
        uniq_real = [t for t in real_terms if t not in set(terms)]
        uniq_model = [t for t in terms if t not in set(real_terms)]
        pct_real = overlap_percent_of_real(terms, real_terms)   # plotted: fraction of Real recovered
        pct_model = overlap_percent_of_model(terms, real_terms)
        jac = jaccard_percent(terms, real_terms)

        overlap_counts[label] = len(shared)
        overlap_pcts[label] = pct_real

        mdir = os.path.join(out_root, model)
        os.makedirs(mdir, exist_ok=True)
        model_go.to_csv(os.path.join(mdir, "go_results.csv"), index=False)
        write_series(os.path.join(mdir, "overlapping_go.csv"), shared, "GO_Term")
        write_series(os.path.join(mdir, "unique_to_Real.csv"), uniq_real, "GO_Term")
        write_series(os.path.join(mdir, f"unique_to_Synthetic_{label.upper()}.csv"),
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
                f"{pct_real:.2f}%",
                f"{pct_model:.2f}%",
            ],
        }).to_csv(os.path.join(mdir, "overlap_summary.csv"), index=False)

        print(f"    {label:9s}: {len(shared)}/{len(real_terms)} of Real's sig GO "
              f"recovered ({pct_real:.1f}%)  [model sig GO: {len(terms)}]")

    if not overlap_counts:
        print(f"[skip] {disp}: no synthetic models produced results")
        return None

    # ---- shared summary (json + csv), mirroring classifier ----
    best = max(overlap_counts, key=overlap_counts.get)
    summary = {
        "dataset_name": disp,
        "timestamp": datetime.now().isoformat(),
        "organism": organism,
        "top_n_genes": TOP_N_GENES,
        "max_go_per_model": TOP_N_GO,
        "fdr_q_threshold": FDR_Q_THRESHOLD,
        "real_significant_go": len(real_terms),
        "msigdb_collections": MSIGDB[organism]["collections"],
        "msigdb_version": MSIGDB[organism]["version"],
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

    plot_dataset_bar(disp, overlap_pcts, out_root)
    return overlap_pcts


# --------------------------------------------------------------------------
# Plots (publication styling)
# --------------------------------------------------------------------------
def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": PLOT_DPI,
    })


def plot_dataset_bar(disp, overlap_pcts, out_root):
    _style()
    labels = [m for m in MODELS.values() if m in overlap_pcts]
    vals = [overlap_pcts[m] for m in labels]
    cols = [COLORS[m] for m in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, vals, color=cols, edgecolor="black", linewidth=0.6, width=0.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Real ontology groups recovered (%)")
    ax.set_title(f"{disp}: M5/C5 ontology overlap with Real\n"
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


def plot_cross_dataset(results):
    """results: {display_name: {model_label: pct}} in dataset order."""
    _style()
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
    ax.set_title(f"Cross-dataset: M5/C5 ontology overlap with Real "
                 f"(top-{TOP_N_GO} sets, FDR q<{FDR_Q_THRESHOLD}, top-{TOP_N_GENES} genes)",
                 fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(title="Model", frameon=False)
    fig.tight_layout()

    stable = os.path.join(NASA_DIR, f"cross_dataset_fgsea_overlap{RUN_SUFFIX}.png")
    fig.savefig(stable, bbox_inches="tight")
    fig.savefig(os.path.join(NASA_DIR, f"cross_dataset_fgsea_overlap{RUN_SUFFIX}_{TIMESTAMP}.png"),
                bbox_inches="tight")
    plt.close(fig)

    # matrix CSV (datasets x models)
    pd.DataFrame(
        [[results[d].get(m, np.nan) for m in labels] for d in datasets],
        index=datasets, columns=labels,
    ).to_csv(os.path.join(NASA_DIR, f"cross_dataset_fgsea_overlap{RUN_SUFFIX}.csv"))
    print(f"\ncross-dataset plot -> {stable}")


# --------------------------------------------------------------------------
def main():
    only = sys.argv[1:] or None  # optional: pass dataset folder name(s) to limit the run

    results: dict[str, dict[str, float]] = {}
    for folder, disp in DATASETS.items():
        if only and folder not in only and disp not in only:
            continue
        res = process_dataset(folder, disp)
        if res:
            results[display_name(folder, disp)] = res

    if len(results) >= 1:
        plot_cross_dataset(results)
    print(f"\nDone. {len(results)} dataset(s) processed. Output under {NASA_DIR}/")


if __name__ == "__main__":
    main()
