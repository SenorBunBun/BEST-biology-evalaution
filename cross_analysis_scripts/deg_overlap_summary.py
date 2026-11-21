#!/usr/bin/env python3
"""Cross-dataset DEG overlap summary (real vs each synthetic model).

Standalone — no pipeline edits, no args. Reads the per-model all-genes DEG tables
written by the pipeline:

    pipeline/outputs/<folder>/analysis/deg/<model>/deg_results_all.csv
    columns: names, scores, logfoldchanges, pvals, pvals_adj, abs_logfoldchanges

For every (dataset, model) it computes four top-50 overlap counts of the
synthetic gene list against the REAL gene list. Ranking matches the pipeline's
standard DEG ordering so these numbers are consistent with deg_results.csv and
the shared_deg_comparison / directional overlap outputs:

    sort by [pvals_adj asc, abs_logfoldchanges desc]

  Unsigned Overlap      top-50 genes (real) vs top-50 (synth); count shared
                        names. (Equivalent to the deg_results.csv overlap.)
  Signed Overlap        same two top-50 lists, but a shared gene counts ONLY if
                        sign(logFC) matches (up=up, down=down).
  Top50 Up Overlap      top-50 upregulated real (logFC>0) vs top-50 upregulated
                        synth; count shared names.
  Top50 Down Overlap    top-50 downregulated real (logFC<0) vs top-50
                        downregulated synth; count shared names.

Each count is out of 50 (or out of the available genes in that direction when a
model has fewer than 50 up/down genes — reported in the *_denom columns).

Output: cross_analysis_scripts/deg_overlap_summary.csv  (one row per
dataset x model). Run with .venv_pipeline (needs pandas)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/ljeytl/Arsh/Research/NASA/AI4LS/BEST_bioeval")
OUTPUTS = REPO / "pipeline" / "outputs"
OUT_CSV = REPO / "cross_analysis_scripts" / "deg_overlap_summary.csv"
TOP_N = 50
MODELS = ["cvae", "diffusion", "gan"]

# (display dataset name, local output folder)
DATASETS = [
    ("OSDR-402",         "OSDR-402-mmd-complete-pca"),
    ("OSDR-403",         "OSDR-403-mmd-complete-pca"),
    ("OSDR-404",         "OSDR-404-mmd-complete-pca"),
    ("OSDR-405",         "OSDR-405-mmd-pca"),
    ("OSDR-352-612-613", "OSDR-352-612-613-mmd-complete-pca"),
    ("CRC_IMM_10X",      "CRC_IMM_10X-mmd-complete-pca"),
]


def load_deg(folder, model):
    p = OUTPUTS / folder / "analysis" / "deg" / model / "deg_results_all.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # Ensure the magnitude column exists / is consistent (pure |logFC|).
    df["abs_logfoldchanges"] = df["logfoldchanges"].abs()
    return df


SORT_KEYS = ["pvals_adj", "abs_logfoldchanges"]
SORT_ASC = [True, False]


def top_unsigned(df, top_n=TOP_N):
    """Return a dict name -> logFC for the top-N genes by the pipeline ranking
    [pvals_adj asc, abs_logfoldchanges desc] (= the deg_results.csv top-N)."""
    t = df.sort_values(SORT_KEYS, ascending=SORT_ASC).head(top_n)
    return dict(zip(t["names"], t["logfoldchanges"]))


def top_direction(df, direction, top_n=TOP_N):
    """Top-N names for one direction, ranked by the pipeline key
    [pvals_adj asc, abs_logfoldchanges desc]."""
    if direction == "up":
        sub = df[df["logfoldchanges"] > 0]
    else:
        sub = df[df["logfoldchanges"] < 0]
    sub = sub.sort_values(SORT_KEYS, ascending=SORT_ASC)
    names = sub["names"].head(top_n).tolist()
    return names, len(sub)


def compute_row(dataset, model, real_df, model_df):
    # --- Unsigned & Signed (shared top-50-by-|logFC| lists) ---
    real_u = top_unsigned(real_df)
    model_u = top_unsigned(model_df)
    shared = set(real_u) & set(model_u)
    unsigned_overlap = len(shared)
    signed_overlap = sum(
        1 for g in shared if np.sign(real_u[g]) == np.sign(model_u[g]) and real_u[g] != 0
    )

    # --- Directional (separate top-50 per direction) ---
    real_up, real_up_n = top_direction(real_df, "up")
    model_up, model_up_n = top_direction(model_df, "up")
    up_overlap = len(set(real_up) & set(model_up))

    real_dn, real_dn_n = top_direction(real_df, "down")
    model_dn, model_dn_n = top_direction(model_df, "down")
    down_overlap = len(set(real_dn) & set(model_dn))

    return {
        "Dataset": dataset,
        "Model": model.upper(),
        "Unsigned_Overlap": unsigned_overlap,
        "Signed_Overlap": signed_overlap,
        "Top50_Up_Overlap": up_overlap,
        "Top50_Down_Overlap": down_overlap,
        # context / denominators (how many genes were actually available)
        "Up_denom_real": min(real_up_n, TOP_N),
        "Up_denom_model": min(model_up_n, TOP_N),
        "Down_denom_real": min(real_dn_n, TOP_N),
        "Down_denom_model": min(model_dn_n, TOP_N),
    }


def main():
    rows = []
    for dataset, folder in DATASETS:
        real_df = load_deg(folder, "real")
        if real_df is None:
            print(f"[WARN] missing real deg_results_all for {dataset}; skipping", flush=True)
            continue
        for model in MODELS:
            model_df = load_deg(folder, model)
            if model_df is None:
                print(f"[WARN] missing {model} deg_results_all for {dataset}; skipping", flush=True)
                continue
            row = compute_row(dataset, model, real_df, model_df)
            rows.append(row)
            print(
                f"  {dataset:18s} {model.upper():10s} "
                f"unsigned={row['Unsigned_Overlap']:2d}  signed={row['Signed_Overlap']:2d}  "
                f"up={row['Top50_Up_Overlap']:2d}  down={row['Top50_Down_Overlap']:2d}",
                flush=True,
            )

    if not rows:
        print("[ERROR] no rows computed — no deg_results_all.csv files found.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Wrote {len(df)} rows ({df['Dataset'].nunique()} datasets x "
          f"{df['Model'].nunique()} models) to:\n  {OUT_CSV}")


def self_test():
    """Tiny synthetic check of the overlap logic."""
    # real: up g1(+3) g2(+2); down g3(-3) g4(-2); plus filler
    # pvals_adj all 0 so the pipeline ranking reduces to |logFC| order here.
    real = pd.DataFrame({
        "names": ["g1", "g2", "g3", "g4", "g5"],
        "logfoldchanges": [3.0, 2.0, -3.0, -2.0, 0.1],
        "pvals_adj": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    # model: g1 flips sign (-3), g2 same (+2), g3 same (-3), g6 new
    model = pd.DataFrame({
        "names": ["g1", "g2", "g3", "g6", "g5"],
        "logfoldchanges": [-3.0, 2.0, -3.0, 2.5, 0.1],
        "pvals_adj": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    for d in (real, model):
        d["abs_logfoldchanges"] = d["logfoldchanges"].abs()
    r = compute_row("TEST", "cvae", real, model)
    # top-|logFC| real(50)=all5, model=all5 -> shared {g1,g2,g3,g5}=4 unsigned
    assert r["Unsigned_Overlap"] == 4, r
    # signed: g1 flips (no), g2 +/+ (yes), g3 -/- (yes), g5 +/+ (yes) -> 3
    assert r["Signed_Overlap"] == 3, r
    # up: real{g1,g2,g5}, model{g2,g6,g5} -> {g2,g5}=2
    assert r["Top50_Up_Overlap"] == 2, r
    # down: real{g3,g4}, model{g1,g3} -> {g3}=1
    assert r["Top50_Down_Overlap"] == 1, r
    print("self_test passed:", r)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        self_test()
    else:
        main()
