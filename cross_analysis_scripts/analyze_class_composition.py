#!/usr/bin/env python3
"""
Class-composition of the REAL single-cell datasets -> one CSV.

Reports the two classes the RF phenotype classifier actually distinguishes in each
real dataset, with per-class percentages:

  * Spaceflight datasets (OSDR): obs["Flight"]  ->  Ground Control vs Spaceflight
  * Cancer dataset (CRC_IMM_10X): obs["tissue"] ->  Metastasis Normal vs Metastasis Tumor
    (the cancer task uses only the two metastasis groups; PBMC / lymph node cells are in
     the file but are NOT classifier classes, so percentages are taken over the two
     metastasis classes only.)

Only obs/ metadata is read (via h5py), never the expression matrix, so the 4.4 GB pooled
file is handled in a few MB of memory.

Key output:
  class_composition.csv  -- one row per dataset:
      dataset, task, label_column,
      class_1, class_1_pct, class_1_n,
      class_2, class_2_pct, class_2_n,
      classifier_cells

Run:
    /Users/ljeytl/Arsh/Research/NASA/fgsea/.venv_fgsea/bin/python analyze_class_composition.py
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pandas as pd

# ==========================================================================
# CONFIG  -- edit here only
# ==========================================================================

REAL_DATASETS_DIR = "/Volumes/Windows/Creations/Research/NASASLSTP25/AI4LS/AI4LS/CVAE/pipeline/real_datasets"
OUT_DIR = "/Users/ljeytl/Arsh/Research/NASA/analysis"
OUT_CSV = os.path.join(OUT_DIR, "class_composition.csv")

# filename -> (display name, label kind). "flight" = spaceflight task, "tissue" = cancer task.
# CRC_IMM_10X appears once: cancer_0.997_filtered_hvg_2000_norm.h5ad is the CRC_IMM_10X data
# after the pipeline's 0.997 filter + 2000-HVG + normalization -- the exact file the cancer RF
# classifier reads (REAL_PATH in analysis/rf_classifier_analysis_run_cancer.py). The separate
# CRC_IMM_10X_SS2_..._processed.h5ad is the earlier processed export of the SAME 190,519 cells,
# so it is not listed again.
DATASETS = {
    "osd_402_preprocessed.h5ad":                               ("OSDR-402",         "flight"),
    "osd_403_preprocessed.h5ad":                               ("OSDR-403",         "flight"),
    "osd_404_preprocessed.h5ad":                               ("OSDR-404",         "flight"),
    "osd_405_preprocessed.h5ad":                               ("OSDR-405",         "flight"),
    "osdr_352_612_613_raw_0.997_filtered_hvg_2000_norm.h5ad":  ("OSDR-352-612-613", "flight"),
    "cancer_0.997_filtered_hvg_2000_norm.h5ad":                ("CRC_IMM_10X",      "tissue"),
}

# The two classifier classes per task, in (class_1, class_2) order.
FLIGHT_CLASSES = ("Ground Control", "Spaceflight")
TISSUE_CLASSES = ("metastasis normal", "metastasis tumor")

# Flight raw value -> class label (handles float 0.0/1.0 and categorical Ground/Flight).
FLIGHT_MAP = {
    "0.0": "Ground Control", "0": "Ground Control", "ground": "Ground Control", "control": "Ground Control",
    "1.0": "Spaceflight",    "1": "Spaceflight",    "flight": "Spaceflight",    "spaceflight": "Spaceflight",
}

# ==========================================================================
# end CONFIG
# ==========================================================================


def _read_obs_column(obs, key):
    """numpy array of values for an obs column (categorical or plain)."""
    node = obs[key]
    if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
        cats = [c.decode() if isinstance(c, bytes) else c for c in node["categories"][:]]
        codes = node["codes"][:]
        return np.array([cats[c] if c >= 0 else None for c in codes], dtype=object)
    vals = node[:]
    if vals.dtype.kind == "S" or vals.dtype == object:
        return np.array([v.decode() if isinstance(v, bytes) else v for v in vals], dtype=object)
    return vals


def _norm_flight(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    key = str(v).strip()
    return FLIGHT_MAP.get(key.lower(), FLIGHT_MAP.get(key))


def _counts(path, kind):
    """Return {class_label: count} over the two classifier classes for one dataset."""
    classes = FLIGHT_CLASSES if kind == "flight" else TISSUE_CLASSES
    col = "Flight" if kind == "flight" else "tissue"
    with h5py.File(path, "r") as h:
        vals = _read_obs_column(h["obs"], col)
    if kind == "flight":
        vals = [_norm_flight(v) for v in vals]
    else:
        vals = [str(v).strip() if v is not None else None for v in vals]
    counts = {c: 0 for c in classes}
    for v in vals:
        if v in counts:
            counts[v] += 1
    return counts, classes, col


def analyze():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for fn, (disp, kind) in DATASETS.items():
        path = os.path.join(REAL_DATASETS_DIR, fn)
        if not os.path.exists(path):
            print(f"[warn] missing: {path}")
            continue
        counts, classes, col = _counts(path, kind)
        c1, c2 = classes
        n1, n2 = counts[c1], counts[c2]
        total = n1 + n2
        task = "Spaceflight vs Ground Control" if kind == "flight" else "Metastasis Tumor vs Normal"
        rows.append({
            "dataset": disp,
            "task": task,
            "label_column": col,
            "class_1": c1,
            "class_1_pct": round(100.0 * n1 / total, 2) if total else 0.0,
            "class_1_n": n1,
            "class_2": c2,
            "class_2_pct": round(100.0 * n2 / total, 2) if total else 0.0,
            "class_2_n": n2,
            "classifier_cells": total,
        })
        print(f"{disp:18s} {c1}={rows[-1]['class_1_pct']}% ({n1:,})  "
              f"{c2}={rows[-1]['class_2_pct']}% ({n2:,})")

    df = pd.DataFrame(rows, columns=[
        "dataset", "task", "label_column",
        "class_1", "class_1_pct", "class_1_n",
        "class_2", "class_2_pct", "class_2_n", "classifier_cells",
    ])
    df.to_csv(OUT_CSV, index=False)
    print(f"\n-> {OUT_CSV}")


if __name__ == "__main__":
    analyze()
