#!/usr/bin/env python3
"""Augmentation dose-response trend across dataset size, all 3 models.

Reads each dataset's classifier_augmented summary and plots delta_vs_dup
(synthetic minus duplicate-real, balanced accuracy) against inject fraction,
one panel per dataset ordered by cell count (small -> large), so the sign-flip
with dataset size is visible. Shaded bands = +/-1 std across resamples.
Model colours match the paper. Run with .venv_pipeline."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/Users/ljeytl/Arsh/Research/NASA/AI4LS/BEST_bioeval")
OUTPUTS = REPO / "pipeline" / "outputs"
OUT = REPO / "cross_analysis_scripts" / "results" / "figures" / "augmentation_dose_trend.png"

COLORS = {"CVAE": "#8E44AD", "Diffusion": "#E57373", "GAN": "#E69F00"}
MODELS = ["CVAE", "Diffusion", "GAN"]
FR = [0.25, 0.5, 1.0, 1.5, 2.0]
FRK = ["0.25", "0.5", "1.0", "1.5", "2.0"]

# ordered by N cells (small -> large)
DATASETS = [
    ("OSDR-402",         "OSDR-402-mmd-complete-pca",         1534),
    ("OSDR-403",         "OSDR-403-mmd-complete-pca",         1561),
    ("OSDR-405",         "OSDR-405-mmd-pca",                  3384),
    ("OSDR-404",         "OSDR-404-mmd-complete-pca",        14045),
    ("CRC IMM 10X",      "CRC_IMM_10X-mmd-complete-pca",    190519),
    ("OSDR-352/612/613", "OSDR-352-612-613-mmd-complete-pca", 213854),
]


def load(folder):
    pf = json.load(open(OUTPUTS / folder / "analysis" / "classifier_augmented"
                        / "shared_classifier_augmented_summary.json"))["per_inject_frac"]
    out = {}
    for m in MODELS:
        out[m] = (np.array([pf[k][m]["delta_vs_dup"] for k in FRK]),
                  np.array([pf[k][m].get("delta_vs_dup_std", 0.0) for k in FRK]))
    return out


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 300})
    data = [(name, load(folder), n) for name, folder, n in DATASETS]
    ymax = max(np.max(d[m][0] + d[m][1]) for _, d, _ in data for m in MODELS)
    ymin = min(np.min(d[m][0] - d[m][1]) for _, d, _ in data for m in MODELS)
    pad = 0.01
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for ax, (name, d, n) in zip(axes.ravel(), data):
        ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.6, zorder=1)
        for m in MODELS:
            mean, std = d[m]
            ax.plot(FR, mean, "-o", color=COLORS[m], lw=2, ms=5, label=m, zorder=3)
            ax.fill_between(FR, mean - std, mean + std, color=COLORS[m], alpha=0.12, zorder=2)
        ax.set_title(f"{name}  (N={n:,})", fontsize=12, fontweight="bold")
        ax.set_xticks(FR)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylim(ymin - pad, ymax + pad)
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ balanced-acc\nvs duplicate-real")
    for ax in axes[1, :]:
        ax.set_xlabel("inject fraction (× val size)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Model", frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Synthetic augmentation dose-response by dataset size "
                 "(panels ordered small → large)\n"
                 "above 0 = synthetic beats duplicating real; below 0 = synthetic hurts",
                 fontsize=15, y=1.00)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
