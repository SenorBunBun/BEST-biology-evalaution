#!/usr/bin/env python3
"""All-encompassing augmentation trend across datasets vs dataset size.

Collapses the dose-response sweep into two per-model summaries plotted against
cell count (log scale), one point per dataset:
  (top)    mean delta_vs_dup over all inject fractions  -> "does synthetic beat
           duplicating real, on average?"
  (bottom) dose slope (change in delta per +1x inject)  -> "does MORE synthetic
           help or hurt as you add it?"
Each model gets an OLS fit vs log10(N) with its Pearson r annotated. Run with
.venv_pipeline."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/Users/ljeytl/Arsh/Research/NASA/AI4LS/BEST_bioeval")
OUTPUTS = REPO / "pipeline" / "outputs"
OUT = REPO / "cross_analysis_scripts" / "results" / "figures" / "augmentation_size_trend.png"

COLORS = {"CVAE": "#8E44AD", "Diffusion": "#E57373", "GAN": "#E69F00"}
MODELS = ["CVAE", "Diffusion", "GAN"]
FR = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
FRK = ["0.25", "0.5", "1.0", "1.5", "2.0"]
DATASETS = [
    ("OSDR-402", "OSDR-402-mmd-complete-pca", 1534),
    ("OSDR-403", "OSDR-403-mmd-complete-pca", 1561),
    ("OSDR-405", "OSDR-405-mmd-pca", 3384),
    ("OSDR-404", "OSDR-404-mmd-complete-pca", 14045),
    ("CRC", "CRC_IMM_10X-mmd-complete-pca", 190519),
    ("OSDR-352", "OSDR-352-612-613-mmd-complete-pca", 213854),
]


def series(folder, model):
    pf = json.load(open(OUTPUTS / folder / "analysis" / "classifier_augmented"
                        / "shared_classifier_augmented_summary.json"))["per_inject_frac"]
    return np.array([pf[k][model]["delta_vs_dup"] for k in FRK])


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 300})
    names = [d[0] for d in DATASETS]
    logN = np.log10([d[2] for d in DATASETS])
    # separate the two near-identical-N pairs (402/403, CRC/352) for legible labels
    offsets = np.array([-0.07, +0.07, 0.0, 0.0, -0.07, +0.07])
    xplot = logN + offsets
    means = {m: [] for m in MODELS}
    slopes = {m: [] for m in MODELS}
    for _, folder, _ in DATASETS:
        for m in MODELS:
            d = series(folder, m)
            means[m].append(d.mean())
            slopes[m].append(np.polyfit(FR, d, 1)[0])

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    xline = np.linspace(xplot.min() - 0.1, xplot.max() + 0.1, 50)

    def panel(ax, vals, ylabel, title):
        ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.6, zorder=1)
        for m in MODELS:
            y = np.array(vals[m])
            r = np.corrcoef(logN, y)[0, 1]              # r on true log N
            b, a = np.polyfit(logN, y, 1)
            ax.plot(xline, b * xline + a, color=COLORS[m], lw=2, alpha=0.5, zorder=2)
            ax.scatter(xplot, y, color=COLORS[m], s=70, edgecolor="black",
                       linewidth=0.6, zorder=3, label=f"{m}  (r={r:+.2f})")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="best", fontsize=11)

    panel(axT, means, "mean Δ balanced-acc\nvs duplicate-real",
          "Average augmentation benefit vs dataset size")
    panel(axB, slopes, "dose slope\n(Δ per +1× inject)",
          "Dose sensitivity vs dataset size  (positive = more synthetic helps)")

    axB.set_xticks(xplot)
    axB.set_xticklabels([f"{n}\nN={d[2]:,}" for n, d in zip(names, DATASETS)],
                        fontsize=9, rotation=30, ha="right")
    axB.set_xlabel("dataset size (cells, log scale) — small → large", labelpad=8)
    fig.suptitle("All-dataset augmentation trend: synthetic value shrinks and "
                 "turns negative as data grows", fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
