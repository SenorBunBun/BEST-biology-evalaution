#!/usr/bin/env python3
"""Paper-ready figures, matching the cross-dataset paper figure styling exactly:
same fonts, colours, gridlines, value labels, legend, DPI, AND the paper layout
where the out-of-domain cancer control (CRC) sits all the way on the right in a
shaded column, with bracket labels separating in-domain spaceflight from the
out-of-domain cancer control.

Produces three PNGs into cross_analysis_scripts/results/figures/:

  1. cross_dataset_GO_overlap.png
       GO graph regenerated from fgsea/cross_dataset_fgsea_overlap.csv
       (grouped bars, all 3 models).
  2. gan_deg_overlap_signed.png
       GAN only. Top-50 DEG overlap with Real (unsigned) vs sign-matched subset
       (up=up / down=down). Equal bars => direction preserved. Unit = % of 50.
  3. gan_deg_overlap_updown.png
       GAN only. Top-50 upregulated overlap (blue) vs downregulated (red).
       Unit = % of 50.

GAN overlaps use the pipeline ranking [pvals_adj asc, |logFC| desc] on the
all-genes DEG tables, so they match deg_overlap_summary.csv. Run with
.venv_pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path("/Users/ljeytl/Arsh/Research/NASA/AI4LS/BEST_bioeval")
OUTPUTS = REPO / "pipeline" / "outputs"
FGSEA_DIR = REPO / "fgsea"           # balanced GO = <dataset>-newclassifier summaries
FIG_DIR = REPO / "cross_analysis_scripts" / "results" / "figures"
TOP_N = 50
PLOT_DPI = 300

# --- unified styling (from fgsea/fgsea_go_overlap.py & cross_dataset_* scripts) ---
COLORS = {"CVAE": "#8E44AD", "Diffusion": "#E57373", "GAN": "#E69F00"}
UP_COLOR, DOWN_COLOR = "#2E6FB7", "#C0392B"       # up=blue, down=red
UNSIGNED_COLOR, SIGNED_COLOR = "#95A5A6", "#E69F00"
SHADE_COLOR = "#ECECEC"
SORT_KEYS, SORT_ASC = ["pvals_adj", "abs_logfoldchanges"], [True, False]

# Paper order: in-domain spaceflight first, out-of-domain cancer control LAST.
# (display name, local DEG output folder, balanced-GO fgsea folder = -newclassifier)
DATASETS = [
    ("OSDR-352/612/613", "OSDR-352-612-613-mmd-complete-pca", "OSDR-352-612-613-newclassifier"),
    ("OSDR-404",         "OSDR-404-mmd-complete-pca",         "OSDR-404-newclassifier"),
    ("OSDR-402",         "OSDR-402-mmd-complete-pca",         "OSDR-402-newclassifier"),
    ("OSDR-403",         "OSDR-403-mmd-complete-pca",         "OSDR-403-newclassifier"),
    ("OSDR-405",         "OSDR-405-mmd-pca",                  "OSDR-405-newclassifier"),
    ("CRC IMM 10X",      "CRC_IMM_10X-mmd-complete-pca",      "CRC_IMM_10X-newclassifier"),
]
DISPLAY = [d[0] for d in DATASETS]
OOD_INDEX = len(DATASETS) - 1          # CRC = out-of-domain control (rightmost)


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": PLOT_DPI,
    })


# ------------------------------------------------------------------ data
def load_deg(folder, model):
    p = OUTPUTS / folder / "analysis" / "deg" / model / "deg_results_all.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["abs_logfoldchanges"] = d["logfoldchanges"].abs()
    return d


def top_map(df, top_n=TOP_N):
    t = df.sort_values(SORT_KEYS, ascending=SORT_ASC).head(top_n)
    return dict(zip(t["names"], t["logfoldchanges"]))


def top_dir(df, direction, top_n=TOP_N):
    sub = df[df["logfoldchanges"] > 0] if direction == "up" else df[df["logfoldchanges"] < 0]
    sub = sub.sort_values(SORT_KEYS, ascending=SORT_ASC)
    return set(sub["names"].head(top_n))


def gan_overlaps_pct():
    """Return list (aligned with DATASETS) of {'unsigned','signed','up','down'} % for GAN."""
    rows = []
    for disp, folder, _ in DATASETS:
        real, gan = load_deg(folder, "real"), load_deg(folder, "gan")
        if real is None or gan is None:
            rows.append(None); continue
        ru, gu = top_map(real), top_map(gan)
        shared = set(ru) & set(gu)
        unsigned = len(shared)
        signed = sum(1 for g in shared if np.sign(ru[g]) == np.sign(gu[g]) and ru[g] != 0)
        up = len(top_dir(real, "up") & top_dir(gan, "up"))
        down = len(top_dir(real, "down") & top_dir(gan, "down"))
        rows.append({k: 100.0 * v / TOP_N for k, v in
                     dict(unsigned=unsigned, signed=signed, up=up, down=down).items()})
    return rows


def gan_signed_composition_pct():
    """Per dataset: up/down composition (% of 50) of the unsigned overlap (by real
    direction) and of the sign-matched overlap (concordant up/down)."""
    rows = []
    for disp, folder, _ in DATASETS:
        real, gan = load_deg(folder, "real"), load_deg(folder, "gan")
        if real is None or gan is None:
            rows.append(None); continue
        ru, gu = top_map(real), top_map(gan)
        shared = set(ru) & set(gu)
        c = dict(
            uns_up=sum(1 for g in shared if ru[g] > 0),
            uns_down=sum(1 for g in shared if ru[g] < 0),
            sgn_up=sum(1 for g in shared if ru[g] > 0 and gu[g] > 0),
            sgn_down=sum(1 for g in shared if ru[g] < 0 and gu[g] < 0),
        )
        rows.append({k: 100.0 * v / TOP_N for k, v in c.items()})
    return rows


# ------------------------------------------------------------------ paper layout bits
def _ood_decorations(ax, n_ds):
    """Shade the out-of-domain column and draw the two bracket labels, paper-style."""
    # shaded column behind the CRC group
    ax.axvspan(OOD_INDEX - 0.5, OOD_INDEX + 0.5, color=SHADE_COLOR, zorder=0)
    ax.set_axisbelow(True)
    trans = ax.get_xaxis_transform()   # x in data coords, y in axes fraction
    yb, tick = -0.30, 0.025            # bracket baseline & tick height (below x labels)

    def bracket(x0, x1, text):
        ax.plot([x0, x0, x1, x1], [yb + tick, yb, yb, yb + tick],
                transform=trans, color="black", lw=1.2, clip_on=False)
        ax.text((x0 + x1) / 2, yb - 0.035, text, transform=trans,
                ha="center", va="top", fontsize=11, fontweight="bold", clip_on=False)

    bracket(-0.4, OOD_INDEX - 1 + 0.4, "In-domain spaceflight")
    bracket(OOD_INDEX - 0.4, OOD_INDEX + 0.4, "Out-of-domain\ncancer control")


def _grouped(series, title, ylabel, legend_title, fname, ymax=100):
    """series: list of (label, color, [values per dataset]) aligned with DATASETS."""
    _style()
    n_ds, n_s = len(DISPLAY), len(series)
    x = np.arange(n_ds)
    width = 0.8 / n_s
    fig, ax = plt.subplots(figsize=(max(9, 1.9 * n_ds), 6.0))
    for i, (label, color, vals) in enumerate(series):
        offset = (i - (n_s - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color,
                      edgecolor="black", linewidth=0.5, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8, zorder=4)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY, rotation=30, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title=legend_title, frameon=False, loc="upper center")
    _ood_decorations(ax, n_ds)
    fig.subplots_adjust(bottom=0.30)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / fname
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  plot -> {out}")


# ------------------------------------------------------------------ figures
def _load_balanced_go():
    """{model: [pct per dataset]} from the -newclassifier fgsea summaries."""
    vals = {m: [] for m in ["CVAE", "Diffusion", "GAN"]}
    for disp, _, fgsea_folder in DATASETS:
        p = FGSEA_DIR / fgsea_folder / "fgsea_overlap" / "shared_fgsea_summary.csv"
        s = pd.read_csv(p).set_index("Model")["Overlap_Pct_of_Real"].to_dict()
        for m in vals:
            vals[m].append(float(s.get(m, 0.0)))
    return vals


def plot_go():
    vals = _load_balanced_go()
    series = [(m, COLORS[m], vals[m]) for m in ["CVAE", "Diffusion", "GAN"]]
    _grouped(series,
             "Cross-dataset: M5/C5 ontology overlap with Real "
             "(top-10 sets, FDR q<0.05, top-50 genes)",
             "Real ontology groups recovered (%)", "Model",
             "cross_dataset_GO_overlap.png")


def plot_signed_composition(rows):
    """Stacked grouped bars: per dataset, unsigned overlap (solid) and sign-matched
    overlap (hatched), each split into Up(+) red and Down(-) blue (% of 50)."""
    _style()
    # This plot's convention (per request): Up(+) = blue, Down(-) = red.
    comp_up, comp_down = "#2E6FB7", "#C0392B"
    n = len(DISPLAY)
    x = np.arange(n)
    bw = 0.36
    xu, xs = x - bw / 2 - 0.02, x + bw / 2 + 0.02   # unsigned / signed positions
    fig, ax = plt.subplots(figsize=(max(9, 1.9 * n), 6.0))

    def stack(xpos, up, down, hatch):
        ax.bar(xpos, up, bw, color=comp_up, edgecolor="black", linewidth=0.5,
               hatch=hatch, zorder=3)
        ax.bar(xpos, down, bw, bottom=up, color=comp_down, edgecolor="black",
               linewidth=0.5, hatch=hatch, zorder=3)
        for xp, tot in zip(xpos, np.array(up) + np.array(down)):
            ax.text(xp, tot + 1.0, f"{tot:.0f}", ha="center", va="bottom", fontsize=8, zorder=4)

    uns_up = [r["uns_up"] for r in rows];   uns_dn = [r["uns_down"] for r in rows]
    sgn_up = [r["sgn_up"] for r in rows];   sgn_dn = [r["sgn_down"] for r in rows]
    stack(xu, uns_up, uns_dn, "")
    stack(xs, sgn_up, sgn_dn, "////")

    ax.set_ylim(0, 100)
    ax.set_ylabel("Top-50 DEGs recovered (%)")
    ax.set_title("GAN: Top-50 DEG overlap with Real — up/down composition\n"
                 "(left bar = unsigned overlap, right hatched = sign-matched; % of 50)",
                 fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY, rotation=30, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    handles = [
        Patch(facecolor=comp_up, edgecolor="black", label="Up (+)"),
        Patch(facecolor=comp_down, edgecolor="black", label="Down (−)"),
        Patch(facecolor="white", edgecolor="black", label="Unsigned overlap"),
        Patch(facecolor="white", edgecolor="black", hatch="////", label="Sign-matched"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", ncol=1)
    _ood_decorations(ax, n)
    fig.subplots_adjust(bottom=0.30)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "gan_deg_overlap_signed_composition.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  plot -> {out}")


def plot_gan(rows):
    _grouped(
        [("Top-50 overlap", UNSIGNED_COLOR, [r["unsigned"] for r in rows]),
         ("Sign-matched (up=up, down=down)", SIGNED_COLOR, [r["signed"] for r in rows])],
        "GAN: Top-50 DEG overlap with Real — unsigned vs sign-matched\n"
        "(equal bars => directionality preserved)",
        "Top-50 DEGs recovered (%)", "Overlap",
        "gan_deg_overlap_signed.png")
    _grouped(
        [("Upregulated", UP_COLOR, [r["up"] for r in rows]),
         ("Downregulated", DOWN_COLOR, [r["down"] for r in rows])],
        "GAN: Directional Top-50 DEG overlap with Real\n"
        "(up- vs down-regulated, % of 50)",
        "Top-50 directional DEGs recovered (%)", "Direction",
        "gan_deg_overlap_updown.png")


def main():
    print("Regenerating GO figure...")
    plot_go()
    print("Building GAN DEG figures...")
    rows = gan_overlaps_pct()
    for (disp, _, _), r in zip(DATASETS, rows):
        print(f"  {disp:18s} unsigned={r['unsigned']:.0f}%  signed={r['signed']:.0f}%  "
              f"up={r['up']:.0f}%  down={r['down']:.0f}%")
    plot_gan(rows)
    print("Building GAN signed up/down composition figure...")
    comp = gan_signed_composition_pct()
    plot_signed_composition(comp)
    print(f"\n[OK] 4 figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
