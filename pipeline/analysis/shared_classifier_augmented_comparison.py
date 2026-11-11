#!/usr/bin/env python3
"""
Shared Augmentation-Classifier Comparison.

Augmentation-only companion to shared_classifier_comparison.py. Reads the
numeric tables written by compute_classifier_augmented.py (scores_table.csv and
deltas_table.csv) and produces the combined comparison plot plus a best-model
summary (which generative model's synthetic most improves held-out real
classification, over the duplicated-real control).

Usage:
    python shared_classifier_augmented_comparison.py \
        --aug-dir .../analysis/classifier_augmented \
        --output-dir .../analysis/classifier_augmented \
        --core-output-dir .../analysis \
        --dataset-name "OSDR-403" --headline-metric balanced_accuracy
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 16, 'font.family': 'Arial', 'axes.labelsize': 20,
    'axes.titlesize': 22, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'legend.fontsize': 15, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.facecolor': 'white', 'axes.linewidth': 1.5,
})

CONDITION_ORDER = ['real', 'real_dup', 'cvae', 'diffusion', 'gan']
CONDITION_DISPLAY = {
    'real': 'Real only', 'real_dup': 'Real + dup',
    'cvae': 'Real + CVAE', 'diffusion': 'Real + Diffusion', 'gan': 'Real + GAN',
}
CONDITION_COLORS = {
    'real': '#95A5A6', 'real_dup': '#34495E',
    'cvae': '#8E44AD', 'diffusion': '#E57373', 'gan': '#E69F00',
}
MODEL_DISPLAY = {'cvae': 'CVAE', 'diffusion': 'Diffusion', 'gan': 'GAN'}


def parse_args():
    p = argparse.ArgumentParser(description='Combined augmentation comparison plot + summary')
    p.add_argument('--aug-dir', required=True,
                   help='classifier_augmented dir with scores_table.csv / deltas_table.csv')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--core-output-dir', default=None)
    p.add_argument('--dataset-name', default='Dataset')
    p.add_argument('--headline-metric', default='balanced_accuracy')
    p.add_argument('--show-plots', action='store_true')
    return p.parse_args()


def plot_single_frac(scores, frac, metric, output_dir, dataset_name,
                     core_output_dir=None, show_plots=False):
    sub = scores[(scores['inject_frac'] == frac) & (scores['metric'] == metric)]
    conds, means, stds, colors = [], [], [], []
    for c in CONDITION_ORDER:
        row = sub[sub['condition'] == c]
        if row.empty:
            continue
        conds.append(CONDITION_DISPLAY[c])
        means.append(float(row['mean'].values[0]))
        stds.append(float(row['std'].values[0]))
        colors.append(CONDITION_COLORS[c])

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(conds, means, yerr=stds, color=colors, alpha=0.9,
                  edgecolor='black', linewidth=2, width=0.6, capsize=6)
    ax.set_title(f'Augmentation: {metric.replace("_", " ")}\n'
                 f'{dataset_name} (inject {int(frac*100)}% of val)',
                 fontweight='bold', pad=20)
    ax.set_ylabel(metric.replace('_', ' '), fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#FAFAFA')
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    return _save(fig, output_dir, f"augmentation_comparison_frac{frac}", core_output_dir,
                 core_name=None, show_plots=show_plots)


def plot_sweep(scores, fracs, metric, output_dir, dataset_name,
               core_output_dir=None, show_plots=False):
    fig, ax = plt.subplots(figsize=(12, 8))
    for c in CONDITION_ORDER:
        ys, es, xs = [], [], []
        for frac in fracs:
            row = scores[(scores['inject_frac'] == frac) & (scores['metric'] == metric)
                         & (scores['condition'] == c)]
            if row.empty:
                continue
            xs.append(frac * 100)
            ys.append(float(row['mean'].values[0]))
            es.append(float(row['std'].values[0]))
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker='o', markersize=8, linewidth=2.5,
                        capsize=5, label=CONDITION_DISPLAY[c], color=CONDITION_COLORS[c])
    ax.set_title(f'Augmentation dose-response: {metric.replace("_", " ")}\n{dataset_name}',
                 fontweight='bold', pad=20)
    ax.set_xlabel('Injected synthetic (% of val size)', fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' '), fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#FAFAFA')
    ax.legend(loc='best')
    plt.tight_layout()
    return _save(fig, output_dir, "augmentation_sweep", core_output_dir,
                 core_name="augmentation_sweep", show_plots=show_plots)


def _save(fig, output_dir, stem, core_output_dir, core_name, show_plots):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{stem}_{ts}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches='tight', facecolor='white')
    if core_output_dir and core_name:
        core = Path(core_output_dir); core.mkdir(parents=True, exist_ok=True)
        fig.savefig(core / f"{core_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    print(f"[OK] Plot saved: {path}")
    return str(path)


def build_summary(deltas, fracs, metric, dataset_name):
    summary = {'dataset_name': dataset_name, 'timestamp': datetime.now().isoformat(),
               'headline_metric': metric, 'per_inject_frac': {}, 'best_overall': None}
    best_overall = None
    for frac in fracs:
        sub = deltas[(deltas['inject_frac'] == frac) & (deltas['metric'] == metric)]
        entry = {}
        for _, r in sub.iterrows():
            entry[MODEL_DISPLAY.get(r['model'], r['model'])] = {
                'delta_vs_dup': round(float(r['delta_vs_dup']), 4),
                'delta_vs_dup_std': round(float(r['delta_vs_dup_std']), 4),
                'delta_vs_real': round(float(r['delta_vs_real']), 4),
            }
        if not sub.empty:
            best_row = sub.loc[sub['delta_vs_dup'].idxmax()]
            best_model = MODEL_DISPLAY.get(best_row['model'], best_row['model'])
            entry['best_model'] = best_model
            entry['best_delta_vs_dup'] = round(float(best_row['delta_vs_dup']), 4)
            if best_overall is None or best_row['delta_vs_dup'] > best_overall[1]:
                best_overall = (best_model, float(best_row['delta_vs_dup']), frac)
        summary['per_inject_frac'][str(frac)] = entry
    if best_overall:
        summary['best_overall'] = {'model': best_overall[0],
                                   'delta_vs_dup': round(best_overall[1], 4),
                                   'inject_frac': best_overall[2]}
    return summary


def main():
    args = parse_args()
    aug_dir = Path(args.aug_dir)
    scores_path = aug_dir / "scores_table.csv"
    deltas_path = aug_dir / "deltas_table.csv"
    if not scores_path.exists():
        print(f"[ERROR] scores_table.csv not found in {aug_dir}")
        sys.exit(1)

    scores = pd.read_csv(scores_path)
    deltas = pd.read_csv(deltas_path) if deltas_path.exists() else pd.DataFrame()
    metric = args.headline_metric
    fracs = sorted(scores['inject_frac'].unique())
    print(f"Augmentation comparison: {args.dataset_name} | metric={metric} | fracs={fracs}")

    output_dir = Path(args.output_dir)
    if len(fracs) == 1:
        plot_single_frac(scores, fracs[0], metric, output_dir, args.dataset_name,
                         core_output_dir=args.core_output_dir, show_plots=args.show_plots)
    else:
        plot_sweep(scores, fracs, metric, output_dir, args.dataset_name,
                   core_output_dir=args.core_output_dir, show_plots=args.show_plots)
        # also a per-frac bar for each dosage
        for frac in fracs:
            plot_single_frac(scores, frac, metric, output_dir, args.dataset_name,
                             show_plots=False)

    if not deltas.empty:
        summary = build_summary(deltas, fracs, metric, args.dataset_name)
        with open(output_dir / "shared_classifier_augmented_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        # flat CSV
        rows = []
        for frac in fracs:
            sub = deltas[(deltas['inject_frac'] == frac) & (deltas['metric'] == metric)]
            for _, r in sub.iterrows():
                rows.append({'inject_frac': frac, 'model': MODEL_DISPLAY.get(r['model'], r['model']),
                             'delta_vs_dup': round(float(r['delta_vs_dup']), 4),
                             'delta_vs_real': round(float(r['delta_vs_real']), 4)})
        pd.DataFrame(rows).to_csv(output_dir / "shared_classifier_augmented_summary.csv", index=False)
        print(f"[OK] Summary written. Best overall: {summary.get('best_overall')}")

    sys.exit(0)


if __name__ == "__main__":
    main()
