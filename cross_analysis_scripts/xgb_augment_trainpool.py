#!/usr/bin/env python3
"""GAN augmentation dose-response with XGBoost, TRAIN-POOL (leakage-proper) setup,
generalized over datasets. Same design as xgb_augment_402_trainpool.py:

  - rf_train_pool = FULL TRAIN split; rf_eval = ENTIRE VAL split (fixed, shared);
  - dosage n_add = round(inject_frac * |train|), injected class/condition-matched
    to the TRAIN joint-strata mix; synthetic w/o replacement (fallback), dup with.

Reads real path / condition cols / groups from the dataset config JSON; the SSD
output folder (for data_split_indices.npz + synthetic) comes from FOLDER_MAP.
NO pipeline code changed; reuses the pipeline module's helpers.

Usage:  python xgb_augment_trainpool.py --dataset 403
Output: cross_analysis_scripts/results/<DATASET>-xgb-trainpool/
Run with .venv_pipeline (requires the external SSD mounted)."""
import argparse
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from xgboost import XGBClassifier

REPO = Path("/Users/ljeytl/Arsh/Research/NASA/AI4LS/BEST_bioeval")
PIPELINE = REPO / "pipeline"
SSD = Path("/Volumes/Ljeytl's SSD/NASAResearch/AI4LS/pipeline/outputs")
sys.path.insert(0, str(PIPELINE / "analysis"))
sys.path.insert(0, str(PIPELINE))
from compute_classifier_augmented import (  # noqa: E402
    score_predictions, to_dense, strata_key, allocate_counts,
    draw_by_stratum, aggregate, write_rf_results, METRIC_ORDER)
from utils.data_splitter import load_split_indices  # noqa: E402

# short name -> (config stem, SSD/local output folder, display label)
DATASETS = {
    "402": ("osdr-402-mmd-complete-pca", "OSDR-402-mmd-complete-pca", "OSDR-402"),
    "403": ("osdr-403-mmd-complete-pca", "OSDR-403-mmd-complete-pca", "OSDR-403"),
    "404": ("osdr-404-mmd-complete-pca", "OSDR-404-mmd-complete-pca", "OSDR-404"),
    "405": ("osdr-405-mmd-complete-pca", "OSDR-405-mmd-pca", "OSDR-405"),
    "352": ("osdr-352-612-613-mmd-complete-pca", "OSDR-352-612-613-mmd-complete-pca", "OSDR-352-612-613"),
    "crc": ("CRC_IMM_10X-mmd-complete-pca", "CRC_IMM_10X-mmd-complete-pca", "CRC_IMM_10X"),
}
INJECT_FRACS = [0.25, 0.5, 1.0, 1.5, 2.0]
N_RESAMPLES = 10
N_GENES = 50
SEED = 42
XGB_PARAMS = dict(n_estimators=100, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.3, reg_lambda=1.0,
                  tree_method="hist", eval_metric="logloss", n_jobs=-1)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (PIPELINE / p)


def xgb_fit_eval(Xtr, ytr, Xev, yev, seed):
    npos, nneg = int(np.sum(ytr == 1)), int(np.sum(ytr == 0))
    spw = (nneg / npos) if npos > 0 else 1.0
    clf = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw, random_state=seed)
    clf.fit(Xtr, ytr)
    return (score_predictions(yev, clf.predict(Xev), clf.predict_proba(Xev)[:, 1]),
            clf.feature_importances_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    args = ap.parse_args()
    cfg_stem, folder, label = DATASETS[args.dataset]

    cfg = json.load(open(PIPELINE / "dataset_configs" / f"{cfg_stem}.json"))
    real_path = resolve(cfg["paths"]["processed_data"])
    prim = cfg["conditions"]["primary"]
    COLUMN, G1, G2 = prim["column"], str(prim["group1"]), str(prim["group2"])
    COND_COLS = cfg["conditions"]["columns"]
    split = SSD / folder / "data_split_indices.npz"
    gans = sorted((SSD / folder / "synthetic_data").glob("gan_bestmmd_*.h5ad"))
    if not split.exists():
        sys.exit(f"[ERROR] split indices not found (SSD mounted?): {split}")
    if not gans:
        sys.exit(f"[ERROR] GAN synthetic not found (SSD mounted?): {SSD/folder/'synthetic_data'}")
    gan_path = gans[0]

    out = REPO / "cross_analysis_scripts" / "results" / f"{folder}-xgb-trainpool"
    aug_dir = out / "analysis" / "classifier_augmented"
    aug_dir.mkdir(parents=True, exist_ok=True)

    def two_class(a):
        a = a.copy(); a.obs[COLUMN] = a.obs[COLUMN].astype(str)
        return a[a.obs[COLUMN].isin([G1, G2])].copy()

    real = sc.read_h5ad(real_path); real.obs_names_make_unique()
    real.obs[COLUMN] = real.obs[COLUMN].astype(str)
    train_idx, val_idx = load_split_indices(str(split))
    pool_a, eval_a = two_class(real[train_idx]), two_class(real[val_idx])
    gene_names = pool_a.var_names.values
    strat_cols = [c for c in COND_COLS if c in pool_a.obs.columns]
    if COLUMN not in strat_cols:
        strat_cols = [COLUMN] + strat_cols
    X_pool = to_dense(pool_a.X); y_pool = (pool_a.obs[COLUMN].values == G2).astype(int)
    X_eval = to_dense(eval_a.X); y_eval = (eval_a.obs[COLUMN].values == G2).astype(int)
    N_train = pool_a.n_obs
    print(f"{label}: rf_train(TRAIN)={N_train}  rf_eval(VAL)={eval_a.n_obs}  "
          f"{COLUMN} {G1} vs {G2}  strat={strat_cols}")

    pool_keys = strata_key(pool_a.obs, strat_cols)
    train_sizes = OrderedDict(pd.Series(pool_keys).value_counts().sort_index().items())
    pool_positions = {s: np.where(pool_keys == s)[0] for s in train_sizes}
    pool_all = np.arange(N_train)

    sa = two_class(sc.read_h5ad(gan_path)); sa.obs_names_make_unique()
    if list(sa.var_names) != list(pool_a.var_names):
        sys.exit("[ERROR] GAN var_names differ from real")
    scols = [c for c in strat_cols if c in sa.obs.columns]
    skeys = strata_key(sa.obs, scols) if scols else np.array([''] * sa.n_obs)
    s_X, s_y = sa.X, (sa.obs[COLUMN].values == G2).astype(int)
    s_pos = {s: np.where(skeys == s)[0] for s in np.unique(skeys)}
    print(f"GAN pool: {sa.n_obs}")

    rmd, rimp = zip(*[xgb_fit_eval(X_pool, y_pool, X_eval, y_eval, SEED + r)
                      for r in range(N_RESAMPLES)])
    real_mean, real_std, real_imp = aggregate(list(rmd), list(rimp))
    write_rf_results(real_imp, gene_names, N_GENES, aug_dir / "real")
    print(f"real baseline bal_acc={real_mean['balanced_accuracy']:.3f}")

    all_scores, all_deltas = [], []

    def rec(frac, cond, mean, std):
        for mt in METRIC_ORDER:
            all_scores.append({'inject_frac': frac, 'condition': cond, 'metric': mt,
                               'mean': mean[mt], 'std': std[mt]})

    for frac in INJECT_FRACS:
        fdir = aug_dir / f"frac_{frac}"; fdir.mkdir(parents=True, exist_ok=True)
        n_add = int(round(frac * N_train))
        alloc = allocate_counts(train_sizes, n_add)
        rec(frac, 'real', real_mean, real_std)

        dmd, dimp = [], []
        for r in range(N_RESAMPLES):
            rng = np.random.RandomState(SEED + 1000 + r)
            add, _ = draw_by_stratum(pool_positions, alloc, rng, True, pool_all)
            m, imp = xgb_fit_eval(np.vstack([X_pool, X_pool[add]]),
                                  np.concatenate([y_pool, y_pool[add]]), X_eval, y_eval, SEED + r)
            dmd.append(m); dimp.append(imp)
        dup_mean, dup_std, dup_imp = aggregate(dmd, dimp)
        write_rf_results(dup_imp, gene_names, N_GENES, fdir / "real_dup")
        rec(frac, 'real_dup', dup_mean, dup_std)

        smd, simp, fb = [], [], False
        for r in range(N_RESAMPLES):
            rng = np.random.RandomState(SEED + 2000 + r)
            add, f = draw_by_stratum(s_pos, alloc, rng, False, np.arange(len(s_y)))
            fb = fb or f
            m, imp = xgb_fit_eval(np.vstack([X_pool, to_dense(s_X[add])]),
                                  np.concatenate([y_pool, s_y[add]]), X_eval, y_eval, SEED + r)
            smd.append(m); simp.append(imp)
        sy_mean, sy_std, sy_imp = aggregate(smd, simp)
        write_rf_results(sy_imp, gene_names, N_GENES, fdir / "gan")
        rec(frac, "gan", sy_mean, sy_std)
        for mt in METRIC_ORDER:
            all_deltas.append({'inject_frac': frac, 'model': 'gan', 'metric': mt,
                               'delta_vs_dup': sy_mean[mt] - dup_mean[mt],
                               'delta_vs_dup_std': float(np.sqrt(sy_std[mt] ** 2 + dup_std[mt] ** 2)),
                               'delta_vs_real': sy_mean[mt] - real_mean[mt],
                               'delta_vs_real_std': float(np.sqrt(sy_std[mt] ** 2 + real_std[mt] ** 2))})
        print(f"  frac={frac}: n_add={n_add}  gan={sy_mean['balanced_accuracy']:.3f} "
              f"(dup {dup_mean['balanced_accuracy']:.3f}, real {real_mean['balanced_accuracy']:.3f}) "
              f"Δdup={sy_mean['balanced_accuracy']-dup_mean['balanced_accuracy']:+.3f} [fallback={fb}]")
        json.dump({'inject_frac': frac, 'N_train': int(N_train), 'n_add_total': int(n_add),
                   'rf_eval_n': int(eval_a.n_obs), 'eval_set': 'val_split', 'train_set': 'train_split',
                   'strat_cols': strat_cols, 'n_resamples_augment': N_RESAMPLES,
                   'n_add_per_stratum': {str(k): int(v) for k, v in alloc.items()},
                   'replacement_used': {'real_dup': True, 'gan': bool(fb)},
                   'classifier': 'XGBoost', 'xgb_params': dict(XGB_PARAMS)},
                  open(fdir / "injection_log.json", 'w'), indent=2)

    pd.DataFrame(all_scores).to_csv(aug_dir / "scores_table.csv", index=False)
    pd.DataFrame(all_deltas).to_csv(aug_dir / "deltas_table.csv", index=False)
    print(f"\n[OK] tables -> {aug_dir}")
    subprocess.run([sys.executable, str(PIPELINE / "analysis" / "shared_classifier_augmented_comparison.py"),
                    "--aug-dir", str(aug_dir), "--output-dir", str(aug_dir),
                    "--core-output-dir", str(out / "analysis"),
                    "--dataset-name", f"{label}-XGB-trainpool", "--headline-metric", "balanced_accuracy"],
                   check=False)
    print(f"[DONE] outputs under {out}")


if __name__ == "__main__":
    main()
