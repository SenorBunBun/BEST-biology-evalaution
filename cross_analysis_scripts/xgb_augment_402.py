#!/usr/bin/env python3
"""GAN augmentation dose-response on OSDR-402 using the tuned XGBoost classifier
instead of the RF, requested as a reality check despite XGB not being
significantly better than RF on the real data.

NO pipeline code changed. This reuses the EXACT augmentation strategy from
pipeline/analysis/compute_classifier_augmented.py (same leakage-safe design:
val-only real cells, one fixed fully-stratified rf_eval split, inject
round(frac*N_val) class-matched cells across val's joint strata, n_resamples
redraw+refit, delta vs duplicate-real) by importing its helper functions — only
the classifier is swapped (RF -> XGBoost with the best real-402 hyperparameters;
imbalance via scale_pos_weight recomputed on each training set).

GAN only, full dose suite [0.25, 0.5, 1.0, 1.5, 2.0]. Outputs mirror the
pipeline augmentation step (scores_table.csv, deltas_table.csv, per-frac dirs,
injection_log.json, rf_results.csv) plus the shared comparison plot + summary.

Output folder (dedicated): cross_analysis_scripts/results/OSDR-402-xgb/
Run with .venv_pipeline."""
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
# reuse the pipeline's own helpers unchanged
sys.path.insert(0, str(PIPELINE / "analysis"))
sys.path.insert(0, str(PIPELINE))
from compute_classifier_augmented import (  # noqa: E402
    score_predictions, to_dense, strata_key, allocate_counts,
    draw_by_stratum, aggregate, write_rf_results, METRIC_ORDER)
from utils.data_splitter import stratified_split, load_split_indices  # noqa: E402

# ---- config ----
REAL = PIPELINE / "real_datasets" / "osd_402_preprocessed.h5ad"
FOLDER = "OSDR-402-mmd-complete-pca"
SPLIT = SSD / FOLDER / "data_split_indices.npz"
GAN = sorted((SSD / FOLDER / "synthetic_data").glob("gan_bestmmd_*.h5ad"))[0]
COLUMN, G1, G2 = "Flight", "Flight", "Ground"
COND_COLS = ["Flight", "Strain", "Sex", "Age at Launch", "Duration"]
PCT_VAL_EVAL = 0.25
INJECT_FRACS = [0.25, 0.5, 1.0, 1.5, 2.0]
N_RESAMPLES = 10
N_GENES = 50
SEED = 42
# tuned XGB (best real-402 config from xgb_402_real.py)
XGB_PARAMS = dict(n_estimators=100, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.3, reg_lambda=1.0,
                  tree_method="hist", eval_metric="logloss", n_jobs=-1)
OUT = REPO / "cross_analysis_scripts" / "results" / "OSDR-402-xgb"
AUG_DIR = OUT / "analysis" / "classifier_augmented"
MODEL = "gan"


def xgb_fit_eval(X_train, y_train, X_eval, y_eval, seed):
    npos, nneg = int(np.sum(y_train == 1)), int(np.sum(y_train == 0))
    spw = (nneg / npos) if npos > 0 else 1.0
    clf = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_eval)
    y_proba = clf.predict_proba(X_eval)[:, 1]
    return score_predictions(y_eval, y_pred, y_proba), clf.feature_importances_


def main():
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Real: {REAL.name}\nGAN:  {GAN.name}\nOut:  {AUG_DIR}")

    real = sc.read_h5ad(REAL); real.obs_names_make_unique()
    real.obs[COLUMN] = real.obs[COLUMN].astype(str)
    _, val_idx = load_split_indices(str(SPLIT))
    val = real[val_idx].copy()
    val = val[val.obs[COLUMN].isin([G1, G2])].copy()
    N_val = val.n_obs
    gene_names = val.var_names.values
    strat_cols = [c for c in COND_COLS if c in val.obs.columns]
    if COLUMN not in strat_cols:
        strat_cols = [COLUMN] + strat_cols
    print(f"Stratify on {strat_cols} | N_val(2 classes)={N_val}")

    # fixed eval / pool via the shared splitter
    pool_pos, eval_pos = stratified_split(val, strat_cols, val_ratio=PCT_VAL_EVAL, seed=SEED)
    eval_a, pool_a = val[eval_pos], val[pool_pos]
    X_eval = to_dense(eval_a.X); y_eval = (eval_a.obs[COLUMN].values == G2).astype(int)
    X_pool = to_dense(pool_a.X); y_pool = (pool_a.obs[COLUMN].values == G2).astype(int)
    print(f"rf_eval={eval_a.n_obs}  rf_train_pool={pool_a.n_obs}")

    val_keys = strata_key(val.obs, strat_cols)
    val_sizes = OrderedDict(pd.Series(val_keys).value_counts().sort_index().items())
    pool_keys = strata_key(pool_a.obs, strat_cols)
    pool_positions = {s: np.where(pool_keys == s)[0] for s in val_sizes}
    pool_all = np.arange(pool_a.n_obs)
    eval_keys = strata_key(eval_a.obs, strat_cols)

    # GAN synthetic pool
    sa = sc.read_h5ad(GAN); sa.obs_names_make_unique()
    if list(sa.var_names) != list(val.var_names):
        raise ValueError("GAN var_names differ from real")
    sa.obs[COLUMN] = sa.obs[COLUMN].astype(str)
    sa = sa[sa.obs[COLUMN].isin([G1, G2])].copy()
    scols = [c for c in strat_cols if c in sa.obs.columns]
    skeys = strata_key(sa.obs, scols) if scols else np.array([''] * sa.n_obs)
    s_X, s_y = sa.X, (sa.obs[COLUMN].values == G2).astype(int)
    s_pos = {s: np.where(skeys == s)[0] for s in np.unique(skeys)}
    print(f"GAN pool: {sa.n_obs} cells")

    # baseline 'real' (compute once)
    rmd, rimp = [], []
    for r in range(N_RESAMPLES):
        m, imp = xgb_fit_eval(X_pool, y_pool, X_eval, y_eval, SEED + r)
        rmd.append(m); rimp.append(imp)
    real_mean, real_std, real_imp = aggregate(rmd, rimp)
    write_rf_results(real_imp, gene_names, N_GENES, AUG_DIR / "real")
    print(f"real baseline bal_acc={real_mean['balanced_accuracy']:.3f}")

    all_scores, all_deltas = [], []

    def rec(frac, cond, mean, std):
        for mt in METRIC_ORDER:
            all_scores.append({'inject_frac': frac, 'condition': cond, 'metric': mt,
                               'mean': mean[mt], 'std': std[mt]})

    for frac in INJECT_FRACS:
        fdir = AUG_DIR / f"frac_{frac}"; fdir.mkdir(parents=True, exist_ok=True)
        n_add = int(round(frac * N_val))
        alloc = allocate_counts(val_sizes, n_add)
        rec(frac, 'real', real_mean, real_std)

        # real_dup (with-replacement)
        dmd, dimp = [], []
        for r in range(N_RESAMPLES):
            rng = np.random.RandomState(SEED + 1000 + r)
            add, _ = draw_by_stratum(pool_positions, alloc, rng, True, pool_all)
            m, imp = xgb_fit_eval(np.vstack([X_pool, X_pool[add]]),
                                  np.concatenate([y_pool, y_pool[add]]),
                                  X_eval, y_eval, SEED + r)
            dmd.append(m); dimp.append(imp)
        dup_mean, dup_std, dup_imp = aggregate(dmd, dimp)
        write_rf_results(dup_imp, gene_names, N_GENES, fdir / "real_dup")
        rec(frac, 'real_dup', dup_mean, dup_std)

        # real_synth_gan (without replacement, fallback inside draw_by_stratum)
        smd, simp, fb_any = [], [], False
        for r in range(N_RESAMPLES):
            rng = np.random.RandomState(SEED + 2000 + r)
            add, fb = draw_by_stratum(s_pos, alloc, rng, False, np.arange(len(s_y)))
            fb_any = fb_any or fb
            m, imp = xgb_fit_eval(np.vstack([X_pool, to_dense(s_X[add])]),
                                  np.concatenate([y_pool, s_y[add]]),
                                  X_eval, y_eval, SEED + r)
            smd.append(m); simp.append(imp)
        sy_mean, sy_std, sy_imp = aggregate(smd, simp)
        write_rf_results(sy_imp, gene_names, N_GENES, fdir / MODEL)
        rec(frac, MODEL, sy_mean, sy_std)
        for mt in METRIC_ORDER:
            all_deltas.append({
                'inject_frac': frac, 'model': MODEL, 'metric': mt,
                'delta_vs_dup': sy_mean[mt] - dup_mean[mt],
                'delta_vs_dup_std': float(np.sqrt(sy_std[mt] ** 2 + dup_std[mt] ** 2)),
                'delta_vs_real': sy_mean[mt] - real_mean[mt],
                'delta_vs_real_std': float(np.sqrt(sy_std[mt] ** 2 + real_std[mt] ** 2)),
            })
        d = sy_mean['balanced_accuracy'] - dup_mean['balanced_accuracy']
        print(f"  frac={frac}: n_add={n_add}  gan bal_acc={sy_mean['balanced_accuracy']:.3f} "
              f"(dup {dup_mean['balanced_accuracy']:.3f}, real {real_mean['balanced_accuracy']:.3f}) "
              f"Δvs_dup={d:+.3f}  [fallback={fb_any}]")
        json.dump({'inject_frac': frac, 'N_val': int(N_val), 'n_add_total': int(n_add),
                   'rf_eval_n': int(eval_a.n_obs), 'rf_train_pool_n': int(pool_a.n_obs),
                   'strat_cols': strat_cols, 'n_resamples_augment': N_RESAMPLES,
                   'n_add_per_stratum': {str(k): int(v) for k, v in alloc.items()},
                   'replacement_used': {'real_dup': True, MODEL: bool(fb_any)},
                   'classifier': 'XGBoost', 'xgb_params': {k: v for k, v in XGB_PARAMS.items()}},
                  open(fdir / "injection_log.json", 'w'), indent=2)

    pd.DataFrame(all_scores).to_csv(AUG_DIR / "scores_table.csv", index=False)
    pd.DataFrame(all_deltas).to_csv(AUG_DIR / "deltas_table.csv", index=False)
    print(f"\n[OK] scores_table.csv + deltas_table.csv -> {AUG_DIR}")

    # shared comparison plot + best-model summary (unchanged pipeline script)
    subprocess.run([sys.executable, str(PIPELINE / "analysis" / "shared_classifier_augmented_comparison.py"),
                    "--aug-dir", str(AUG_DIR), "--output-dir", str(AUG_DIR),
                    "--core-output-dir", str(OUT / "analysis"),
                    "--dataset-name", "OSDR-402-XGB", "--headline-metric", "balanced_accuracy"],
                   check=False)
    print(f"\n[DONE] all outputs under {OUT}")


if __name__ == "__main__":
    main()
