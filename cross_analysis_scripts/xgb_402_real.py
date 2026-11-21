#!/usr/bin/env python3
"""Reality check: tuned XGBoost on REAL OSDR-402 (Flight vs Ground), compared to
the RF phenotype classifier and the tuned logistic regression on the same data.

No pipeline code changed. Standalone. Matches the RF classifier evaluation:
  - features: all HVG columns of adata.X (log-normalized), both groups only;
  - splits: RepeatedStratifiedKFold(5 x 3, random_state=42);
  - 8 metrics, mean +/- std over the 15 folds.

Class imbalance handled with scale_pos_weight = n_neg / n_pos (no feature
scaling needed for trees). Hyperparameters tuned to MAXIMIZE CV balanced
accuracy on the real data, then the best config is re-evaluated across the same
splits for all 8 metrics. Run with .venv_pipeline."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, balanced_accuracy_score, roc_auc_score,
                             average_precision_score)

REPO = Path("/Users/ljeytl/Arsh/Research/NASA/AI4LS/BEST_bioeval")
REAL = REPO / "pipeline" / "real_datasets" / "osd_402_preprocessed.h5ad"
COLUMN, GROUP1, GROUP2 = "Flight", "Flight", "Ground"   # group1=0, group2=1
N_FOLDS, N_REPEATS, SEED = 5, 3, 42
OUT_DIR = REPO / "cross_analysis_scripts" / "results"
METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "f1_macro",
                "balanced_accuracy", "roc_auc", "pr_auc"]
RF_REF = {"accuracy": 0.6658, "precision": 0.3982, "recall": 0.2232, "f1": 0.2855,
          "f1_macro": 0.5337, "balanced_accuracy": 0.5393, "roc_auc": 0.5858,
          "pr_auc": 0.3779}
LR_REF = {"accuracy": 0.5813, "precision": 0.3566, "recall": 0.4935, "f1": 0.4131,
          "f1_macro": 0.5434, "balanced_accuracy": 0.5562, "roc_auc": 0.5847,
          "pr_auc": 0.3714}


def load_xy():
    adata = sc.read_h5ad(REAL)
    adata.obs[COLUMN] = adata.obs[COLUMN].astype(str)
    a = adata[adata.obs[COLUMN].isin([str(GROUP1), str(GROUP2)])].copy()
    X = a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
    y = (a.obs[COLUMN] == str(GROUP2)).astype(int).values
    return np.asarray(X, dtype=float), y


def score_fold(params, spw, Xtr, Xte, ytr, yte):
    clf = XGBClassifier(**params, scale_pos_weight=spw, tree_method="hist",
                        eval_metric="logloss", n_jobs=1, random_state=SEED)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    pr = clf.predict_proba(Xte)[:, 1]
    roc = roc_auc_score(yte, pr) if len(np.unique(yte)) > 1 else np.nan
    prc = average_precision_score(yte, pr) if len(np.unique(yte)) > 1 else np.nan
    return {"accuracy": accuracy_score(yte, yp),
            "precision": precision_score(yte, yp, zero_division=0),
            "recall": recall_score(yte, yp, zero_division=0),
            "f1": f1_score(yte, yp, zero_division=0),
            "f1_macro": f1_score(yte, yp, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(yte, yp),
            "roc_auc": roc, "pr_auc": prc}


def main():
    X, y = load_xy()
    spw = np.sum(y == 0) / np.sum(y == 1)
    print(f"OSDR-402 real: X={X.shape}  Flight(0)={np.sum(y==0)}  Ground(1)={np.sum(y==1)}  "
          f"scale_pos_weight={spw:.3f}")
    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)

    base = XGBClassifier(scale_pos_weight=spw, tree_method="hist",
                         eval_metric="logloss", n_jobs=1, random_state=SEED)
    grid = {
        "n_estimators": [100, 300],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.3, 0.6],
        "reg_lambda": [1.0],
    }
    print("Tuning XGBoost to maximize CV balanced accuracy...")
    gs = GridSearchCV(base, grid, scoring="balanced_accuracy", cv=rskf,
                      n_jobs=-1, refit=True)
    gs.fit(X, y)
    best = gs.best_params_
    print(f"  best params: {best}")
    print(f"  best CV balanced_accuracy = {gs.best_score_:.4f}")

    # Re-evaluate best config across the SAME 15 splits for all 8 metrics.
    rows = [score_fold(best, spw, X[tr], X[te], y[tr], y[te]) for tr, te in rskf.split(X, y)]
    mdf = pd.DataFrame(rows)
    xm, xs = mdf.mean(), mdf.std()

    print("\n=== XGB vs LR vs RF (real 402, balanced accuracy comparison) ===")
    print(f"{'metric':18s} {'XGB':>9s} {'XGBstd':>8s} {'LR':>9s} {'RF':>9s} {'XGB-RF':>8s}")
    comp = []
    for m in METRIC_ORDER:
        print(f"{m:18s} {xm[m]:>9.4f} {xs[m]:>8.4f} {LR_REF[m]:>9.4f} {RF_REF[m]:>9.4f} "
              f"{xm[m]-RF_REF[m]:>+8.4f}")
        comp.append({"metric": m, "XGB_mean": round(float(xm[m]), 4),
                     "XGB_std": round(float(xs[m]), 4), "LR_mean": LR_REF[m],
                     "RF_mean": RF_REF[m], "XGB_minus_RF": round(float(xm[m] - RF_REF[m]), 4)})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comp).to_csv(OUT_DIR / "xgb_vs_lr_rf_402_real.csv", index=False)
    json.dump({"dataset": "OSDR-402", "contrast": "Flight vs Ground",
               "best_params": {k: (float(v) if isinstance(v, np.floating) else v)
                               for k, v in best.items()},
               "scale_pos_weight": round(float(spw), 4),
               "best_cv_balanced_accuracy": round(float(gs.best_score_), 4),
               "xgb_metrics_mean": {m: round(float(xm[m]), 4) for m in METRIC_ORDER},
               "lr_reference": LR_REF, "rf_reference": RF_REF},
              open(OUT_DIR / "xgb_402_real_summary.json", "w"), indent=2)
    print(f"\n[OK] wrote {OUT_DIR/'xgb_vs_lr_rf_402_real.csv'}")
    print(f"[OK] wrote {OUT_DIR/'xgb_402_real_summary.json'}")


if __name__ == "__main__":
    main()
