#!/usr/bin/env python3
"""Reality check: optimized logistic regression on REAL OSDR-402 (Flight vs
Ground), compared to the RF phenotype classifier on the same data.

No pipeline code changed. Standalone. Matches the RF classifier's evaluation
exactly for a fair comparison:
  - same features: all HVG columns of adata.X (log-normalized, as the pipeline
    leaves them), both groups only;
  - same splits: RepeatedStratifiedKFold(5 x 3, random_state=42);
  - same 8 metrics, mean +/- std over the 15 folds.

Logistic regression is L2 / elastic-net with class_weight='balanced', features
standardized inside each CV fold (StandardScaler in a Pipeline, so no leakage).
Hyperparameters are tuned to MAXIMIZE cross-validated balanced accuracy on the
real data (GridSearchCV over the same 15 splits), then the best configuration is
re-evaluated across those splits to report all 8 metrics.

Run with .venv_pipeline. Writes results to cross_analysis_scripts/results/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
# RF (balanced) real-402 reference, from
# outputs/OSDR-402-mmd-complete-pca/analysis/classifier_balanced/real/rf_metrics.csv
RF_REF = {"accuracy": 0.6658, "precision": 0.3982, "recall": 0.2232, "f1": 0.2855,
          "f1_macro": 0.5337, "balanced_accuracy": 0.5393, "roc_auc": 0.5858,
          "pr_auc": 0.3779}


def load_xy():
    adata = sc.read_h5ad(REAL)
    adata.obs[COLUMN] = adata.obs[COLUMN].astype(str)
    mask = adata.obs[COLUMN].isin([str(GROUP1), str(GROUP2)])
    a = adata[mask].copy()
    X = a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
    y = (a.obs[COLUMN] == str(GROUP2)).astype(int).values
    return np.asarray(X, dtype=float), y


def score_fold(clf, Xtr, Xte, ytr, yte):
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
    print(f"OSDR-402 real: X={X.shape}  Flight(0)={np.sum(y==0)}  Ground(1)={np.sum(y==1)}")
    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=2000,
                                  tol=1e-3, random_state=SEED)),
    ])
    grid = [
        # L2: fast lbfgs solver
        {"lr__penalty": ["l2"], "lr__solver": ["lbfgs"],
         "lr__C": np.logspace(-3, 2, 10)},
        # elastic-net: saga (only solver supporting it), smaller grid
        {"lr__penalty": ["elasticnet"], "lr__solver": ["saga"],
         "lr__C": np.logspace(-2, 1, 5), "lr__l1_ratio": [0.3, 0.6, 0.9]},
    ]
    print("Tuning LR (L2 / elastic-net) to maximize CV balanced accuracy...")
    gs = GridSearchCV(pipe, grid, scoring="balanced_accuracy", cv=rskf,
                      n_jobs=-1, refit=True)
    gs.fit(X, y)
    best = gs.best_params_
    print(f"  best params: {best}")
    print(f"  best CV balanced_accuracy = {gs.best_score_:.4f}")

    # Re-evaluate the best config across the SAME 15 splits for all 8 metrics.
    best_pipe = gs.best_estimator_
    rows = []
    for tr, te in rskf.split(X, y):
        rows.append(score_fold(
            Pipeline([("scaler", StandardScaler()),
                      ("lr", best_pipe.named_steps["lr"].__class__(**best_pipe.named_steps["lr"].get_params()))]),
            X[tr], X[te], y[tr], y[te]))
    mdf = pd.DataFrame(rows)
    lr_mean, lr_std = mdf.mean(), mdf.std()

    print("\n=== LR (real 402) vs RF (real 402, balanced) ===")
    print(f"{'metric':18s} {'LR mean':>9s} {'LR std':>8s} {'RF mean':>9s} {'LR-RF':>8s}")
    comp = []
    for m in METRIC_ORDER:
        d = lr_mean[m] - RF_REF[m]
        print(f"{m:18s} {lr_mean[m]:>9.4f} {lr_std[m]:>8.4f} {RF_REF[m]:>9.4f} {d:>+8.4f}")
        comp.append({"metric": m, "LR_mean": round(float(lr_mean[m]), 4),
                     "LR_std": round(float(lr_std[m]), 4),
                     "RF_mean": RF_REF[m], "LR_minus_RF": round(float(d), 4)})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comp).to_csv(OUT_DIR / "logreg_vs_rf_402_real.csv", index=False)
    json.dump({"dataset": "OSDR-402", "contrast": "Flight vs Ground",
               "best_params": {k: (v if not isinstance(v, np.floating) else float(v))
                               for k, v in best.items()},
               "best_cv_balanced_accuracy": round(float(gs.best_score_), 4),
               "lr_metrics_mean": {m: round(float(lr_mean[m]), 4) for m in METRIC_ORDER},
               "rf_reference": RF_REF},
              open(OUT_DIR / "logreg_402_real_summary.json", "w"), indent=2)
    print(f"\n[OK] wrote {OUT_DIR/'logreg_vs_rf_402_real.csv'}")
    print(f"[OK] wrote {OUT_DIR/'logreg_402_real_summary.json'}")


if __name__ == "__main__":
    main()
