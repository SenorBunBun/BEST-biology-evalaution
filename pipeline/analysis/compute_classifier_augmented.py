#!/usr/bin/env python3
"""
Compute Synthetic-Data AUGMENTATION Classification for a Single Dataset.

Augmentation-only companion to compute_classifier.py. Tests whether adding
synthetic cells to a scarce real training set improves downstream phenotype
classification, and which generative model's data helps most — measured on a
single fixed, fully-stratified, real-only evaluation set per dataset.

Design (see pipeline/AUGMENTATION_PROTOCOL.md):
  - Real cells used = the VALIDATION split only (val_indices from the shared
    data_split_indices.npz). Generative models never trained on val, so it is
    leakage-safe.
  - A fixed `rf_eval` set is split off ONCE via the same stratified_split used
    everywhere else (fully stratified on all condition columns, NOT 50/50).
    The remainder is `rf_train_pool`. Both are frozen and shared across every
    condition and model.
  - Conditions (all scored on the same real `rf_eval`):
        real              - rf_train_pool only (floor)
        real_dup          - + duplicated real cells (controls for pure volume)
        real_synth_<model>- + synthetic cells from that model
    real_dup and real_synth_* reach identical size and the same full joint
    stratum mix, so they differ only by duplicate-vs-synthetic.
  - Injection volume `n_add = round(inject_frac * |val|)`, allocated across
    val's full joint strata (same combined key as stratified_split). Synthetic
    drawn without replacement (warn+fallback if a stratum is exhausted); real
    duplicates drawn with replacement.
  - `n_resamples_augment` repeats of (redraw augment sample + refit RF) give
    mean +/- std, since the fixed eval set precludes CV-fold rotation.

Writes the numeric outputs (per inject_frac scores/deltas, per-condition
rf_results.csv, injection_log.json) plus combined scores_table.csv /
deltas_table.csv. The plot + best-model summary are produced by
shared_classifier_augmented_comparison.py (mirrors the classifier two-script
split).

Usage:
    python compute_classifier_augmented.py \
        --real-data real.h5ad --split-indices data_split_indices.npz \
        --cvae-data cvae.h5ad --diffusion-data diff.h5ad --gan-data gan.h5ad \
        --output-dir .../analysis/classifier_augmented \
        --name OSDR-403 --column-name Flight --group1 Flight --group2 Ground \
        --condition-cols "Flight,Strain,Sex,Age at Launch,Duration" \
        --pct-val-eval 0.25 --inject-fracs 0.5 --n-resamples-augment 10
"""

import argparse
import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
)
import warnings

warnings.filterwarnings('ignore')

# Reuse the exact stratified splitter used everywhere else in the pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_splitter import stratified_split  # noqa: E402

MODEL_ORDER = ['cvae', 'diffusion', 'gan']
METRIC_ORDER = ['accuracy', 'precision', 'recall', 'f1',
                'f1_macro', 'balanced_accuracy', 'roc_auc', 'pr_auc']


# ---------------------------------------------------------------------------
# Shared helpers (RF params + metric block copied from compute_classifier.py so
# in-fold behavior is byte-identical to the GO classifier)
# ---------------------------------------------------------------------------

def parse_class_weight(value):
    """'', 'none', 'null' -> None; otherwise pass through (e.g. 'balanced_subsample')."""
    if value is None:
        return None
    v = str(value).strip()
    if v.lower() in ('', 'none', 'null'):
        return None
    return v


def build_rf_params(n_estimators, max_depth, min_samples_split,
                    min_samples_leaf, class_weight, random_state):
    return {
        'n_estimators': n_estimators,
        'max_depth': max_depth if (max_depth is not None and max_depth > 0) else None,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'class_weight': parse_class_weight(class_weight),
        'random_state': random_state,
        'n_jobs': -1,
    }


def score_predictions(y_true, y_pred, y_proba):
    """The exact 8-metric block from compute_classifier.py."""
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
    else:
        roc_auc = np.nan
        pr_auc = np.nan
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
    }


def to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def strata_key(obs, cols):
    """Combined condition key, identical to stratified_split's keying."""
    return obs[cols].astype(str).agg('_'.join, axis=1).values


def allocate_counts(stratum_sizes, n_total):
    """Allocate n_total across strata proportional to their val share; largest
    remainder method so the integer counts sum exactly to n_total."""
    N = sum(stratum_sizes.values())
    if N == 0 or n_total <= 0:
        return {s: 0 for s in stratum_sizes}
    raw = {s: n_total * c / N for s, c in stratum_sizes.items()}
    out = {s: int(np.floor(v)) for s, v in raw.items()}
    remainder = n_total - sum(out.values())
    order = sorted(stratum_sizes.keys(), key=lambda s: raw[s] - out[s], reverse=True)
    for i in range(remainder):
        out[order[i % len(order)]] += 1
    return out


def draw_by_stratum(positions_by_stratum, alloc, rng, replace, fallback_pool):
    """Draw `alloc[s]` row-positions from each stratum. Without replacement when
    replace=False, falling back to with-replacement for any stratum that is
    empty or too small (returns whether that fallback was triggered)."""
    chosen = []
    fallback_used = False
    for s, k in alloc.items():
        if k <= 0:
            continue
        pos = positions_by_stratum.get(s)
        if pos is None or len(pos) == 0:
            pos = fallback_pool
            fallback_used = True
        if replace:
            sel = rng.choice(pos, size=k, replace=True)
        elif len(pos) < k:
            sel = rng.choice(pos, size=k, replace=True)
            fallback_used = True
        else:
            sel = rng.choice(pos, size=k, replace=False)
        chosen.append(np.asarray(sel))
    if not chosen:
        return np.array([], dtype=int), fallback_used
    return np.concatenate(chosen).astype(int), fallback_used


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def fit_eval(X_train, y_train, X_eval, y_eval, rf_params):
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_eval)
    y_proba = rf.predict_proba(X_eval)[:, 1]
    return score_predictions(y_eval, y_pred, y_proba), rf.feature_importances_


def aggregate(metric_dicts, importances):
    mdf = pd.DataFrame(metric_dicts)
    mean = {m: float(mdf[m].mean()) for m in METRIC_ORDER}
    std = {m: float(mdf[m].std()) for m in METRIC_ORDER}
    imp = np.mean(np.array(importances), axis=0)
    return mean, std, imp


def write_rf_results(importances, gene_names, n_genes, out_dir):
    top = np.argsort(importances)[::-1][:n_genes]
    df = pd.DataFrame({
        'gene': gene_names[top],
        'mean_importance': importances[top],
        'std_importance': np.zeros(len(top)),  # std across resamples not tracked per-gene
        'rank': np.arange(1, len(top) + 1),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "rf_results.csv", index=False)
    return df


def run_dataset(real_adata, synth_adatas, column_name, group1, group2,
                condition_cols, pct_val_eval, inject_fracs, n_resamples_augment,
                split_path, n_genes, rf_params, random_state, output_dir,
                augment_pool="train_eval_same_as_model"):
    """Run the full augmentation experiment for one dataset.

    augment_pool selects where the real train pool and eval set come from:
      "train_eval_same_as_model" (DEFAULT): rf_train_pool = the FULL TRAIN split,
          rf_eval = the ENTIRE VAL split (the same train/val the generative models
          used). Dosage + injection strata are based on the TRAIN pool. Uses all
          the real training data and keeps eval leakage-proof (val is never in
          the classifier's training set and is identical across conditions).
      "only_val" (original behavior): carve BOTH rf_train_pool and the fixed
          rf_eval out of the VALIDATION split (a stratified sub-split of val).
          Dosage + injection strata match the full val distribution (scarce-data
          regime).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- choose the real train pool + eval set per augment_pool mode ---
    from utils.data_splitter import load_split_indices
    train_idx, val_idx = load_split_indices(str(split_path))
    g1, g2 = str(group1), str(group2)
    real_adata.obs[column_name] = real_adata.obs[column_name].astype(str)

    def _two_class(a):
        return a[a.obs[column_name].isin([g1, g2])].copy()

    if augment_pool == "train_eval_same_as_model":
        pool_adata = _two_class(real_adata[train_idx])   # rf_train = full train split
        eval_adata = _two_class(real_adata[val_idx])     # rf_eval  = entire val split
        ref_adata = pool_adata                           # dosage/strata from train
    elif augment_pool == "only_val":
        val_adata = _two_class(real_adata[val_idx])
        strat0 = [c for c in condition_cols if c in val_adata.obs.columns]
        if column_name not in strat0:
            strat0 = [column_name] + strat0
        pool_pos, eval_pos = stratified_split(
            val_adata, strat0, val_ratio=pct_val_eval, seed=random_state)
        eval_adata = val_adata[eval_pos]
        pool_adata = val_adata[pool_pos]
        ref_adata = val_adata                            # dosage/strata from full val
    else:
        raise ValueError(f"Unknown augment_pool '{augment_pool}' "
                         f"(expected 'only_val' or 'train_eval_same_as_model')")

    gene_names = pool_adata.var_names.values
    strat_cols = [c for c in condition_cols if c in ref_adata.obs.columns]
    if column_name not in strat_cols:
        strat_cols = [column_name] + strat_cols

    X_eval = to_dense(eval_adata.X)
    y_eval = (eval_adata.obs[column_name].values == g2).astype(int)
    X_pool = to_dense(pool_adata.X)
    y_pool = (pool_adata.obs[column_name].values == g2).astype(int)

    # Dosage base + injection strata come from ref_adata (val for only_val, the
    # train pool for train_eval_same_as_model). Kept under the original names so
    # the rest of the routine is unchanged; in only_val these are byte-identical.
    N_val = ref_adata.n_obs
    print(f"  augment_pool={augment_pool} | stratifying on: {strat_cols}")
    print(f"  dosage base (ref) cells: {N_val}")
    print(f"  rf_eval: {eval_adata.n_obs} cells | rf_train_pool: {pool_adata.n_obs} cells")

    ref_keys = strata_key(ref_adata.obs, strat_cols)
    val_stratum_sizes = OrderedDict(pd.Series(ref_keys).value_counts().sort_index().items())
    pool_keys = strata_key(pool_adata.obs, strat_cols)
    pool_positions = {s: np.where(pool_keys == s)[0] for s in val_stratum_sizes}
    eval_keys = strata_key(eval_adata.obs, strat_cols)

    # synthetic pools: filter to the 2 classes, align genes, index by stratum
    synth = {}
    for model, sa in synth_adatas.items():
        if sa is None:
            continue
        if list(sa.var_names) != list(val_adata.var_names):
            raise ValueError(f"[{model}] var_names differ from real; cannot augment.")
        sa.obs[column_name] = sa.obs[column_name].astype(str)
        sa = sa[sa.obs[column_name].isin([g1, g2])].copy()
        scols = [c for c in strat_cols if c in sa.obs.columns]
        if scols != strat_cols:
            print(f"  [WARN] {model} missing cols {set(strat_cols) - set(scols)}; "
                  f"stratifying synthetic on {scols}")
        skeys = strata_key(sa.obs, scols) if scols else np.array([''] * sa.n_obs)
        synth[model] = {
            'X': sa.X,  # keep sparse; densify only the sampled rows (large pools)
            'y': (sa.obs[column_name].values == g2).astype(int),
            'keys': skeys,
            'pos_by_stratum': {s: np.where(skeys == s)[0] for s in np.unique(skeys)},
            'scols': scols,
        }
        print(f"  {model} pool: {sa.n_obs} cells")

    pool_all_pos = np.arange(pool_adata.n_obs)

    # --- baseline 'real' condition (X-independent): compute once ---
    real_metric_dicts, real_imps = [], []
    for r in range(n_resamples_augment):
        p = dict(rf_params); p['random_state'] = random_state + r
        m, imp = fit_eval(X_pool, y_pool, X_eval, y_eval, p)
        real_metric_dicts.append(m); real_imps.append(imp)
    real_mean, real_std, real_imp = aggregate(real_metric_dicts, real_imps)
    write_rf_results(real_imp, gene_names, n_genes, out / "real")

    all_scores = []   # rows: inject_frac, condition, metric, mean, std
    all_deltas = []   # rows: inject_frac, model, metric, delta_vs_dup/std, delta_vs_real/std

    def record_scores(frac, cond, mean, std):
        for mt in METRIC_ORDER:
            all_scores.append({'inject_frac': frac, 'condition': cond, 'metric': mt,
                               'mean': mean[mt], 'std': std[mt]})

    # --- per inject fraction ---
    for frac in inject_fracs:
        frac_dir = out / f"frac_{frac}"
        frac_dir.mkdir(parents=True, exist_ok=True)
        n_add_total = int(round(frac * N_val))
        alloc = allocate_counts(val_stratum_sizes, n_add_total)
        print(f"\n  === inject_frac={frac}: adding {n_add_total} cells "
              f"across {len([s for s in alloc if alloc[s] > 0])} strata ===")

        injection_log = {
            'inject_frac': frac,
            'augment_pool': augment_pool,
            'dosage_base_n': int(N_val),
            'N_val': int(N_val),
            'n_add_total': int(n_add_total),
            'pct_val_eval': pct_val_eval,
            'rf_eval_n': int(eval_adata.n_obs),
            'rf_train_pool_n': int(pool_adata.n_obs),
            'strat_cols': strat_cols,
            'n_resamples_augment': n_resamples_augment,
            'random_state': random_state,
            'strata_val_sizes': {str(k): int(v) for k, v in val_stratum_sizes.items()},
            'strata_eval_sizes': {str(k): int(v) for k, v in
                                  pd.Series(eval_keys).value_counts().items()},
            'n_add_per_stratum': {str(k): int(v) for k, v in alloc.items()},
            'replacement_used': {},
        }

        # record 'real' into this frac's table (shared baseline)
        record_scores(frac, 'real', real_mean, real_std)

        # ---- real_dup ----
        dup_metric_dicts, dup_imps, dup_fallback = [], [], False
        for r in range(n_resamples_augment):
            rng = np.random.RandomState(random_state + 1000 + r)
            add_pos, fb = draw_by_stratum(pool_positions, alloc, rng,
                                          replace=True, fallback_pool=pool_all_pos)
            dup_fallback = dup_fallback or fb
            X_tr = np.vstack([X_pool, X_pool[add_pos]])
            y_tr = np.concatenate([y_pool, y_pool[add_pos]])
            p = dict(rf_params); p['random_state'] = random_state + r
            m, imp = fit_eval(X_tr, y_tr, X_eval, y_eval, p)
            dup_metric_dicts.append(m); dup_imps.append(imp)
        dup_mean, dup_std, dup_imp = aggregate(dup_metric_dicts, dup_imps)
        write_rf_results(dup_imp, gene_names, n_genes, frac_dir / "real_dup")
        record_scores(frac, 'real_dup', dup_mean, dup_std)
        injection_log['replacement_used']['real_dup'] = True  # dup is always with-replacement

        # ---- real_synth_<model> ----
        for model in MODEL_ORDER:
            if model not in synth:
                continue
            s = synth[model]
            synth_metric_dicts, synth_imps, syn_fb = [], [], False
            for r in range(n_resamples_augment):
                rng = np.random.RandomState(random_state + 2000 + r)
                add_pos, fb = draw_by_stratum(s['pos_by_stratum'], alloc, rng,
                                              replace=False,
                                              fallback_pool=np.arange(len(s['y'])))
                syn_fb = syn_fb or fb
                X_tr = np.vstack([X_pool, to_dense(s['X'][add_pos])])
                y_tr = np.concatenate([y_pool, s['y'][add_pos]])
                p = dict(rf_params); p['random_state'] = random_state + r
                m, imp = fit_eval(X_tr, y_tr, X_eval, y_eval, p)
                synth_metric_dicts.append(m); synth_imps.append(imp)
            sy_mean, sy_std, sy_imp = aggregate(synth_metric_dicts, synth_imps)
            write_rf_results(sy_imp, gene_names, n_genes, frac_dir / model)
            record_scores(frac, model, sy_mean, sy_std)
            injection_log['replacement_used'][model] = bool(syn_fb)

            # deltas (synth vs dup, synth vs real); std via quadrature
            for mt in METRIC_ORDER:
                d_dup = sy_mean[mt] - dup_mean[mt]
                d_real = sy_mean[mt] - real_mean[mt]
                s_dup = float(np.sqrt(sy_std[mt] ** 2 + dup_std[mt] ** 2))
                s_real = float(np.sqrt(sy_std[mt] ** 2 + real_std[mt] ** 2))
                all_deltas.append({
                    'inject_frac': frac, 'model': model, 'metric': mt,
                    'delta_vs_dup': d_dup, 'delta_vs_dup_std': s_dup,
                    'delta_vs_real': d_real, 'delta_vs_real_std': s_real,
                })
            print(f"    {model:9s} bal_acc={sy_mean['balanced_accuracy']:.3f} "
                  f"(dup {dup_mean['balanced_accuracy']:.3f}, real {real_mean['balanced_accuracy']:.3f}) "
                  f"| Δvs_dup={sy_mean['balanced_accuracy'] - dup_mean['balanced_accuracy']:+.3f}")

        # per-frac scores + injection log
        pd.DataFrame([r for r in all_scores if r['inject_frac'] == frac]).to_csv(
            frac_dir / "scores.csv", index=False)
        pd.DataFrame([r for r in all_deltas if r['inject_frac'] == frac]).to_csv(
            frac_dir / "deltas.csv", index=False)
        with open(frac_dir / "injection_log.json", 'w') as f:
            json.dump(injection_log, f, indent=2)

    # combined tables (compute owns the numbers; the shared script plots them)
    pd.DataFrame(all_scores).to_csv(out / "scores_table.csv", index=False)
    pd.DataFrame(all_deltas).to_csv(out / "deltas_table.csv", index=False)
    print(f"\n[OK] Wrote scores_table.csv and deltas_table.csv to {out}")
    return all_scores, all_deltas


def parse_args():
    p = argparse.ArgumentParser(description='Synthetic-data augmentation classifier test')
    p.add_argument('--real-data', required=True, help='Processed real h5ad')
    p.add_argument('--split-indices', required=True, help='data_split_indices.npz (train/val)')
    p.add_argument('--cvae-data', default=None)
    p.add_argument('--diffusion-data', default=None)
    p.add_argument('--gan-data', default=None)
    p.add_argument('--output-dir', required=True, help='analysis/classifier_augmented dir')
    p.add_argument('--name', default='Dataset')
    p.add_argument('--column-name', required=True)
    p.add_argument('--group1', required=True)
    p.add_argument('--group2', required=True)
    p.add_argument('--condition-cols', required=True,
                   help='Comma-separated condition columns for stratification')
    p.add_argument('--augment-pool', default='train_eval_same_as_model',
                   choices=['only_val', 'train_eval_same_as_model'],
                   help="train_eval_same_as_model (default): rf_train_pool = full train "
                        "split, rf_eval = entire val split (dosage based on the train pool). "
                        "only_val: rf_train_pool + rf_eval are both carved from the "
                        "validation split.")
    p.add_argument('--pct-val-eval', type=float, default=0.25)
    p.add_argument('--inject-fracs', default='0.5',
                   help='Comma-separated fractions of val size to inject (sweep)')
    p.add_argument('--n-resamples-augment', type=int, default=10)
    # RF passthrough (inherited from the classifier config)
    p.add_argument('--n-genes', type=int, default=50)
    p.add_argument('--n-estimators', type=int, default=100)
    p.add_argument('--max-depth', type=int, default=10)
    p.add_argument('--min-samples-split', type=int, default=5)
    p.add_argument('--min-samples-leaf', type=int, default=2)
    p.add_argument('--class-weight', default=None)
    p.add_argument('--random-state', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print(f"AUGMENTATION CLASSIFIER: {args.name}")
    print("=" * 60)

    condition_cols = [c.strip() for c in args.condition_cols.split(',') if c.strip()]
    inject_fracs = [float(x) for x in str(args.inject_fracs).split(',') if str(x).strip()]
    rf_params = build_rf_params(args.n_estimators, args.max_depth, args.min_samples_split,
                                args.min_samples_leaf, args.class_weight, args.random_state)
    print(f"  RF: {rf_params}")
    print(f"  augment_pool={args.augment_pool} pct_val_eval={args.pct_val_eval} "
          f"inject_fracs={inject_fracs} n_resamples_augment={args.n_resamples_augment}")

    print("\nLoading real data...")
    real_adata = sc.read_h5ad(args.real_data)
    real_adata.obs_names_make_unique()

    synth_adatas = {}
    for model, path in [('cvae', args.cvae_data), ('diffusion', args.diffusion_data),
                        ('gan', args.gan_data)]:
        if path:
            print(f"Loading {model} synthetic: {Path(path).name}")
            sa = sc.read_h5ad(path); sa.obs_names_make_unique()
            synth_adatas[model] = sa
    if not synth_adatas:
        print("[ERROR] No synthetic data provided; nothing to augment.")
        sys.exit(1)

    try:
        run_dataset(
            real_adata, synth_adatas,
            column_name=args.column_name, group1=args.group1, group2=args.group2,
            condition_cols=condition_cols, pct_val_eval=args.pct_val_eval,
            inject_fracs=inject_fracs, n_resamples_augment=args.n_resamples_augment,
            split_path=args.split_indices, n_genes=args.n_genes, rf_params=rf_params,
            random_state=args.random_state, output_dir=args.output_dir,
            augment_pool=args.augment_pool,
        )
    except Exception as e:
        print(f"[ERROR] Augmentation run failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
