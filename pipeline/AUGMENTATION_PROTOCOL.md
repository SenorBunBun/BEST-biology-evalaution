# Synthetic-Data Augmentation Testing Protocol (`classifier_augmented`)

## Purpose

Answer two questions, **per dataset**, in a leakage-free way:

1. **Does synthetic data improve downstream phenotype classification** when real training
   data is limited?
2. **Which generative model (CVAE / Diffusion / GAN) produces the most useful synthetic
   data** for that task?

Biological faithfulness of the synthetic data is **out of scope here** — it is already
measured separately by the fgsea GO-overlap analysis. This protocol measures *predictive
utility only*. (When reporting, a model that wins here but scored ~0% GO overlap should be
flagged as "predictively helpful but biologically incoherent" — that is a reporting note,
not part of this test.)

The classification target is always the config's primary phenotype
(`group1` vs `group2`, e.g. Flight vs Ground). Synthetic cells carry those same labels;
they never form their own class.

---

## Core design principle (why this is leakage-free)

- The only **real** cells used are the **validation cells** (`val_indices` from
  `data_split_indices.npz`). The generative models were trained on `train_indices` only,
  so **no val cell ever entered generative-model weight training** — val is genuinely
  held out and is the honest set to evaluate "does synthetic help predict unseen real
  cells."
- Within val, a **single fixed evaluation set (`rf_eval`) of entirely real cells** is
  split off **once per dataset** and **frozen**. It is drawn as a **fully stratified
  sub-sample that preserves val's complete joint condition distribution** (see Step 1) —
  not a 50/50 Flight/Ground balance. Every condition and every model for that dataset is
  scored on this identical `rf_eval`. It is distinct per dataset (the data differs) but
  constant across all classifiers of a dataset.
- **Synthetic (and duplicated) cells only ever enter the training set — never `rf_eval`.**
- Therefore the only thing that changes between conditions is what is added to training,
  and every comparison is on the same real yardstick.

---

## Parameters (config block `analysis.classifier_augmented`)

| Param | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Off by default → stage is a no-op unless turned on. |
| `pct_val_eval` | `0.25` | Fraction of val reserved as the fixed `rf_eval` set, drawn **fully stratified** on all `conditions.columns` (preserves val's joint distribution, not 50/50). |
| `pct_val_added_for_augment` | `0.5` | How many cells to add to training, as a fraction of val size, allocated across **val's full joint condition strata**. **Float or list** (e.g. `[0.25, 0.5, 1.0]`) to sweep; results are saved per value. |
| `n_resamples_augment` | `10` | Number of repeated augment-draw + RF-fit cycles per condition, for mean ± std. Each is a single RF fit (cheap), so 10 gives solid error bars. (Distinct from the classifier's CV `n_repeats` — see "The `n_resamples_augment` parameter" below.) |
| `n_genes` | inherit | Top-N genes reported in the importance output (for fgsea compatibility). |
| `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight`, `random_state` | inherit from `analysis.classifier` | RF settings — identical to the GO-enrichment classifier so results are comparable. |

All RF settings default to the same values used in the GO-overlap runs
(`class_weight="balanced_subsample"`, etc.). Absence of the block ⇒ disabled.

---

## Notation

- `N_val` = number of val cells (after filtering to the two primary classes).
- **strata** = the unique combinations of `conditions.columns` (the exact combined key
  `stratified_split` uses: `conditions.columns` joined as strings, continuous columns
  included as their stringified values — just like everywhere else in the pipeline). For
  404 the only varying columns are Flight and Age at Launch, so the strata are Flight×Age;
  Strain/Sex/Duration are constant.
- `p_s` = stratum `s`'s share of val.
- `pct_val_eval` → the eval set; `X = pct_val_added_for_augment` → the injected volume.

---

## Procedure (per dataset)

### Step 0 — Load & align
- Load the real processed h5ad; subset to `val_indices`; filter to `{group1, group2}` →
  `val_adata`.
- Load each model's generated pool (`<model>_bestmmd_<DATASET>_<ts>.h5ad`); filter to the
  two classes. **Assert** `var_names` equality with real and that the class labels match
  `group1`/`group2` exactly; error loudly otherwise.

### Step 1 — Build the fixed eval set (ONCE, seeded)
- Draw `rf_eval` as a **fully stratified sub-sample of val** that preserves val's complete
  joint condition distribution, using the SAME combined-condition key as the pipeline's
  `stratified_split` (all of `conditions.columns`). Concretely this is
  `stratified_split(val_adata, condition_cols, val_ratio=pct_val_eval, seed=…)`: the
  `val_ratio` portion becomes `rf_eval`, the remainder becomes `rf_train_pool`. So every
  stratum contributes `pct_val_eval` of its cells to eval and the rest to the train pool.
- This is **not** 50/50 Flight/Ground balancing — `rf_eval` mirrors val's real class *and*
  covariate proportions exactly, consistent with every other split in the pipeline. Both
  `rf_eval` and `rf_train_pool` therefore carry val's full joint structure.
- Both are **frozen** and reused for every condition and model of this dataset.
- Because `rf_eval` reflects val's real (often imbalanced) distribution, read results off
  the imbalance-robust metrics (`balanced_accuracy`, `f1_macro`, `pr_auc`, `recall`)
  rather than raw accuracy.

### Step 2 — For each augment fraction `X` in `pct_val_added_for_augment`
Compute the injection budget:
- `n_add_total = round(X * N_val)`
- Allocate `n_add_total` across **val's full joint condition strata**, proportional to each
  stratum's share: `n_add_s = round(p_s * n_add_total)` (fix rounding so `Σ n_add_s =
  n_add_total`). This reproduces val's complete distribution (Flight × all covariates) in
  the injected cells → **volume augmentation matched to val's full shape**, not just the
  Flight/Ground ratio and not rebalancing.

Define the **five conditions** (all share `rf_train_pool` as their real core and `rf_eval`
as their test set):
1. **`real`** — train on `rf_train_pool` only. The floor/reference.
2. **`real_dup`** — `rf_train_pool` + `n_add_s` cells per **stratum** sampled **with
   replacement** from the matching stratum of `rf_train_pool` (duplicated real). Controls
   for *pure volume*.
3. **`real_synth_cvae`** — `rf_train_pool` + `n_add_s` per **stratum** sampled **without
   replacement** from the matching stratum of the CVAE pool (fallback to with-replacement +
   warn if a stratum is exhausted).
4. **`real_synth_diffusion`** — same, from the Diffusion pool.
5. **`real_synth_gan`** — same, from the GAN pool.

`real_dup` and `real_synth_*` reach the **identical total size and full joint stratum
mix**, so they differ *only* by "duplicated real" vs "novel synthetic" — this is the
contrast that isolates synthetic value from volume.

### Step 3 — Repeated fit (variance)
For `r = 1 … n_resamples_augment`:
- Re-draw the condition's augment sample (per stratum) with seed `random_state + r` (for
  `real`, nothing is added; the RF seed still varies for a comparable variance estimate).
- Fit RF (all 2000 genes, inherited hyperparameters) on the condition's training matrix.
- Predict on the frozen `rf_eval`; compute all 8 metrics (accuracy, precision, recall, f1,
  f1_macro, balanced_accuracy, roc_auc, pr_auc) — `roc_auc`/`pr_auc` from `predict_proba`
  on `rf_eval` only.
- Record per-repeat metrics and the RF feature importances.

### Step 4 — Aggregate
- Per (`X`, condition): **mean ± std** over the `n_resamples_augment` for every metric.
- Average feature importances across repeats → top-`n_genes` list, written in the existing
  `rf_results.csv` format (`gene, mean_importance, std_importance, rank`) so fgsea can
  consume it unchanged.

### Step 5 — Deltas & model ranking
- **Primary signal:** `Δ_synth = (real_synth_m) − (real_dup)` per metric → "does model
  *m*'s synthetic beat simply duplicating real cells?"
- **Secondary:** `(real_synth_m) − (real)` → lift over the no-augmentation floor.
- Rank models by `Δ_synth` on the chosen headline metric (default **balanced_accuracy**),
  with its std, so "most effective model" is a sortable, variance-aware column — not a
  single-split coin flip.

---

## The `n_resamples_augment` parameter (vs. the classifier's `n_repeats`)

**Yes, the pipeline already repeats runs elsewhere** — but via a different mechanism, which
is why this needs its own name.

- The GO/phenotype classifier (`compute_classifier.py`) uses
  `RepeatedStratifiedKFold(n_splits=n_folds=5, n_repeats=3)` = **15 train/test splits**, and
  reports mean ± std across those folds. Its variance comes from *rotating the held-out test
  fold* and re-randomizing the fold assignment 3 times.
- This augmentation test **cannot** rotate folds: you specified a **single fixed `rf_eval`**
  shared across all conditions and models, so there is no fold to rotate. If we fit once, we
  get a single point per condition with no error bars — and on small datasets one unlucky
  augment draw can swing the metric by 0.1+.
- So the repeat here is over a **different random step**: redraw the random augment sample
  (which synthetic / duplicate cells get pulled) and re-seed the RF, `n_resamples_augment`
  times, then average. Same purpose (error bars for a defensible "which model wins"),
  different unit of randomness. Calling it `n_resamples_augment` instead of `n_repeats` keeps
  it from being confused with the classifier's CV-repeat count, which means something else.

**How many?** Default is `10`, which gives solid error bars. Each resample is a *single* RF
fit (not a 15-fold CV), so the cost is modest: per X value that's 5 conditions × 10 = 50
fits — the same order as the existing GO classifier's ~60 fits, which handled 214k cells
fine. The floor is: 3–4 already gives a usable mean and rough spread, but with that few
points the std is itself noisy, so ranking two close models isn't reliable; 10 makes the
spread trustworthy. A single fit (n=1) gives no variance and must not be used to rank
models. If processing time ever bites on the largest pooled set (many strata × large
`rf_train_pool` × a long X sweep), lowering to 4–5 there is a safe trade.

## Outputs (`pipeline/outputs/<FOLDER>/analysis/classifier_augmented/`)

```
classifier_augmented/
  frac_0.5/                         # one subdir per X in pct_val_added_for_augment
    scores.csv                      # condition × metric × {mean, std}
    deltas.csv                      # model × metric × {synth−dup, synth−real, std}
    real/            rf_results.csv
    real_dup/        rf_results.csv
    cvae/            rf_results.csv  # augmented importances (fgsea-compatible)
    diffusion/       rf_results.csv
    gan/             rf_results.csv
    injection_log.json              # N_val, rf_eval size & per-stratum composition,
                                    #   rf_train_pool size, n_add per stratum,
                                    #   with/without replacement used, all seeds
  scores_table.csv                  # all X stacked, carries an `inject_frac` column
  shared_classifier_augmented_summary.json / .csv   # best model per X by headline metric
  shared_classifier_augmented_comparison.png        # grouped bars (real / dup / 3 synth),
                                                    #   or Δ-vs-X curves when X is swept
```

All writes go to the **local** `pipeline/outputs/` mirror (the SSD is read-only).

**Reported metric:** all 8 are saved; the default comparison plot uses
`balanced_accuracy`. The headline metric for ranking can be chosen later from the saved
table without re-running.

---

## Guarantees & sanity gates

- `rf_eval` is real, balanced, frozen, and never appears in any training set, is never
  duplicated, and is never a source for synthetic sampling.
- Synthetic/duplicate cells appear **only** in training.
- **Sanity gate:** with `X = 0`, `real_dup` and `real_synth_*` must reproduce the `real`
  condition's metrics exactly (nothing injected).
- **Sanity gate (control logic):** if a *label-permuted* synthetic variant is ever added,
  its `Δ_synth` should be ≈ 0; a positive delta there would mean the benefit is row-count,
  not label signal. (Not in the default run; optional diagnostic.)

---

## Implementation notes & deviations from this spec (recorded during build)

The implementation follows the protocol above; these points differ from or
refine the original write-up:

1. **Two scripts, not one** (matches the existing classifier convention). The
   numeric work lives in `pipeline/analysis/compute_classifier_augmented.py`; the
   plot + best-model summary live in
   `pipeline/analysis/shared_classifier_augmented_comparison.py` — exactly
   mirroring `compute_classifier.py` + `shared_classifier_comparison.py`.
2. **One invocation does the whole dataset.** The compute script takes
   `--real-data`, `--split-indices`, and per-model `--cvae-data/--diffusion-data/
   --gan-data`, and runs every condition, model, and inject fraction in a single
   call. This is required so the fixed `rf_eval`, the `rf_train_pool`, and the
   `real` baseline are built **once** and shared across all conditions (separate
   per-model calls would redraw them). The `real` baseline is computed once and
   reused across all inject fractions.
3. **Config → CLI mapping.** `pct_val_added_for_augment` (float or list) is passed
   to the module's `--inject-fracs`. `headline_metric` was added as a config key
   (default `balanced_accuracy`) rather than chosen only at report time. All RF
   settings inherit from the `classifier` block via the driver (`aug override →
   classifier → default`), so the in-fold classifier matches the GO run.
4. **`rf_results.csv` schema kept** (`gene, mean_importance, std_importance,
   rank`) for fgsea compatibility, but `std_importance` is written as 0 —
   importances are averaged across resamples and per-gene spread across resamples
   is not tracked (not needed downstream; avoids holding every fold's full
   importance vector).
5. **Delta std by quadrature:** `delta_vs_dup_std = sqrt(std_synth² + std_dup²)`.
   The two conditions share the eval set so are not fully independent; this is a
   mildly conservative approximation, fine for flagging whether a gap exceeds
   noise.
6. **Replacement is fixed, not a config knob** (the planned `sample_replacement`
   key was dropped): duplicates are always with-replacement; synthetic is
   without-replacement with an automatic with-replacement fallback if a stratum is
   too small/empty. Whether the fallback fired is recorded per run in
   `injection_log.json` (`replacement_used`).
7. **Outputs** per inject fraction: `frac_<X>/{scores.csv, deltas.csv,
   injection_log.json, real_dup/rf_results.csv, <model>/rf_results.csv}` plus a
   shared `real/rf_results.csv`; top level: `scores_table.csv`, `deltas_table.csv`,
   `shared_classifier_augmented_summary.{json,csv}`, and the plots
   (`augmentation_sweep.png` when X is swept, `augmentation_comparison_frac<X>.png`
   per dosage). All written to the local `pipeline/outputs/` mirror.

## Known properties / caveats (by design, not bugs)

1. **Scarcity is dataset-dependent, not controlled.** `rf_train_pool` is whatever
   `(1 − pct_val_eval) · N_val` happens to be — genuinely scarce for small sets (402 ≈ a
   couple hundred cells) but large for pooled 352 (~tens of thousands). Expect augmentation
   lift to appear mainly on the small/scarce datasets and to null out on the large ones
   simply because the real baseline is already saturated. If a *controlled* data-efficiency
   curve is wanted later, add an optional `rf_train_size` subsample knob (e.g.
   25/50/100/200/full) that sits on top of this same machinery; the fixed `rf_eval` stays
   untouched.
2. **Synthetic can only encode what the generator learned from `train`.** It approximates
   the train manifold; if train and val differ, the measurable lift shrinks. That is part
   of what the test honestly captures, not a flaw.
3. **Predictive lift ≠ biological correctness.** A model can raise accuracy via
   class-correlated-but-artifactual features (observed: CVAE scored ROC-AUC 0.98 with 0% GO
   overlap). Always read this protocol's result alongside the GO-overlap result before
   calling a model "good."
4. **Thin strata.** Full joint stratification can create small strata (e.g. a rare
   Flight×Age×… combination). Proportional allocation then rounds some strata to 0 injected
   cells, and without-replacement synthetic draws can exhaust a thin stratum (→ warn and
   fall back to with-replacement for the remainder). The `rf_eval` stratified draw likewise
   needs ≥1 cell per occupied stratum; `stratified_split` already guarantees this (it takes
   `max(1, …)` per group). Record realized per-stratum counts in `injection_log.json` so any
   degenerate stratum is visible. For the current datasets most covariates are constant, so
   strata stay coarse (404 = Flight×Age), but this will matter on any dataset with several
   varying covariates.
```
