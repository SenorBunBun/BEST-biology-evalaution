# BEST_bioeval

Consolidated, working copy of the NASA AI4LS synthetic-data benchmarking project, assembled on the
Mac. It brings the generative pipeline (CVAE / Diffusion / GAN) together with the downstream analysis
(cross-dataset comparison, fGSEA) and the manuscript, in one self-contained folder with its own
environment.

---

## At a glance

| | |
|---|---|
| **Location** | `/Users/ljeytl/Arsh/Research/NASA/AI4LS/BEST_bioeval` |
| **Pipeline env** | `.venv_pipeline/` — Python 3.12 (arm64): PyTorch 2.4.1 (MPS), TensorFlow 2.17 (Keras 3), scanpy stack |
| **fGSEA env** | `fgsea/.venv_fgsea/` — separate, Python + system R (`/usr/local/bin/R`) |
| **Runs locally** | Small OSDR configs (osd_402/403/404/405) |
| **Needs SSD data** | `cancer`, `CRC_IMM_10X`, `osdr-352-612-613` (large h5ads left on the Windows SSD) |

---

## Layout

```
BEST_bioeval/
├── pipeline/                 # scRNA-seq synthetic-data pipeline (CVAE / Diffusion / GAN + analysis)
│                             #   copied from the Windows SSD, minus outputs & large datasets
├── Paper/                    # manuscript + figures
├── cross_analysis_scripts/   # cross-dataset analysis scripts
├── fgsea/                    # gene-set enrichment (fGSEA); self-contained .venv_fgsea + needs R
├── .venv_pipeline/           # Python 3.12 venv for pipeline / cross_analysis / paper
├── requirements.txt          # Mac-native (arm64) dependency pins for .venv_pipeline
├── .gitignore
└── README_BEST_bioeval.md    # this file
```

---

## What was and wasn't brought over

- **Pipeline code**: all of it — `driver.py`, `config_loader.py`, `models/`, `analysis/`,
  `preprocessing/`, `utils/`, `dataset_configs/`, the `cross_dataset_*` scripts, and the docs.
- **Datasets**: only the **small OSDRs** were copied — `osd_402/403/404/405.h5ad` and their
  `_preprocessed` versions (plus `real_datasets/unprocessed/`), ~130 MB total, under
  `pipeline/real_datasets/`.
- **Left on the Windows SSD** (too large to copy) at
  `/Volumes/Windows/Creations/Research/NASASLSTP25/AI4LS/AI4LS/CVAE/pipeline`:
  - the large h5ads — `CRC_IMM_10X` (~4.5 GB across raw + processed), `cancer` (~2.9 GB),
    `osdr_352_612_613` (~4.4 GB)
  - the ~25 GB `outputs/` folder
- **Dropped as obsolete** (verified — never referenced by `driver.py`): `original_rele_files/`
  and `real_datasets/analysis/`, plus the junk `nul` file and `__pycache__/`.

So the **small OSDR configs run as-is**; to run `cancer`, `CRC_IMM_10X`, or `osdr-352-612-613`,
copy the matching h5ad(s) from the SSD into `pipeline/real_datasets/` first.

---

## Environment

```bash
# Pipeline (CVAE/Diffusion = PyTorch, GAN = TensorFlow/Keras 3)
source .venv_pipeline/bin/activate         # Python 3.12, arm64
# …or call the interpreter directly, no activation needed:
./.venv_pipeline/bin/python pipeline/driver.py ...
```

Rebuild it anytime with:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv_pipeline
./.venv_pipeline/bin/pip install -r requirements.txt
```

**fGSEA is separate.** `fgsea/` keeps its own `.venv_fgsea` and relies on system R. The venv was
relocated with the folder, so its `activate` script still points at the old path — call
`fgsea/.venv_fgsea/bin/python` directly, or recreate the venv, rather than `source`-ing its
`activate`.

---

## Running the pipeline

```bash
cd pipeline
PY=../.venv_pipeline/bin/python

$PY driver.py --config dataset_configs/osdr-402.json --dry-run          # preview commands only
$PY driver.py --config dataset_configs/osdr-402.json                    # full pipeline
$PY driver.py --config dataset_configs/osdr-402.json --stage train --model cvae   # one stage/model
$PY driver.py --config dataset_configs/osdr-402.json --stage posthoc-mmd
```

Stages run in order: **preprocess → train (CVAE→Diffusion→GAN) → generate → posthoc-mmd → analyze**.
See `pipeline/README.md` for the full stage/config/CLI reference.

**Verified on this machine:** dry-run resolves all paths to the new location; a real CVAE train on
`osd_402` ran to completion (`best_model.pt`, checkpoints, loss curves). The GAN (TensorFlow) and
Diffusion stages are dependency-verified and wired, but have not yet been run end-to-end here.

---

## Train / validation split & data handling

All three models share **one stratified train/val split**, created once per dataset by
`utils/data_splitter.py` and saved to `outputs/{dataset}/data_split_indices.npz`, then passed to
every model via `--split-indices`. The split is stratified by condition combination (default 80/20),
so each condition group is proportionally represented in both sets.

**Do the models ever train on the validation data? No.** None of the three backpropagate or update
weights on val:

- **CVAE** — trains on the train loader; the val set is passed as `test_loader` and only ever
  evaluated inside `_validate_epoch` under `torch.no_grad()` (validation loss + early stopping).
- **Diffusion** — both phases (VAE, then diffusion) fit on the train loader and evaluate on the val
  loader only. *Note:* under the shared split, `test_idx = all − (train ∪ val)` is **empty**, so the
  diffusion "test" loader is a no-op when run through the driver — there is no separate held-out test
  set.
- **GAN** — the WGAN-GP critic/generator loop iterates over `train_dataset` only; `valid_dataset`
  (shuffle off) is touched solely inside the validation/MMD block.

**But validation is not an untouched hold-out.** It is used for model *selection*, not fitting:

1. **Early stopping / validation loss** (CVAE, Diffusion VAE phase).
2. **MMD kernel bandwidth (sigma)** is computed from the validation data.
3. **MMD-based early stopping and best-checkpoint selection** compare synthetic data against the
   validation set (all three models, when `mmd_eval.enabled`).

So the models never *learn* from val, but val does decide *when training stops* and *which checkpoint
is kept*, and it sets the MMD sigma — worth keeping in mind for any leakage-sensitive claim, since
there is no independent test set beyond this val set under the driver.
