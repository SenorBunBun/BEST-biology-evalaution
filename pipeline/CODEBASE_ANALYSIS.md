# Pipeline Codebase Comprehensive Analysis

## Overview

This is a **scRNA-seq synthetic data generation pipeline** that:
1. **Trains 3 generative models** (CVAE, Diffusion, GAN) on single-cell RNA sequencing data
2. **Generates synthetic data** using trained models
3. **Evaluates quality** via DEG analysis, classifiers, UMAP, and scientific metrics

All orchestrated by a **single command**: `python driver.py --config dataset_configs/{dataset}.json`

---

## Directory Structure

```
pipeline/
├── driver.py                      # MAIN ORCHESTRATOR (957 lines)
├── config_loader.py               # Config loading with defaults
├── utils/
│   └── model_tracker.py           # Checkpoint tracking system
├── preprocessing/
│   └── preprocess.py              # Data preprocessing script
├── dataset_configs/               # JSON configuration files
│   ├── cancer.json
│   ├── spaceflight.json
│   └── osdr-{402,403,404,405,40X}.json
├── models/
│   ├── CVAE/                      # PyTorch CVAE model
│   ├── Diffusion/                 # PyTorch VAE + Diffusion
│   └── GAN/                       # TensorFlow WGAN-GP
├── analysis/                      # Evaluation scripts
├── outputs/                       # Generated outputs (runtime)
├── real_datasets/                 # Input data
└── original_rele_files/           # ARCHIVED/BACKUP FILES
```

---

## Pipeline Flow (Single Command Execution)

```
python driver.py --config dataset_configs/cancer.json

┌─────────────────────────────────────────────────────────────────┐
│ 1. LOAD CONFIG                                                   │
│    └── config_loader.py loads JSON + merges with DEFAULTS        │
├─────────────────────────────────────────────────────────────────┤
│ 2. PREPROCESSING (if enabled)                                   │
│    └── preprocessing/preprocess.py                               │
│        - Gene filtering (min_cells)                              │
│        - Sparsity filtering (>99.7% zeros removed)               │
│        - HVG selection (2000 genes)                              │
│        - Normalization (normalize_total + log1p)                 │
├─────────────────────────────────────────────────────────────────┤
│ 3. TRAINING (Sequential)                                         │
│    ├── models/CVAE/train.py                                      │
│    │   └── Uses Enhanced_Vanilla_CVAE_Categorical class          │
│    │   └── Saves: best_model.pt + latest_model.json              │
│    │                                                             │
│    ├── models/Diffusion/train.py (TWO-PHASE)                     │
│    │   └── Phase 1: Train VAE (vae_diffusion.py)                 │
│    │   └── Phase 2: Train Diffusion (diffusion_model_cfg.py)     │
│    │   └── Saves: vae_model.pt + best_model.pt + condition_config│
│    │                                                             │
│    └── models/GAN/train.py                                       │
│        └── TensorFlow WGAN-GP (generator.py + critic.py)         │
│        └── Saves: best_model/ directory + condition_config.json  │
├─────────────────────────────────────────────────────────────────┤
│ 4. GENERATION (Sequential)                                       │
│    ├── models/CVAE/generate.py                                   │
│    ├── models/Diffusion/generate.py                              │
│    └── models/GAN/generate.py                                    │
│    └── Outputs: {model}_{dataset}_{timestamp}.h5ad + _meta.json  │
├─────────────────────────────────────────────────────────────────┤
│ 5. ANALYSIS                                                      │
│    ├── analysis/run_deg_analysis.py (per model)                  │
│    ├── analysis/shared_deg_comparison.py (combined plot)         │
│    ├── analysis/run_classifier_analysis.py (per model)           │
│    ├── analysis/run_combined_classifier_analysis.py              │
│    ├── analysis/run_umap_analysis.py                             │
│    └── analysis/run_metrics_analysis.py                          │
│    └── Outputs: plots + CSV/JSON metrics in analysis/ directory  │
└─────────────────────────────────────────────────────────────────┘
```

---

## CLI Arguments

```bash
--config       Path to dataset config JSON (REQUIRED)
--stage        all | preprocess | train | generate | analyze (default: all)
--model        all | cvae | vae | diffusion | gan (default: all)
--checkpoint   Override checkpoint path for generation
--dry-run      Print commands without executing
--fail-fast    Stop on first error
--show-plots   Display plots interactively
```

**Examples:**
```bash
# Full pipeline
python driver.py --config dataset_configs/cancer.json

# Train only CVAE
python driver.py --config dataset_configs/cancer.json --stage train --model cvae

# Generate only (uses latest models)
python driver.py --config dataset_configs/cancer.json --stage generate

# Analyze existing synthetic data
python driver.py --config dataset_configs/cancer.json --stage analyze

# Dry run (preview commands)
python driver.py --config dataset_configs/cancer.json --dry-run
```

---

## Configuration Structure

Key sections in `dataset_configs/{name}.json`:

```json
{
  "dataset_name": "cancer",
  "paths": {
    "raw_data": "real_datasets/cancer_raw.h5ad",
    "processed_data": "outputs/cancer/preprocessed_data.h5ad",
    "output_dir": "outputs/cancer"
  },
  "preprocessing": {
    "enabled": true,
    "min_cells": 3,
    "sparsity_threshold": 0.997,
    "n_hvg": 2000,
    "normalize": true
  },
  "conditions": {
    "columns": ["annotation_major", "tissue"],
    "types": { "annotation_major": "categorical", "tissue": "categorical" },
    "primary": {
      "column": "tissue",
      "group1": "metastasis normal",
      "group2": "metastasis tumor",
      "group1_display_name": "Normal",
      "group2_display_name": "Tumor"
    }
  },
  "models": {
    "cvae": { "enabled": true, "latent_size": 32, ... },
    "diffusion": { "enabled": true, "vae_latent_size": 128, ... },
    "gan": { "enabled": true, "latent_dim": 128, ... }
  },
  "analysis": {
    "deg": { "enabled": true, "n_genes": 50 },
    "classifier": { "enabled": true, "n_genes": 50 },
    "umap": { "enabled": true },
    "metrics": { "enabled": true }
  }
}
```

---

## Key Files Reference

### Core Pipeline Files (USED IN MAIN LOOP)

| File | Purpose | Called By |
|------|---------|-----------|
| `driver.py` | Main orchestrator | CLI entry point |
| `config_loader.py` | Load/validate configs | driver.py |
| `utils/model_tracker.py` | Checkpoint management | driver.py, train scripts |
| `preprocessing/preprocess.py` | Data preprocessing | driver.py |

### CVAE Model (PyTorch)

| File | Purpose | Called By |
|------|---------|-----------|
| `models/CVAE/train.py` | CVAE training | driver.py |
| `models/CVAE/generate.py` | Synthetic data generation | driver.py |
| `models/CVAE/unbatch_vanilla_cvae_relu_qol_categorical.py` | Model architecture | train.py, generate.py |

### Diffusion Model (PyTorch, Two-Phase)

| File | Purpose | Called By |
|------|---------|-----------|
| `models/Diffusion/train.py` | VAE + Diffusion training | driver.py |
| `models/Diffusion/generate.py` | Synthetic data generation | driver.py |
| `models/Diffusion/vae_diffusion.py` | DiffusionIntegrationVAE class | train.py, generate.py |
| `models/Diffusion/diffusion_model_cfg.py` | ClassifierFreeDiffusionModel | train.py, generate.py |
| `models/Diffusion/diffusion_training_utils.py` | Training utilities | train.py |
| `models/Diffusion/diffusion_generation_utils.py` | Generation utilities | generate.py |

### GAN Model (TensorFlow)

| File | Purpose | Called By |
|------|---------|-----------|
| `models/GAN/train.py` | WGAN-GP training | driver.py |
| `models/GAN/generate.py` | Synthetic data generation | driver.py |
| `models/GAN/models/generator.py` | MultiConditionalGenerator | train.py, generate.py |
| `models/GAN/models/critic.py` | MultiConditionalCritic | train.py |
| `models/GAN/utils/condition_processor_generic.py` | Generic condition handling | train.py, generate.py |

### Analysis Scripts

| File | Purpose | Called By |
|------|---------|-----------|
| `analysis/run_deg_analysis.py` | DEG analysis runner | driver.py |
| `analysis/run_classifier_analysis.py` | RF classifier runner | driver.py |
| `analysis/run_combined_classifier_analysis.py` | Combined classifier plot | driver.py |
| `analysis/run_umap_analysis.py` | UMAP visualization | driver.py |
| `analysis/run_metrics_analysis.py` | Scientific metrics | driver.py |
| `analysis/shared_deg_comparison.py` | DEG overlap bar chart | driver.py |
| `analysis/diff_exp_gene_pipeline_v2.py` | DEG logic (DEGCrossComparison) | run_deg_analysis.py |
| `analysis/rf_classifier_gene_pipeline.py` | RF logic (RFClassifierComparison) | run_classifier_analysis.py |

---

## Model Tracking System

Each model type has a `latest_model.json` file:

```
outputs/{dataset}/trained_models/{model}/
├── latest_model.json              # Points to best model
├── run_YYYYMMDD_HHMMSS/           # Timestamped run
│   ├── best_model.pt              # Best checkpoint
│   ├── training_metrics.json      # Loss curves
│   ├── checkpoints/               # Epoch checkpoints
│   └── run_config.json            # Config snapshot
└── run_YYYYMMDD_HHMMSS/           # Another run
```

The `use_checkpoint` config option supports:
- `"latest"` - Use most recent training run
- `"path/to/run_XXX/best_model.pt"` - Specific checkpoint

---

## Output Structure

```
outputs/{dataset_name}/
├── trained_models/
│   ├── cvae/
│   ├── diffusion/
│   └── gan/
├── synthetic_data/
│   ├── cvae_{dataset}_{timestamp}.h5ad
│   ├── cvae_{dataset}_{timestamp}_meta.json
│   ├── diffusion_...
│   └── gan_...
└── analysis/
    ├── deg/
    ├── classifier/
    ├── umap/
    └── metrics/
```

---

## Three Model Architectures

### 1. CVAE (Conditional Variational Autoencoder)
- **Framework:** PyTorch
- **Architecture:** Encoder → Latent (32D) → Decoder with conditions
- **Key Params:** `latent_size=32`, `hidden_dims=[800,400]`, `beta=0.05`
- **Training:** Standard VAE with KL annealing, early stopping

### 2. Diffusion (VAE + Latent Diffusion)
- **Framework:** PyTorch
- **Architecture:**
  - Phase 1: VAE compresses genes → 128D latent
  - Phase 2: Diffusion model operates in latent space
- **Key Params:** `vae_latent_size=128`, `num_timesteps=1000`, `classifier_free_prob=0.1`
- **Training:** Two-phase, classifier-free guidance enabled

### 3. GAN (Wasserstein GAN with Gradient Penalty)
- **Framework:** TensorFlow
- **Architecture:** Generator + Critic with conditional batch norm
- **Key Params:** `latent_dim=128`, `lambda_gp=15.0`, `critic_iters=4`
- **Training:** Step-based (epochs converted to steps)

---

## Analysis Metrics

1. **Differential Expression (DEG):** Wilcoxon test, top N significant genes
2. **RF Classifier:** Phenotypically predictive genes via feature importance
3. **UMAP:** Real vs synthetic embedding visualization
4. **Scientific Metrics:** Wasserstein distance, MAD, cosine similarity

---

## FILES NOT USED IN MAIN PIPELINE LOOP

These files exist in the codebase but are **NOT** called by `driver.py`. They are either:
- Legacy/obsolete versions
- Cancer-specific standalone scripts (pre-pipeline era)
- Debug utilities
- Backup/archived files

### Entire Backup Directory (ARCHIVED)
```
original_rele_files/           # ENTIRE DIRECTORY - Archived copies
├── models/CVAE/               # Archived CVAE files
├── models/Diffusion/          # Archived Diffusion files
├── models/GAN/                # Archived GAN files
└── analysis/                  # Archived analysis files
```
**Status:** Backup of pre-pipeline files. Can be safely deleted.

---

### CVAE Obsolete Files

| File | Reason Unused |
|------|---------------|
| `models/CVAE/train_cvae_cancer.py` | Cancer-specific standalone trainer (pre-pipeline) |
| `models/CVAE/cvae_generate_dataset_cancer.py` | Cancer-specific standalone generator (pre-pipeline) |
| `models/CVAE/unbatch_vanilla_cvae_relu_qol.py` | Older non-categorical version; replaced by `*_categorical.py` |

---

### Diffusion Obsolete Files

| File | Reason Unused |
|------|---------------|
| `models/Diffusion/train_vae.py` | Convenience wrapper; use `train.py --vae-only` instead |
| `models/Diffusion/train_diffusion.py` | Older standalone diffusion training script |
| `models/Diffusion/diffusion_generate_cancer_data.py` | Cancer-specific standalone generator (pre-pipeline) |

---

### GAN Obsolete Files

| File | Reason Unused |
|------|---------------|
| `models/GAN/train_original.py` | Original/legacy training script |
| `models/GAN/generate_original.py` | Original/legacy generation script |
| `models/GAN/evaluate_conditions_cancer.py` | Cancer-specific evaluation utility |
| `models/GAN/utils/condition_processor.py` | Original processor; replaced by `*_generic.py` |
| `models/GAN/utils/condition_processor_cancer.py` | Cancer-specific processor |
| `models/GAN/parameters_gp_fix.json` | Old parameter file (not used by current train.py) |

---

### Analysis Standalone/Legacy Files

| File | Reason Unused |
|------|---------------|
| `analysis/full_umap.py` | Legacy standalone UMAP; use `run_umap_analysis.py` |
| `analysis/scientific_metrics_comparison_v2.py` | Standalone version; `run_metrics_analysis.py` is pipeline entry |
| `analysis/diff_exp_analysis_cancer_all.py` | Cancer-specific DEG analysis |
| `analysis/rf_classifier_analysis_run_cancer.py` | Cancer-specific classifier analysis |
| `analysis/osdr_612_phenotype_chart.py` | OSDR-612 dataset-specific plotting |
| `analysis/manual_plotting_shared_deg.py` | Manual plotting utility (not in loop) |
| `analysis/manual_plotting_phenotype.py` | Manual plotting utility (not in loop) |

---

### Duplicate Files in real_datasets/

| File | Reason Unused |
|------|---------------|
| `real_datasets/analysis/scientific_metrics_comparison_v2.py` | Duplicate |
| `real_datasets/analysis/diff_exp_gene_pipeline_v2.py` | Duplicate |
| `real_datasets/analysis/shared_deg_comparison.py` | Duplicate |
| `real_datasets/analysis/rf_classifier_analysis_run_cancer.py` | Duplicate |
| `real_datasets/analysis/rf_classifier_gene_pipeline.py` | Duplicate |
| `real_datasets/analysis/full_umap.py` | Duplicate |
| `real_datasets/analysis/diff_exp_analysis_cancer_all.py` | Duplicate |
| `real_datasets/unprocessed/process_osd_datasets.py` | Dataset processing utility (one-time use) |

---

### Pipeline Root Utilities (Optional/Standalone)

| File | Reason Unused | Keep? |
|------|---------------|-------|
| `debug_vae_weights.py` | Debugging utility | Optional |
| `run_all_datasets.py` | Batch runner for all configs | Optional - useful |
| `cross_dataset_comparison.py` | Cross-dataset analysis | Optional - useful |
| `phenotype_predictive_genes_bargraph.py` | Standalone plotting | Optional |

---

## Summary: Cleanup Candidates

**Safe to Delete (28+ files):**
1. Entire `original_rele_files/` directory (backup)
2. Entire `real_datasets/analysis/` directory (duplicates)
3. All `*_cancer.py` files in models/ (cancer-specific pre-pipeline scripts)
4. `*_original.py` files in GAN (legacy)
5. `condition_processor.py` and `condition_processor_cancer.py` (replaced by generic)

**Keep but Unused (5 files):**
1. `debug_vae_weights.py` - Debugging utility
2. `run_all_datasets.py` - Useful batch runner
3. `cross_dataset_comparison.py` - Useful for multi-dataset analysis
4. `analysis/manual_plotting_*.py` - Manual plotting utilities
5. `models/Diffusion/train_vae.py` - Convenience wrapper

---

## Verification Steps

To confirm which files are used, check `driver.py`:
- Lines 175-192: CVAE training → `models/CVAE/train.py`
- Lines 227-273: Diffusion training → `models/Diffusion/train.py`
- Lines 310-340: GAN training → `models/GAN/train.py`
- Lines 470-492: CVAE generation → `models/CVAE/generate.py`
- Lines 519-539: Diffusion generation → `models/Diffusion/generate.py`
- Lines 580-601: GAN generation → `models/GAN/generate.py`
- Lines 690-836: Analysis scripts → `analysis/run_*.py` + `shared_deg_comparison.py`
