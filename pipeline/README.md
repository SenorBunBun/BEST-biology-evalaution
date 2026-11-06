# scRNA-seq Synthetic Data Generation Pipeline

A modular, config-driven pipeline for training generative models (CVAE, Diffusion, GAN) on single-cell RNA sequencing data and generating synthetic datasets.

## Quick Start

```bash
# Run full pipeline
python driver.py --config dataset_configs/cancer.json

# Run specific stages
python driver.py --config dataset_configs/cancer.json --stage preprocess
python driver.py --config dataset_configs/cancer.json --stage train
python driver.py --config dataset_configs/cancer.json --stage train --model cvae
python driver.py --config dataset_configs/cancer.json --stage generate
python driver.py --config dataset_configs/cancer.json --stage analyze

# Post-hoc MMD evaluation across all checkpoints
python driver.py --config dataset_configs/cancer.json --stage posthoc-mmd

# Generate synthetic data from MMD-matched checkpoints
python generate_from_matched_mmd.py --config dataset_configs/cancer.json

# Dry run (preview commands without executing)
python driver.py --config dataset_configs/cancer.json --dry-run
```

## Directory Structure

```
pipeline/
├── driver.py                      # Main orchestrator CLI
├── config_loader.py               # Config loading with defaults
├── utils/
│   └── model_tracker.py           # Tracks latest models for each type
├── preprocessing/
│   └── preprocess.py              # Data preprocessing script
├── dataset_configs/
│   ├── cancer.json                # Cancer dataset config
│   └── spaceflight.json           # Spaceflight dataset config
├── models/
│   ├── CVAE/                      # CVAE model training & generation
│   │   ├── train.py
│   │   └── generate.py
│   ├── Diffusion/                 # Diffusion model training & generation
│   │   ├── train.py
│   │   └── generate.py
│   └── GAN/                       # GAN model training & generation
│       ├── train.py
│       └── generate.py
├── analysis/                      # Analysis scripts
│   ├── run_deg_analysis.py
│   ├── run_classifier_analysis.py
│   ├── shared_deg_comparison.py
│   ├── plot_mmd_history.py
│   └── ... (other analysis modules)
├── utils/
│   ├── model_tracker.py           # Tracks latest models for each type
│   ├── data_splitter.py           # Stratified train/val splitting
│   ├── mmd.py                     # MMD computation (includes stratified sigma)
│   ├── mmd_early_stopping.py      # MMD-based early stopping tracker
│   └── posthoc_mmd.py             # Post-hoc MMD across checkpoints
├── generate_from_matched_mmd.py   # Generate from MMD-matched checkpoints
└── real_datasets/                 # Input data (not tracked in git)
```

## Configuration

Config files are JSON with the following structure:

```json
{
  "dataset_name": "cancer",
  "description": "Description of dataset",

  "paths": {
    "raw_data": "real_datasets/cancer_raw.h5ad",
    "processed_data": "outputs/cancer/preprocessed_data.h5ad",
    "output_dir": "outputs/cancer"
  },

  "conditions": {
    "columns": ["annotation_major", "tissue"],
    "types": {
      "annotation_major": "categorical",
      "tissue": "categorical"
    },
    "primary": {
      "column": "tissue",
      "group1": "metastasis normal",
      "group2": "metastasis tumor",
      "group1_display_name": "Normal",
      "group2_display_name": "Tumor"
    }
  },

  "models": {
    "cvae": {
      "enabled": true,
      "use_checkpoint": "latest",
      "latent_size": 32,
      "hidden_dims": [800, 400],
      "beta": 0.05,
      "dropout": 0.03,
      "lr": 0.01,
      "weight_decay": 0.001,
      "batch_size": 128,
      "n_epochs": 50,
      "patience": 15,
      "val_split": 0.3,
      "checkpoint_freq": 10
    },
    "diffusion": {
      "enabled": true,
      "use_checkpoint": "latest",
      "vae_latent_size": 128,
      "vae_hidden_dims": [1024, 1024],
      "vae_beta": 0.001,
      "vae_dropout": 0.1,
      "vae_lr": 0.001,
      "vae_weight_decay": 0.001,
      "vae_n_epochs": 200,
      "vae_patience": 20,
      "vae_overfitting_patience": 10,
      "vae_min_delta": 1e-4,
      "latent_size": 128,
      "hidden_dims": [512, 512, 256, 128],
      "num_timesteps": 1000,
      "beta_schedule": "cosine",
      "classifier_free_prob": 0.1,
      "guidance_scale": 2.0,
      "dropout": 0.1,
      "lr": 1e-4,
      "weight_decay": 1e-5,
      "batch_size": 128,
      "n_epochs": 50,
      "patience": 15,
      "train_split": 0.8,
      "val_split": 0.1,
      "checkpoint_freq": 5
    },
    "gan": {
      "enabled": true,
      "use_checkpoint": "latest",
      "latent_dim": 128,
      "gen_layers": [256, 512, 1024],
      "critic_layers": [1024, 512, 256],
      "lambda_gp": 15.0,
      "gp_target": 0.6,
      "lr_g": 0.00015,
      "lr_c": 0.00015,
      "beta1": 0.0,
      "beta2": 0.9,
      "batch_size": 32,
      "training_unit": "epochs",
      "n_epochs": 100,
      "critic_iters": 4,
      "train_split": 0.85,
      "save_freq": 500,
      "log_freq": 100,
      "validation_freq": 5,
      "clip_gradients": true,
      "gradient_clip_norm": 1.0
    }
  },

  "mmd_eval": {
    "enabled": true,
    "cvae_freq": 5,
    "diffusion_freq": 5,
    "gan_freq": 1,
    "patience_mmd": 3,
    "min_delta": 0.001,
    "max_sigma_samples": 1000
  },

  "visualization": {
    "model_order": ["cvae", "diffusion", "gan"],
    "colors": {
      "cvae": "#8E44AD",
      "diffusion": "#E57373",
      "gan": "#E69F00"
    },
    "display_names": {
      "cvae": "CVAE",
      "diffusion": "Diffusion",
      "gan": "GAN"
    },
    "dpi": 300,
    "format": "png"
  },

  "plots": {
    "umap_real": {
      "name": "umap_real_only_{model_name}",
      "title": "UMAP: Real Data ({model_name})"
    },
    "umap_combined": {
      "name": "umap_real_vs_{model_name}",
      "title": "UMAP: Real vs. Synthetic Data ({model_name})"
    },
    "metrics": {
      "name": "scientific_metrics_comparison",
      "title": "Synthetic Data Quality Assessment for CVAE, Diffusion, and GAN Models for {dataset_name}"
    },
    "deg_overlap": {
      "name": "shared_deg_comparison",
      "title": "Shared Differentially Expressed Genes between Real and CVAE, Diffusion, & GAN Model Generated Data for {dataset_name}"
    },
    "classifier": {
      "name": "phenotype_predictive_genes",
      "title": "Overlap of Phenotypically Predictive Genes: {group1_display_name} vs. {group2_display_name} Classification in {dataset_name}"
    }
  },

  "analysis": {
    "deg": { "enabled": true, "n_genes": 50, "method": "wilcoxon" },
    "classifier": { "enabled": true, "n_genes": 50, "n_estimators": 100 }
  }
}
```

### Key Config Options

- **paths.raw_data**: Path to raw input h5ad file (used when preprocessing is enabled)
- **paths.processed_data**: Path where preprocessed data will be saved (or path to already-processed data if preprocessing is disabled)

### Path Configuration Examples

**Option 1: With preprocessing (recommended for raw data)**
```json
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
  "normalize": true,
  "target_sum": 10000
}
```

**Option 2: Without preprocessing (for already-processed data)**
```json
"paths": {
  "raw_data": null,
  "processed_data": "real_datasets/cancer_processed.h5ad",
  "output_dir": "outputs/cancer"
},
"preprocessing": {
  "enabled": false
}
```

### Preprocessing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Enable/disable preprocessing stage |
| `min_cells` | `3` | Minimum cells where a gene must be expressed |
| `sparsity_threshold` | `0.997` | Max zero percentage for genes (genes with >99.7% zeros are removed) |
| `n_hvg` | `2000` | Number of highly variable genes to select (0 to skip) |
| `normalize` | `true` | Whether to normalize data (normalize_total + log1p) |
| `target_sum` | `10000` | Target sum for normalization |

**Preprocessing Steps:**
1. **Gene filtering by min_cells**: Remove genes expressed in fewer than `min_cells` cells
2. **Sparsity filtering**: Remove genes with zero percentage above `sparsity_threshold`
3. **HVG selection**: Select top `n_hvg` highly variable genes (uses normalized data for selection, preserves raw counts)
4. **Normalization**: `normalize_total` to `target_sum` + `log1p` transformation
- **use_checkpoint**: `"latest"` uses the most recent trained model, or specify a path to a specific checkpoint
- **conditions.columns**: Columns used for conditioning during training
- **conditions.types**: Type of each condition column ("categorical" or "continuous") - required for Diffusion model
- **conditions.primary**: Column and groups used for differential analysis (e.g., flight vs ground)
- **conditions.primary.group1_display_name / group2_display_name**: Human-readable names for plots (e.g., "Ground Control" instead of "0.0")
- **visualization.model_order**: Order of models in bar charts (left to right)
- **visualization.colors**: Hex colors for each model in plots
- **plots.*.name**: Filename for saved plot (without extension)
- **plots.*.title**: Plot title template. Placeholders: `{dataset_name}`, `{model_name}`, `{group1_display_name}`, `{group2_display_name}`

## Show Plots Option

By default, plots are saved to files without displaying. To show plots interactively:

```bash
# Training scripts
python models/CVAE/train.py --data ... --show-plots

# Analysis scripts
python analysis/run_deg_analysis.py --real ... --synthetic ... --show-plots
python analysis/run_classifier_analysis.py --real ... --synthetic ... --show-plots
```

## Output Structure

```
outputs/{dataset_name}/
├── data_split_indices.npz             # Shared stratified train/val split
├── sigma_heuristic.npz                # Pre-computed sigma (when mmd_eval.enabled)
├── trained_models/
│   ├── cvae/
│   │   ├── run_20251201_103000/
│   │   │   ├── checkpoints/
│   │   │   ├── best_model.pt
│   │   │   ├── training_metrics.json
│   │   │   ├── concurrent_mmd_history.json  # (when mmd_eval.enabled)
│   │   │   └── loss_curves.png
│   │   └── latest_model.json
│   ├── diffusion/
│   │   ├── run_20251201_110000/
│   │   │   ├── checkpoints/
│   │   │   ├── best_model.pt
│   │   │   ├── vae_model.pt
│   │   │   ├── concurrent_mmd_history.json  # (when mmd_eval.enabled)
│   │   │   └── condition_config.json
│   │   └── latest_model.json
│   └── gan/
│       ├── run_20251201_120000/
│       │   ├── checkpoints/
│       │   ├── best_model/
│       │   │   ├── generator.weights.h5
│       │   │   └── critic.weights.h5
│       │   ├── concurrent_mmd_history.json  # (when mmd_eval.enabled)
│       │   └── condition_config.json
│       └── latest_model.json
├── synthetic_data/
│   ├── cvae_{dataset}_{timestamp}.h5ad
│   ├── cvae_{dataset}_{timestamp}_meta.json
│   ├── cvae_matched_{dataset}_{timestamp}.h5ad
│   ├── diffusion_{dataset}_{timestamp}.h5ad
│   ├── diffusion_matched_{dataset}_{timestamp}.h5ad
│   ├── gan_{dataset}_{timestamp}.h5ad
│   ├── gan_matched_{dataset}_{timestamp}.h5ad
│   └── matched_mmd_info_{timestamp}.json
├── posthoc_mmd/
│   ├── cvae/
│   ├── diffusion/
│   └── gan/
└── analysis/
    ├── deg/
    ├── classifier/
    ├── umap/
    └── metrics/
```

## Pipeline Stages

1. **preprocess**: (Optional) Preprocess raw data
2. **train**: Train models (CVAE -> Diffusion -> GAN)
3. **generate**: Generate synthetic data using trained models
4. **posthoc-mmd**: Evaluate MMD across all saved checkpoints (run after training)
5. **analyze**: Run DEG, classifier, UMAP, and metrics analysis (in that order)

### Post-hoc MMD Evaluation

After training, each model has multiple saved checkpoints. The `posthoc-mmd` stage evaluates MMD (Maximum Mean Discrepancy) for every checkpoint across all three models, so you can identify which checkpoint produced the best synthetic data.

```bash
python driver.py --config dataset_configs/osdr-402.json --stage posthoc-mmd
```

This runs in two phases for memory efficiency:
1. **Phase 1**: Generate synthetic data from each checkpoint, save to disk
2. **Phase 2**: Load saved data and compute MMD scores

Results are saved to `outputs/{dataset}/posthoc_mmd/` for each model.

### Generating from MMD-Matched Checkpoints

Once post-hoc MMD scores are computed, you can generate synthetic data using checkpoints that have similar (matched) MMD scores across all three models. This ensures a fair comparison during analysis by selecting checkpoints at a comparable quality level.

```bash
python generate_from_matched_mmd.py --config dataset_configs/osdr-402.json
```

This script:
1. Reads the post-hoc MMD results for each model
2. Finds checkpoints with the closest MMD scores across CVAE, Diffusion, and GAN
3. Generates synthetic data from those matched checkpoints
4. Saves files as `{model}_matched_{dataset}_{timestamp}.h5ad` in `synthetic_data/`

The matched files are automatically picked up by the analyze stage (it uses the most recently modified file per model).

### Analysis Stage Details

The analyze stage runs the following sub-analyses in order:

1. **DEG** (per model) → **DEG comparison plot** (combined)
2. **Classifier** (per model) → **Classifier comparison plot** (combined)
3. **UMAP**
4. **Metrics**

Each sub-analysis can be toggled via the config:

```json
"analysis": {
    "deg": { "enabled": true },
    "classifier": { "enabled": true },
    "umap": { "enabled": true },
    "metrics": { "enabled": true }
}
```

There is no way to run a single sub-analysis from the driver. To run them individually, call the scripts directly:

```bash
# DEG for a single model
python analysis/run_deg_analysis.py --real real_datasets/data.h5ad \
    --synthetic outputs/dataset/synthetic_data/cvae_matched_dataset_*.h5ad \
    --output-dir outputs/dataset/analysis/deg/cvae \
    --column-name Flight --group1 Flight --group2 Ground \
    --group1-display SpaceFlight --group2-display "Ground Control" \
    --method t-test --n-genes 50

# DEG comparison plot (after running DEG for all models)
python analysis/shared_deg_comparison.py --deg-dir outputs/dataset/analysis/deg \
    --output-dir outputs/dataset/analysis/deg \
    --core-output-dir outputs/dataset/analysis \
    --dataset-name OSDR-402 \
    --group1-display SpaceFlight --group2-display "Ground Control" --n-genes 50
```

## Stratified Data Splitting

All three models (CVAE, Diffusion, GAN) use the **same stratified train/validation split** to ensure fair comparison. The split is automatically created before training begins.

**How it works:**
- Cells are grouped by unique condition combinations (e.g., `Flight=1_Strain=A_Sex=M`)
- Within each group, 20% are randomly assigned to validation (configurable via `data_split.val_ratio`)
- This guarantees proportional representation of all condition combinations in both sets
- Split indices are saved to `outputs/{dataset}/data_split_indices.npz` and reused across all models

**Configuration:**
```json
"data_split": {
  "val_ratio": 0.2,
  "seed": 42
}
```

**Key benefits:**
- All models train/validate on identical data subsets
- Stratification prevents imbalanced validation sets
- Reproducible via seed; delete `.npz` file to regenerate

The split utility is located in `utils/data_splitter.py`.

## MMD-Based Early Stopping

When `mmd_eval.enabled` is `true` in the config, the pipeline uses MMD (Maximum Mean Discrepancy) to control early stopping during training, replacing loss-based stopping.

### How It Works

1. **Sigma pre-computation**: Before training, sigma is computed ONCE from training data using stratified sampling (proportional to condition groups). Saved to `outputs/{dataset}/sigma_heuristic.npz` and shared by all models.

2. **During training**: At model-specific intervals (`cvae_freq`, `diffusion_freq`, `gan_freq`), generate synthetic data matching validation conditions and compute MMD. If MMD does not improve by `min_delta` for `patience_mmd` consecutive evaluation intervals, training stops early.

3. **Checkpoint alignment**: When MMD is active, checkpoint saving aligns with MMD evaluation intervals (replacing the model's own `checkpoint_freq`/`save_freq`). When MMD is NOT active, existing checkpoint logic is completely unchanged.

4. **Output**: Each model saves `concurrent_mmd_history.json` in its run directory:

```json
{
  "model_type": "cvae",
  "dataset": "spaceflight_liver",
  "sigma": 12.453,
  "best_epoch": 40,
  "stopped_epoch": 70,
  "stop_reason": "mmd_patience_exceeded",
  "history": [
    {"epoch": 5, "mmd": 0.352, "loss": 0.234, "checkpoint": "checkpoint_epoch_5.pt", "is_best": false}
  ]
}
```

### Configuration

```json
"mmd_eval": {
  "enabled": true,
  "cvae_freq": 5,            // CVAE: eval + checkpoint every N epochs
  "diffusion_freq": 5,       // Diffusion: eval + checkpoint every N epochs
  "gan_freq": 1,             // GAN: eval + checkpoint every N epochs
  "patience_mmd": 3,         // Eval intervals without improvement before stopping
  "min_delta": 0.001,        // Minimum MMD decrease to count as improvement
  "max_sigma_samples": 1000  // Max stratified samples for sigma computation
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Enable MMD-based early stopping |
| `cvae_freq` | `5` | CVAE: evaluate MMD every N epochs |
| `diffusion_freq` | `5` | Diffusion: evaluate MMD every N epochs |
| `gan_freq` | `1` | GAN: evaluate MMD every N epochs |
| `patience_mmd` | `3` | Number of evaluation intervals without improvement before stopping |
| `min_delta` | `0.001` | Minimum MMD decrease required to count as improvement |
| `max_sigma_samples` | `1000` | Maximum samples for stratified sigma computation |

### Backward Compatibility

- When `mmd_eval.enabled` is `false` (default), all training behavior is identical to before -- loss-based early stopping and checkpoint_freq/save_freq work as usual.
- When `mmd_eval.enabled` is `true` but no `sigma_path` is available (e.g., running model scripts directly without the driver), MMD is evaluated passively without early stopping (legacy behavior).

### Key Files

- `utils/mmd_early_stopping.py` - `MMDEarlyStoppingTracker` class and sigma save/load utilities
- `utils/mmd.py` - `compute_sigma_stratified()` for proportional sampling per condition group
- `driver.py` - `ensure_sigma_heuristic()` pre-computes sigma before training

## CLI Arguments

### Driver (Main Pipeline)

```
--config       Path to dataset config JSON (required)
--stage        Pipeline stage: all, preprocess, train, generate, posthoc-mmd, analyze
--model        Model to train/generate: all, cvae, vae, diffusion, gan
--checkpoint   Override checkpoint path for generation
--dry-run      Print commands without executing
--fail-fast    Stop pipeline immediately on any error (default: continue on errors)
```

**Examples:**

```bash
# Run full pipeline, stop on first error
python driver.py --config dataset_configs/cancer.json --fail-fast

# Train only CVAE model
python driver.py --config dataset_configs/cancer.json --stage train --model cvae

# Train only the VAE component for diffusion (Phase 1 only)
python driver.py --config dataset_configs/cancer.json --stage train --model vae

# Generate synthetic data using a specific checkpoint
python driver.py --config dataset_configs/cancer.json --stage generate --model cvae --checkpoint path/to/model.pt
```

### Training Scripts

**CVAE Training (`models/CVAE/train.py`):**
```
--data              Path to input h5ad file (required)
--output-dir        Output directory for trained models (required)
--condition-cols    Comma-separated condition column names (required)
--latent-size       Latent space dimension (default: 32)
--hidden-dims       Hidden layer dimensions (default: 800,400)
--beta              KL divergence weight (default: 0.05)
--dropout           Dropout rate (default: 0.03)
--lr                Learning rate (default: 0.01)
--weight-decay      Weight decay (default: 0.001)
--batch-size        Batch size (default: 128)
--n-epochs          Maximum epochs (default: 50)
--patience          Early stopping patience (default: 15)
--overfitting-patience  Overfitting detection patience (default: 10)
--min-delta         Minimum delta for early stopping (default: 1e-4)
--val-split         Validation split ratio (default: 0.3)
--checkpoint-freq   Save checkpoint every N epochs (default: 10)
--random-state      Random seed (default: 42)
--show-plots        Display plots interactively (default: save only)
--mmd-eval          Enable MMD evaluation during training
--mmd-eval-freq     MMD evaluation frequency in epochs (default: 5)
--sigma-path        Path to pre-computed sigma_heuristic.npz
--mmd-patience      MMD early stopping patience (default: 3)
--mmd-min-delta     Minimum MMD improvement threshold (default: 0.001)
```

**Diffusion Training (`models/Diffusion/train.py`):**
```
--data              Path to input h5ad file (required)
--output-dir        Output directory (required)
--condition-cols    Comma-separated condition columns (required)
--condition-types   Comma-separated types (categorical/continuous) matching columns

# VAE Architecture
--vae-latent-size   VAE latent dimension (default: 128)
--vae-hidden-dims   VAE hidden layers (default: 1024,1024)
--vae-model-path    Path to pre-trained VAE (skip VAE training if provided)

# VAE Training
--vae-beta          VAE KL weight (default: 0.001)
--vae-dropout       VAE dropout (default: 0.1)
--vae-lr            VAE learning rate (default: 0.001)
--vae-weight-decay  VAE weight decay (default: 0.001)
--vae-n-epochs      VAE max epochs (default: 200)
--vae-patience      VAE early stopping patience (default: 20)
--vae-overfitting-patience  VAE overfitting patience (default: 10)
--vae-min-delta     VAE minimum improvement (default: 1e-4)

# Diffusion Architecture
--latent-size       Diffusion latent dimension (default: 128)
--hidden-dims       Diffusion hidden layers (default: 512,512,256,128)
--num-timesteps     Diffusion timesteps (default: 1000)
--beta-schedule     Beta schedule: cosine, linear (default: cosine)
--classifier-free-prob  Classifier-free guidance probability (default: 0.1)
--dropout           Dropout rate (default: 0.1)

# Diffusion Training
--lr                Diffusion learning rate (default: 1e-4)
--weight-decay      Diffusion weight decay (default: 1e-5)
--batch-size        Batch size (default: 128)
--n-epochs          Diffusion max epochs (default: 50)
--patience          Diffusion early stopping patience (default: 15)
--train-split       Training split ratio (default: 0.8)
--val-split         Validation split ratio (default: 0.1)
--checkpoint-freq   Checkpoint frequency (default: 5)

# Mode Options
--vae-only          Only train VAE (Phase 1), skip diffusion
--show-plots        Display plots interactively

# MMD Early Stopping
--mmd-eval          Enable MMD evaluation during training
--mmd-eval-freq     MMD evaluation frequency in epochs (default: 5)
--sigma-path        Path to pre-computed sigma_heuristic.npz
--mmd-patience      MMD early stopping patience (default: 3)
--mmd-min-delta     Minimum MMD improvement threshold (default: 0.001)
```

**GAN Training (`models/GAN/train.py`):**
```
--data              Path to input h5ad file (required)
--output-dir        Output directory (required)
--condition-cols    Comma-separated condition columns (required)
--latent-dim        Latent dimension (default: 128)
--gen-layers        Generator hidden layers (default: 256,512,1024)
--critic-layers     Critic hidden layers (default: 1024,512,256)
--lambda-gp         Gradient penalty lambda (default: 10.0)
--gp-target         Target gradient penalty (default: 0.6)
--lr-g              Generator learning rate (default: 1e-4)
--lr-c              Critic learning rate (default: 1e-4)
--beta1             Adam beta1 (default: 0.0)
--beta2             Adam beta2 (default: 0.9)
--batch-size        Batch size (default: 64)
--max-steps         Maximum training steps (default: 50000)
--critic-iters      Critic iterations per generator iteration (default: 5)
--train-split       Training split ratio (default: 0.8)
--save-freq         Checkpoint frequency (default: 500 steps or 5 epochs)
--save-freq-unit    Unit for save-freq: steps or epochs (default: steps)
--log-freq          Logging frequency in steps (default: 100)
--validation-freq   Validation every N save_freq intervals (default: 5)
--clip-gradients    Enable gradient clipping
--gradient-clip-norm  Max gradient norm (default: 1.0)

# MMD Early Stopping
--mmd-eval          Enable MMD evaluation at checkpoints
--mmd-eval-freq     MMD evaluation frequency (every N checkpoints, default: 1)
--sigma-path        Path to pre-computed sigma_heuristic.npz
--mmd-patience      MMD early stopping patience (default: 3)
--mmd-min-delta     Minimum MMD improvement threshold (default: 0.001)
--mmd-n-epochs      Total epochs for epoch-based MMD tracking (default: 0)
--save-freq-unit    Unit for save-freq: steps or epochs (default: steps)
```

### Generation Scripts

**CVAE Generation (`models/CVAE/generate.py`):**
```
--data              Path to reference h5ad file (required)
--model             Path to trained model .pt file (required)
--output            Output path for synthetic data .h5ad (required)
--condition-cols    Comma-separated condition columns (required)
--latent-size       Latent dimension (must match model, default: 32)
--batch-size        Generation batch size (default: 50)
--scale-factor      Scale factor for sample counts (default: 1.0)
```

**Diffusion Generation (`models/Diffusion/generate.py`):**
```
--data              Path to reference h5ad file (required)
--checkpoint-dir    Path to checkpoint directory (required)
--output            Output path for synthetic data .h5ad (required)
--condition-cols    Comma-separated condition columns (required)
--batch-size        Generation batch size (default: 64)
--scale-factor      Scale factor for sample counts (default: 1.0)
--guidance-scale    Classifier-free guidance scale (default: 2.0)
```

**GAN Generation (`models/GAN/generate.py`):**
```
--data              Path to reference h5ad file (required)
--checkpoint-dir    Path to checkpoint directory (required)
--output            Output path for synthetic data .h5ad (required)
--condition-cols    Comma-separated condition columns (required)
--latent-dim        Latent dimension (must match model, default: 128)
--gen-layers        Generator layers (must match model, default: 256,512,1024)
--batch-size        Generation batch size (default: 64)
--scale-factor      Scale factor for sample counts (default: 1.0)
```

### Preprocessing Script

**Preprocessing (`preprocessing/preprocess.py`):**
```
--input             Path to input h5ad file (required)
--output            Path to save processed h5ad file (required)
--min-cells         Minimum cells for gene filtering (default: 3)
--sparsity-threshold  Max zero percentage for genes (default: 0.997)
--n-hvg             Number of highly variable genes (0 to skip, default: 2000)
--no-normalize      Skip normalization (keep raw counts)
--target-sum        Target sum for normalization (default: 10000)
--quiet             Reduce output verbosity
```

## Model-Specific Notes

### CVAE
- PyTorch-based conditional VAE
- Uses categorical encoding for condition columns
- Checkpoints saved as `.pt` files
- Key parameters: `latent_size`, `hidden_dims`, `beta` (KL weight)

### Diffusion
- Uses a VAE for latent compression + diffusion model in latent space
- **Two-phase training**: VAE is trained first (from scratch), then diffusion model uses the trained VAE
- Classifier-free guidance for conditional generation
- Requires both `vae_model.pt` and diffusion checkpoints
- Key parameters: `num_timesteps`, `classifier_free_prob`, `guidance_scale`

#### Diffusion Training Phases

The diffusion training process runs in two phases:

**Phase 1: VAE Training**
- Trains a VAE to compress gene expression data to a latent space
- Uses early stopping with configurable patience and overfitting detection
- Saves `vae_model.pt` and `vae_training_metrics.json`
- Key VAE parameters:
  - `vae_latent_size`: Latent space dimension (default: 128)
  - `vae_hidden_dims`: Hidden layer sizes (default: [1024, 1024])
  - `vae_beta`: KL divergence weight (default: 0.001)
  - `vae_n_epochs`: Max training epochs (default: 200)
  - `vae_patience`: Early stopping patience (default: 20)
  - `vae_overfitting_patience`: Overfitting detection patience (default: 10)

**Phase 2: Diffusion Training**
- Uses the trained VAE to encode data to latent space
- Trains diffusion model on latent representations
- Saves diffusion checkpoints and `condition_config.json`

#### Diffusion Condition Config

The diffusion model saves a `condition_config.json` file during training that describes how conditions are encoded. This file is required for generation and has the following structure:

```json
{
  "condition_config": {
    "Flight": "categorical",
    "Strain": "categorical",
    "Age at Launch": "continuous",
    "Duration": "continuous"
  },
  "full_condition_config": {
    "Flight": {
      "type": "categorical",
      "n_categories": 2,
      "categorical_mapping": {"0.0": 0, "1.0": 1}
    },
    "Strain": {
      "type": "categorical",
      "n_categories": 3,
      "categorical_mapping": {"C57BL/6J": 0, "BALB/cAnNTac": 1, "C3H/HeJ": 2}
    },
    "Age at Launch": {
      "type": "continuous",
      "mean": 12.5,
      "std": 2.3
    },
    "Duration": {
      "type": "continuous",
      "mean": 35.0,
      "std": 5.2
    }
  }
}
```

**Fields:**
- **condition_config**: Simple mapping of column name to type ("categorical" or "continuous")
- **full_condition_config**: Detailed encoding information
  - For categorical: `n_categories` and `categorical_mapping` (value -> index)
  - For continuous: `mean` and `std` for normalization

The condition processor uses this to:
1. One-hot encode categorical conditions
2. Z-score normalize continuous conditions
3. Concatenate all conditions into a single conditioning vector

### GAN
- TensorFlow-based Wasserstein GAN with gradient penalty
- Uses generic condition processor for arbitrary categorical columns
- Checkpoints saved as directories with `generator.weights.h5` and `critic.weights.h5`
- Key parameters: `lambda_gp`, `gp_target`, `critic_iters`, `gen_layers`, `critic_layers`
- **No loss-based early stopping**: Unlike CVAE and Diffusion, the GAN does not use loss-based early stopping because critic and generator objectives are adversarial. However, when `mmd_eval.enabled` is `true`, MMD-based early stopping is available -- the GAN will stop training if MMD stagnates for `patience_mmd` evaluation intervals. Without MMD, the GAN trains for the full configured duration and model quality should be assessed post-training using the `posthoc-mmd` stage.

#### GAN Training: Steps vs Epochs

Unlike CVAE and Diffusion models that train in epochs, GANs traditionally train in steps. However, you can configure the GAN to use either **steps** or **epochs** in the config file.

**Option 1: Specify epochs (recommended)**
```json
"gan": {
  "training_unit": "epochs",
  "n_epochs": 100,
  "batch_size": 32,
  "critic_iters": 4,
  "train_split": 0.85,
  ...
}
```

**Option 2: Specify steps directly**
```json
"gan": {
  "training_unit": "steps",
  "max_steps": 300000,
  "batch_size": 32,
  "critic_iters": 4,
  "train_split": 0.85,
  ...
}
```

**How the conversion works:**

When using `"training_unit": "epochs"`, the driver automatically calculates `max_steps` based on your dataset size:

```
train_samples = n_samples × train_split
batches_per_epoch = train_samples ÷ batch_size
steps_per_epoch = batches_per_epoch ÷ critic_iters
max_steps = n_epochs × steps_per_epoch
```

**Example calculation:**
- Dataset: 10,000 samples
- `train_split`: 0.85 → 8,500 training samples
- `batch_size`: 32 → 265 batches per epoch
- `critic_iters`: 4 → 66 steps per epoch (critic trains 4x per generator step)
- `n_epochs`: 100 → 6,600 max_steps

The driver prints the conversion when training starts:
```
[INFO] GAN: 100 epochs -> 6600 steps (10000 samples, 66 steps/epoch)
```

**Why steps matter for GANs:**

In WGAN-GP training, each "step" consists of:
1. `critic_iters` updates to the critic (discriminator)
2. 1 update to the generator

So with `critic_iters=4`, one epoch means seeing all training data once by the *critic*, but the generator only updates `batches_per_epoch / critic_iters` times per epoch.

#### GAN Condition Config

Similar to diffusion, the GAN saves a `condition_config.json` for generation:

```json
{
  "condition_columns": ["annotation_major", "tissue"],
  "column_info": {
    "annotation_major": {
      "type": "categorical",
      "categories": ["B cells", "T cells", "Monocytes", ...],
      "n_categories": 8
    },
    "tissue": {
      "type": "categorical",
      "categories": ["metastasis normal", "metastasis tumor"],
      "n_categories": 2
    }
  },
  "total_condition_dim": 10
}
```

## Requirements

- Python 3.8+
- PyTorch (for CVAE, Diffusion)
- TensorFlow (for GAN)
- scanpy, anndata
- scikit-learn
- matplotlib, seaborn
