# scRNA-seq Synthetic Data Generation Pipeline Plan

## Overview

Build a modular, config-driven pipeline in `pipeline/` that orchestrates preprocessing, training (CVAE, Diffusion, GAN), synthetic data generation, and analysis - all from a single JSON config file.

---

## 1. Directory Structure

```
pipeline/
├── driver.py                      # Main orchestrator CLI
├── config_loader.py               # Config loading with defaults
├── utils/
│   └── model_tracker.py           # Tracks latest models for each type
├── dataset_configs/
│   ├── README.md                  # Config format documentation
│   ├── cancer.json                # Cancer dataset config
│   └── spaceflight.json           # Spaceflight dataset config
├── models/
│   ├── CVAE/                      # CVAE model training & generation
│   ├── Diffusion/                 # Diffusion model training & generation
│   └── GAN/                       # GAN model training & generation
├── analysis/                      # Analysis scripts
├── real_datasets/                 # Input data
└── README.md                      # Driver usage documentation
```

---

## 2. Config File Format

**File**: `dataset_configs/{name}.json`

```json
{
  "dataset_name": "cancer",
  "description": "Cancer scRNA-seq dataset",

  "paths": {
    "raw_data": "real_datasets/cancer_raw.h5ad",
    "processed_data": "real_datasets/cancer_processed.h5ad",
    "output_dir": "outputs/cancer"
  },

  "preprocessing": {
    "min_cells": 3,
    "sparsity_threshold": 0.997,
    "n_hvg": 2000,
    "normalize": true
  },

  "conditions": {
    "columns": ["annotation_major", "tissue"],
    "primary": {
      "column": "tissue",
      "group1": "metastasis normal",
      "group2": "metastasis tumor"
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
      "batch_size": 128,
      "n_epochs": 50,
      "patience": 15,
      "checkpoint_freq": 10
    },
    "diffusion": {
      "enabled": true,
      "use_checkpoint": "latest",
      "latent_size": 128,
      "hidden_dims": [1024, 1024],
      "n_epochs": 50,
      "batch_size": 128,
      "patience": 15,
      "checkpoint_freq": 10
    },
    "gan": {
      "enabled": true,
      "use_checkpoint": "latest",
      "latent_dim": 128,
      "max_steps": 50000,
      "batch_size": 64,
      "save_freq": 5000
    }
  },

  "analysis": {
    "deg": {"enabled": true, "n_genes": 50, "method": "t-test"},
    "umap": {"enabled": true},
    "metrics": {"enabled": true},
    "classifier": {"enabled": true}
  }
}
```

**`use_checkpoint` Options**:
- `"latest"` - Use the best checkpoint from the most recent training run (reads from `latest_model.json`)
- `"path/to/specific/run_20251201_103000/best_model.pt"` - Use a specific checkpoint
- `"path/to/specific/run_20251201_103000/checkpoints/epoch_30.pt"` - Use a specific epoch checkpoint

---

## 3. Output Organization

```
outputs/{dataset_name}/
├── trained_models/
│   ├── cvae/
│   │   ├── run_20251201_103000/                  # Timestamped run folder
│   │   │   ├── checkpoints/
│   │   │   │   ├── epoch_10.pt
│   │   │   │   ├── epoch_20.pt
│   │   │   │   └── ...
│   │   │   ├── best_model.pt                     # Copy of best checkpoint
│   │   │   ├── training_metrics.json             # Loss curves, best epoch info
│   │   │   ├── loss_curves.png                   # Visual loss plot
│   │   │   ├── label_encoders.pkl
│   │   │   └── run_config.json                   # Config snapshot for this run
│   │   └── latest_model.json                     # Points to latest run's best
│   ├── diffusion/
│   │   ├── run_20251201_110000/
│   │   │   ├── checkpoints/
│   │   │   ├── best_model.pt
│   │   │   └── ...
│   │   └── latest_model.json
│   └── gan/
│       ├── run_20251201_120000/
│       │   ├── checkpoints/
│       │   ├── best_model/                       # GAN saves as directory
│       │   └── ...
│       └── latest_model.json
│
├── synthetic_data/
│   ├── cvae_{dataset_name}_{timestamp}.h5ad
│   ├── cvae_{dataset_name}_{timestamp}_meta.json
│   ├── diffusion_{dataset_name}_{timestamp}.h5ad
│   ├── diffusion_{dataset_name}_{timestamp}_meta.json
│   ├── gan_{dataset_name}_{timestamp}.h5ad
│   └── gan_{dataset_name}_{timestamp}_meta.json
│
└── analysis/
    ├── deg/
    ├── umap/
    ├── metrics/
    └── classifier/
```

**Model Tracking**: `latest_model.json` contains:
```json
{
  "run_dir": "run_20251201_103000",
  "best_model_path": "run_20251201_103000/best_model.pt",
  "best_epoch": 45,
  "best_val_loss": 0.0234,
  "trained_at": "2025-12-01 10:30:00",
  "config_used": { ... snapshot of model config ... }
}
```

**Training Metrics** (`training_metrics.json`):
```json
{
  "best_epoch": 45,
  "best_val_loss": 0.0234,
  "total_epochs": 50,
  "early_stopped": true,
  "best_checkpoint": "checkpoints/epoch_45.pt",
  "epoch_losses": [
    {"epoch": 1, "train_loss": 0.1234, "val_loss": 0.1456, "checkpoint": "checkpoints/epoch_1.pt"},
    {"epoch": 2, "train_loss": 0.0987, "val_loss": 0.1123, "checkpoint": "checkpoints/epoch_2.pt"},
    ...
  ]
}
```

**Synthetic Data Metadata** (`*_meta.json`):
```json
{
  "model_path": "trained_models/cvae/run_20251201_103000/best_model.pt",
  "source_data": "real_datasets/cancer_processed.h5ad",
  "model_params": {"latent_size": 32, "beta": 0.05},
  "generated_at": "2025-12-01 12:00:00"
}
```

---

## 4. Driver CLI Design

**File**: `pipeline/driver.py`

```bash
# Full pipeline
python driver.py --config dataset_configs/cancer.json

# Individual stages
python driver.py --config dataset_configs/cancer.json --stage preprocess
python driver.py --config dataset_configs/cancer.json --stage train
python driver.py --config dataset_configs/cancer.json --stage train --model cvae
python driver.py --config dataset_configs/cancer.json --stage generate
python driver.py --config dataset_configs/cancer.json --stage analyze

# Override use_checkpoint from config via CLI (useful for one-off tests)
python driver.py --config dataset_configs/cancer.json --stage generate --model cvae --checkpoint path/to/specific.pt

# Dry run
python driver.py --config dataset_configs/cancer.json --dry-run
```

**Arguments**:
- `--config` (required): Path to JSON config
- `--stage`: `all`, `preprocess`, `train`, `generate`, `analyze`
- `--model`: `all`, `cvae`, `diffusion`, `gan` (for train/generate)
- `--checkpoint`: Override `use_checkpoint` from config for generation
- `--dry-run`: Show what would execute without running

---

## 5. Execution Order

1. **Preprocess** (if enabled): raw_data -> processed_data
2. **Train** (sequential): CVAE -> Diffusion -> GAN
3. **Generate** (sequential): Uses latest_model.json for each
4. **Analyze**: Uses generated synthetic data paths

Each stage is independently runnable with the same config.

---

## 6. Condition Handling Strategy

The config specifies conditions flexibly:
```json
"conditions": {
  "columns": ["tissue"],
  "primary": {
    "column": "tissue",
    "group1": "metastasis normal",
    "group2": "metastasis tumor"
  }
}
```

For spaceflight with numeric Flight column:
```json
"conditions": {
  "columns": ["Flight", "Strain"],
  "primary": {
    "column": "Flight",
    "group1": "0.0",
    "group2": "1.0"
  }
}
```

Analysis scripts receive these via CLI args - they already handle string comparisons.
