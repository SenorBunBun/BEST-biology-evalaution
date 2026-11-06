# Pipeline Implementation Guide

This document provides a comprehensive understanding of the pipeline architecture, data flow, and requirements for adding new models or components.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Model Tracker System](#model-tracker-system)
3. [Data Flow](#data-flow)
4. [Adding a New Model](#adding-a-new-model)
5. [Config Structure](#config-structure)

---

## Architecture Overview

```
pipeline/
├── driver.py                  # Main orchestrator - calls all stages
├── config_loader.py           # Config parsing, defaults, validation
├── utils/
│   └── model_tracker.py       # Checkpoint management, latest_model.json
├── dataset_configs/
│   ├── cancer.json            # Dataset-specific configs
│   └── spaceflight.json
├── models/
│   ├── CVAE/
│   │   ├── train.py           # Training script (called by driver)
│   │   └── generate.py        # Generation script (called by driver)
│   ├── Diffusion/
│   │   ├── train.py           # Training script (VAE + Diffusion, supports --vae-only)
│   │   ├── train_vae.py       # Convenience wrapper (calls train.py --vae-only)
│   │   └── generate.py        # Generation script
│   └── GAN/
│       ├── train.py
│       └── generate.py
└── analysis/
    ├── run_deg_analysis.py
    └── run_classifier_analysis.py
```

### Pipeline Stages
1. **preprocess** - Data preprocessing (optional)
2. **train** - Model training (CVAE → Diffusion VAE → Diffusion → GAN)
3. **generate** - Synthetic data generation using trained models
4. **analyze** - DEG, classifier, metrics analysis

---

## Model Tracker System

### Purpose
The `ModelTracker` class manages:
- Timestamped run directories for each training session
- Checkpoint saving and organization
- `latest_model.json` tracking for easy model lookup
- Training metrics and loss curves

### Directory Structure per Model

```
outputs/{dataset}/trained_models/{model_type}/
├── latest_model.json              # Points to best model from latest run
├── run_20251201_103000/           # Timestamped run directory
│   ├── checkpoints/               # Epoch checkpoints
│   │   ├── epoch_10.pt
│   │   └── epoch_20.pt
│   ├── best_model.pt              # Copy of best checkpoint
│   ├── training_metrics.json      # Training history & config
│   ├── loss_curves.png            # Training visualization
│   └── run_config.json            # Config snapshot for this run
└── run_20251202_090000/           # Another run
    └── ...
```

### latest_model.json Structure

```json
{
  "run_dir": "run_20251201_103000",
  "best_model_path": "run_20251201_103000/best_model.pt",
  "best_epoch": 25,
  "best_val_loss": 0.0523,
  "trained_at": "2025-12-01 10:45:00",
  "config_used": {
    "latent_size": 32,
    "hidden_dims": [800, 400],
    "beta": 0.05,
    ...
  }
}
```

### ModelTracker Key Methods

| Method | Purpose |
|--------|---------|
| `create_run_dir()` | Creates timestamped run directory |
| `save_checkpoint()` | Saves epoch checkpoint to checkpoints/ |
| `save_best_model()` | Copies best checkpoint to best_model.pt |
| `save_training_metrics()` | Saves training_metrics.json |
| `save_loss_curves()` | Generates and saves loss_curves.png |
| `update_latest_model()` | Updates latest_model.json |
| `get_model_path("latest")` | Returns path to latest best_model.pt |
| `get_model_path(path)` | Validates and returns specific checkpoint |

### How "latest" Resolution Works

```python
# In driver.py - generation stage
model_path = tracker.get_model_path("latest")  # or specific path

# get_model_path() logic:
if checkpoint_spec == "latest":
    info = self.get_latest_model_info()  # Reads latest_model.json
    return self.model_dir / info["best_model_path"]
else:
    return Path(checkpoint_spec)  # Direct path
```

---

## Data Flow

### Training Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                           driver.py                                  │
│                                                                     │
│  1. load_config(config_path) → config dict                          │
│  2. create_output_dirs(config) → output_paths dict                  │
│  3. run_training(config, output_paths, models)                      │
│     │                                                               │
│     ├─→ run_cvae_training()                                         │
│     │   └─→ subprocess: models/CVAE/train.py --args...              │
│     │                                                               │
│     ├─→ run_vae_training() [NEW - for Diffusion's VAE]              │
│     │   └─→ subprocess: models/Diffusion/train_vae.py --args...     │
│     │                                                               │
│     ├─→ run_diffusion_training()                                    │
│     │   └─→ subprocess: models/Diffusion/train.py --vae-model ...   │
│     │                                                               │
│     └─→ run_gan_training()                                          │
│         └─→ subprocess: models/GAN/train.py --args...               │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Script Requirements

Each `train.py` must:

1. **Accept CLI arguments** for all model parameters
2. **Create a ModelTracker** for checkpoint management
3. **Create a run directory** via `tracker.create_run_dir()`
4. **Train the model** with early stopping
5. **Save checkpoints** periodically
6. **Save best_model.pt** after training
7. **Save training_metrics.json** with epoch losses
8. **Update latest_model.json** via `tracker.update_latest_model()`
9. **Return exit code** 0 for success, 1 for failure

### Generation Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                           driver.py                                  │
│                                                                     │
│  run_generation(config, output_paths, models)                       │
│     │                                                               │
│     ├─→ get_model_checkpoint(config, output_paths, "cvae")          │
│     │   └─→ ModelTracker.get_model_path("latest" or path)           │
│     │       └─→ Returns: outputs/.../cvae/run_xxx/best_model.pt     │
│     │                                                               │
│     └─→ subprocess: models/CVAE/generate.py                         │
│         --model {resolved_path}                                      │
│         --data {processed_data}                                      │
│         --output {synthetic_data/cvae_xxx.h5ad}                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Generation Script Requirements

Each `generate.py` must:

1. **Accept CLI arguments** for model path, data path, output path
2. **Load the trained model** from provided path
3. **Load any config files** (e.g., condition_config.json for Diffusion/GAN)
4. **Generate synthetic data** matching real data distribution
5. **Save as h5ad** with proper obs columns (conditions)
6. **Return exit code** 0 for success, 1 for failure

---

## Adding a New Model

### Step 1: Create Model Directory

```
models/NewModel/
├── train.py              # Training script
├── generate.py           # Generation script
├── model_architecture.py # Model class (optional, can be inline)
└── utils.py              # Model-specific utilities (optional)
```

### Step 2: Implement train.py

```python
#!/usr/bin/env python3
"""Training script for NewModel."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_tracker import ModelTracker

def parse_args():
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument('--data', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--condition-cols', required=True)
    # Model-specific params
    parser.add_argument('--latent-size', type=int, default=64)
    # Training params
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=128)
    return parser.parse_args()

def train(args):
    # 1. Initialize tracker
    output_dir = Path(args.output_dir)
    tracker = ModelTracker(output_dir, 'newmodel')
    run_dir = tracker.create_run_dir()

    # 2. Load data
    adata = sc.read_h5ad(args.data)

    # 3. Create model
    model = NewModel(...)

    # 4. Training loop with early stopping
    best_val_loss = float('inf')
    epoch_losses = []

    for epoch in range(args.n_epochs):
        train_loss = train_epoch(...)
        val_loss = validate_epoch(...)

        epoch_losses.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        # Save checkpoint
        if epoch % checkpoint_freq == 0:
            save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch}.pt")

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            save_model(run_dir / "best_model.pt")

        # Early stopping
        if patience_exceeded:
            break

    # 5. Save training metrics
    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs": len(epoch_losses),
        "early_stopped": patience_exceeded,
        "epoch_losses": epoch_losses,
        "model_params": {...}
    }
    tracker.save_training_metrics(run_dir, metrics)
    tracker.save_loss_curves(run_dir, epoch_losses)

    # 6. Update latest_model.json
    tracker.update_latest_model(run_dir, best_epoch, best_val_loss, config)

    return True

if __name__ == "__main__":
    args = parse_args()
    success = train(args)
    sys.exit(0 if success else 1)
```

### Step 3: Implement generate.py

```python
#!/usr/bin/env python3
"""Generation script for NewModel."""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--condition-cols', required=True)
    # Model-specific params needed for reconstruction
    parser.add_argument('--latent-size', type=int, default=64)
    return parser.parse_args()

def generate(args):
    # 1. Load model
    model = NewModel(...)
    model.load_state_dict(torch.load(args.model))

    # 2. Load real data for condition distribution
    adata = sc.read_h5ad(args.data)

    # 3. Generate synthetic data
    synthetic_X = model.generate(n_samples=adata.n_obs)

    # 4. Create AnnData with conditions
    synthetic_adata = sc.AnnData(X=synthetic_X)
    synthetic_adata.obs = sample_conditions(adata.obs)
    synthetic_adata.var_names = adata.var_names

    # 5. Save
    synthetic_adata.write_h5ad(args.output)

    return True
```

### Step 4: Update config_loader.py

Add to DEFAULTS:

```python
"models": {
    ...,
    "newmodel": {
        "enabled": True,
        "use_checkpoint": "latest",
        "latent_size": 64,
        "hidden_dims": [256, 128],
        "n_epochs": 50,
        "patience": 15,
        "batch_size": 128,
        ...
    }
}
```

### Step 5: Update driver.py

1. Add to MODELS list:
```python
MODELS = ['all', 'cvae', 'diffusion', 'gan', 'newmodel']
```

2. Add training function:
```python
def run_newmodel_training(config, output_paths, dry_run=False):
    nm_config = config["models"]["newmodel"]

    if not nm_config.get("enabled", True):
        print_step("NewModel training disabled", "SKIP")
        return True

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "models" / "NewModel" / "train.py"),
        "--data", config["paths"]["processed_data"],
        "--output-dir", str(output_paths["trained_models"]["newmodel"]),
        "--condition-cols", ",".join(config["conditions"]["columns"]),
        "--latent-size", str(nm_config.get("latent_size", 64)),
        "--n-epochs", str(nm_config.get("n_epochs", 50)),
        ...
    ]

    return run_subprocess(cmd, "NewModel training", dry_run)
```

3. Add generation function:
```python
def run_newmodel_generation(config, output_paths, checkpoint_override=None, dry_run=False):
    # Similar pattern to other generation functions
    model_path = get_model_checkpoint(config, output_paths, "newmodel", checkpoint_override)
    ...
```

4. Update output_paths in config_loader.py:
```python
"trained_models": {
    ...,
    "newmodel": base_dir / "trained_models" / "newmodel",
}
```

### Step 6: Update Dataset Configs

Add model config to each JSON:

```json
"models": {
    ...,
    "newmodel": {
        "enabled": true,
        "use_checkpoint": "latest",
        "latent_size": 64,
        ...
    }
}
```

---

## Config Structure

### Required Fields (validation in config_loader.py)

```
dataset_name        - String identifier
paths.processed_data - Path to h5ad data
paths.output_dir    - Output directory
conditions.columns  - List of condition column names
conditions.primary.column - Column for DEG/classifier
conditions.primary.group1 - First group value
conditions.primary.group2 - Second group value
```

### Config Flow

```
JSON Config File
      │
      ▼
load_config(path)
      │
      ├─→ Load JSON
      ├─→ deep_merge(DEFAULTS, user_config)
      ├─→ resolve_paths(config, config_dir)
      ├─→ validate_config(config)
      └─→ Return merged config dict
```

### Parameter Naming Convention

| Config Key | CLI Argument |
|------------|--------------|
| `latent_size` | `--latent-size` |
| `hidden_dims` | `--hidden-dims` (comma-separated) |
| `n_epochs` | `--n-epochs` |
| `vae_latent_size` | `--vae-latent-size` |

---

## Diffusion Model Special Case

Diffusion uses a **two-phase training** implemented in a single `train.py`:

### Training Modes

**Full Training (default)**: `python train.py --data ... --output-dir ...`
- Phase 1: Trains VAE from scratch (or loads pre-trained via `--vae-model-path`)
- Phase 2: Trains diffusion model using VAE's latent space
- Outputs both `vae_model.pt` and `best_model.pt`

**VAE-Only Training**: `python train.py --data ... --output-dir ... --vae-only`
- Only runs Phase 1 (VAE training)
- Stops after saving VAE model
- Useful for training VAE separately before diffusion
- From driver.py: `python driver.py --config ... --stage train --model vae`

### latest_model.json Structure for Diffusion

```json
{
  "run_dir": "run_20251201_103000",
  "best_model_path": "run_20251201_103000/best_model.pt",
  "best_epoch": 25,
  "best_val_loss": 0.0523,
  "trained_at": "2025-12-01 10:45:00",
  "config_used": {
    "diffusion": {
      "mode": "full",
      "vae_model_path": "run_20251201_103000/vae_model.pt",
      "vae_params": {
        "latent_size": 128,
        "hidden_dims": [1024, 1024],
        "beta": 0.001
      },
      "model_params": {
        "latent_size": 128,
        "hidden_dims": [512, 512, 256, 128],
        "num_timesteps": 1000
      },
      "condition_config": {...}
    }
  }
}
```

For VAE-only mode, `mode` will be `"vae_only"` and diffusion params will be absent.

### Directory Structure

```
trained_models/diffusion/
├── latest_model.json          # Points to best model, includes VAE info
└── run_20251201_103000/
    ├── best_model.pt          # Diffusion model (or absent in vae_only mode)
    ├── vae_model.pt           # VAE model
    ├── vae_training_curves.png
    ├── condition_config.json
    ├── training_metrics.json  # Includes both VAE and diffusion metrics
    ├── run_config.json
    └── checkpoints/
        ├── epoch_5.pt
        └── epoch_10.pt
```

### Generation
- Loads both VAE and Diffusion models from run directory
- VAE path is stored in `latest_model.json` under `config_used.diffusion.vae_model_path`
- Uses VAE to decode latent samples generated by diffusion

---

## Checklist for New Model

- [ ] Create `models/NewModel/train.py`
- [ ] Create `models/NewModel/generate.py`
- [ ] Add model to DEFAULTS in `config_loader.py`
- [ ] Add `trained_models["newmodel"]` path in `get_output_paths()`
- [ ] Add to MODELS list in `driver.py`
- [ ] Add `run_newmodel_training()` in `driver.py`
- [ ] Add `run_newmodel_generation()` in `driver.py`
- [ ] Call training/generation in `run_training()`/`run_generation()`
- [ ] Add model config to all dataset JSON files
- [ ] Update README.md with model documentation
- [ ] Test with `--dry-run` first

---

## MMD Evaluation During Training

Maximum Mean Discrepancy (MMD) evaluation measures how well generated synthetic data matches real validation data during training.

### Configuration

Add to dataset config JSON:

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

### Two Modes of Operation

**Mode 1: MMD Early Stopping (when `enabled: true` and run via driver.py)**

1. **Sigma pre-computation**: Before training, `driver.py` computes sigma ONCE from training data using stratified sampling (proportional to condition groups). Saved to `outputs/{dataset}/sigma_heuristic.npz`.

2. **During training**: At model-specific intervals, the model generates synthetic data, computes MMD, and checks early stopping criteria. If MMD doesn't improve by `min_delta` for `patience_mmd` consecutive intervals, training stops.

3. **Checkpoint alignment**: When MMD is active, checkpoint saving aligns with MMD eval intervals, overriding the model's `checkpoint_freq`/`save_freq`. Loss-based early stopping is disabled.

4. **Output**: Each model saves `concurrent_mmd_history.json`:

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

**Mode 2: Passive MMD (when `enabled: true` but no sigma_path, e.g., running scripts directly)**

Falls back to the legacy behavior: compute sigma from validation data, log MMD at intervals, but no early stopping. Saves `mmd_history.json`.

### Model-Specific Details

| Model | Data Space | Trigger | Early Stop | Notes |
|-------|------------|---------|------------|-------|
| CVAE | Gene expression | Every `cvae_freq` epochs | Yes (via callback return) | Callback returns `"stop"` to break training loop |
| Diffusion | Gene expression (VAE decoded) | Every `diffusion_freq` epochs | Yes (dual-mode train loop) | `mmd_tracker` param controls mode |
| GAN | Gene expression | Every `gan_freq` epochs | Yes (breaks step loop) | `save_freq` replaced by `gan_freq`; supports epoch-based saving |

### GAN save_freq Unit Support

The GAN now supports epoch-based checkpoint saving via `--save-freq-unit`:

- `--save-freq-unit steps` (default): Checkpoints at step intervals (e.g., every 500 steps)
- `--save-freq-unit epochs`: Checkpoints at epoch boundaries (e.g., every 5 epochs)

When `training_unit: "epochs"` in config, the driver automatically sets `--save-freq-unit epochs`. When MMD is enabled, `gan_freq` replaces `save_freq` in the same unit.

### Key Files

- `utils/mmd.py` - Core MMD computation (`compute_sigma_from_real_data`, `compute_sigma_stratified`, `compute_mmd`)
- `utils/mmd_early_stopping.py` - `MMDEarlyStoppingTracker` class, sigma save/load utilities
- `driver.py` - `ensure_sigma_heuristic()` pre-computes sigma from training data
- `models/CVAE/train.py` - `evaluate_mmd_cvae()` + early stopping callback
- `models/Diffusion/diffusion_training_utils.py` - `DiffusionTrainer.evaluate_mmd()` + dual-mode `train()`
- `models/GAN/train.py` - `evaluate_mmd_gan()` + epoch-based checkpoint + MMD stop
- `utils/posthoc_mmd.py` - Loads sigma from `sigma_heuristic.npz` when available

### Interpretation

- **Lower MMD = better** - synthetic data more closely matches real data distribution
- MMD should generally decrease during training
- Compare final "best" MMD values across models to assess relative quality
- Sigma is pre-computed from training data and shared by all models for fair comparison
- Post-hoc MMD (`--stage posthoc-mmd`) also uses the shared sigma when available
