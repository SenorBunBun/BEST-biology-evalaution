"""
Config loader for the scRNA-seq synthetic data generation pipeline.
Loads JSON configs, merges with defaults, and validates required fields.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy

# Pipeline root directory (where this file lives)
PIPELINE_ROOT = Path(__file__).parent.resolve()


# Default configuration values
DEFAULTS = {
    "description": "",

    "data_split": {
        "val_ratio": 0.2,
        "seed": 42
    },

    "mmd_eval": {
        "enabled": False,
        "cvae_freq": 5,            # CVAE: eval every N epochs (also controls checkpoint saving when MMD active)
        "diffusion_freq": 5,       # Diffusion: eval every N epochs
        "gan_freq": 1,             # GAN: eval every N epochs
        "patience_mmd": 3,         # Intervals without improvement before stopping
        "min_delta": 0.001,        # Minimum MMD decrease to count as improvement
        "max_sigma_samples": 1000  # Max samples for sigma computation
    },

    "preprocessing": {
        "enabled": True,
        "min_cells": 3,
        "sparsity_threshold": 0.997,
        "n_hvg": 2000,
        "normalize": True,
        "target_sum": 1e4
    },

    "models": {
        "cvae": {
            "enabled": True,
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
            "enabled": True,
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
            "enabled": True,
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
            "max_steps": 300000,
            "critic_iters": 4,
            "train_split": 0.85,
            "save_freq": 500,
            "log_freq": 100,
            "validation_freq": 1,
            "clip_gradients": True,
            "gradient_clip_norm": 1.0
        }
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
        "deg": {
            "enabled": True,
            "method": "wilcoxon",
            "n_genes": 50
        },
        "umap": {
            "enabled": True,
            "n_neighbors": 15,
            "min_dist": 0.1
        },
        "metrics": {
            "enabled": True
        },
        "classifier": {
            "enabled": True,
            "n_genes": 50,
            "n_estimators": 100,
            "max_depth": 10,
            "n_folds": 5,
            "n_repeats": 3
        }
    }
}


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def validate_config(config: Dict) -> None:
    """Validate that required fields are present in config."""
    required_fields = [
        ("dataset_name", "Root level 'dataset_name' is required"),
        ("paths", "Root level 'paths' is required"),
        ("paths.processed_data", "'paths.processed_data' is required"),
        ("paths.output_dir", "'paths.output_dir' is required"),
        ("conditions", "Root level 'conditions' is required"),
        ("conditions.columns", "'conditions.columns' is required"),
        ("conditions.primary", "'conditions.primary' is required"),
        ("conditions.primary.column", "'conditions.primary.column' is required"),
        ("conditions.primary.group1", "'conditions.primary.group1' is required"),
        ("conditions.primary.group2", "'conditions.primary.group2' is required"),
    ]

    for field_path, error_msg in required_fields:
        parts = field_path.split(".")
        obj = config
        for part in parts:
            if not isinstance(obj, dict) or part not in obj:
                raise ValueError(f"Config validation error: {error_msg}")
            obj = obj[part]


def resolve_paths(config: Dict) -> Dict:
    """Resolve relative paths in config to absolute paths (relative to PIPELINE_ROOT)."""
    config = deepcopy(config)

    # Resolve paths section - all relative paths are relative to PIPELINE_ROOT
    if "paths" in config:
        for key in ["raw_data", "processed_data", "output_dir"]:
            if key in config["paths"] and config["paths"][key] is not None:
                path = Path(config["paths"][key])
                if not path.is_absolute():
                    config["paths"][key] = str(PIPELINE_ROOT / path)

    return config


def load_config(config_path: str, resolve_relative: bool = True) -> Dict:
    """
    Load configuration from JSON file, merge with defaults, and validate.

    Args:
        config_path: Path to the JSON config file
        resolve_relative: Whether to resolve relative paths to absolute

    Returns:
        Merged and validated configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        user_config = json.load(f)

    # Merge with defaults
    config = deep_merge(DEFAULTS, user_config)

    # Resolve relative paths (relative to PIPELINE_ROOT, not config file location)
    if resolve_relative:
        config = resolve_paths(config)

    # Validate required fields
    validate_config(config)

    # Store the original config path and pipeline root for reference
    config["_config_path"] = str(config_path.absolute())
    config["_pipeline_root"] = str(PIPELINE_ROOT)

    return config


def get_output_paths(config: Dict) -> Dict:
    """
    Generate all output paths based on config.

    Returns dict with paths for:
    - trained_models/{model}/
    - synthetic_data/
    - analysis/{type}/
    """
    base_dir = Path(config["paths"]["output_dir"])
    dataset_name = config["dataset_name"]

    paths = {
        "base": base_dir,
        "trained_models": {
            "cvae": base_dir / "trained_models" / "cvae",
            "diffusion": base_dir / "trained_models" / "diffusion",
            "gan": base_dir / "trained_models" / "gan",
        },
        "synthetic_data": base_dir / "synthetic_data",
        "analysis": {
            "base": base_dir / "analysis",
            "deg": base_dir / "analysis" / "deg",
            "umap": base_dir / "analysis" / "umap",
            "metrics": base_dir / "analysis" / "metrics",
            "classifier": base_dir / "analysis" / "classifier",
        }
    }

    return paths


def create_output_dirs(config: Dict) -> Dict:
    """Create all output directories and return paths dict."""
    paths = get_output_paths(config)

    # Create all directories
    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["synthetic_data"].mkdir(parents=True, exist_ok=True)

    for model_dir in paths["trained_models"].values():
        model_dir.mkdir(parents=True, exist_ok=True)

    # Create analysis directories (including base)
    for key, analysis_dir in paths["analysis"].items():
        analysis_dir.mkdir(parents=True, exist_ok=True)

    return paths


def save_config_snapshot(config: Dict, output_path: Path) -> None:
    """Save a copy of the config for reproducibility."""
    # Remove internal fields
    config_copy = {k: v for k, v in config.items() if not k.startswith("_")}

    with open(output_path, 'w') as f:
        json.dump(config_copy, f, indent=2)


if __name__ == "__main__":
    # Test config loading
    import sys

    if len(sys.argv) > 1:
        config = load_config(sys.argv[1])
        print(json.dumps(config, indent=2, default=str))
    else:
        print("Usage: python config_loader.py <config_path>")
        print("\nDefault configuration:")
        print(json.dumps(DEFAULTS, indent=2))
