#!/usr/bin/env python3
"""
Debug script to inspect saved VAE weights for OSDR-402.
Prints weight statistics to verify model integrity.
"""

import torch
import sys
from pathlib import Path

# VAE model path
VAE_MODEL_PATH = Path("outputs/OSDR-402/trained_models/diffusion/run_20251201_121858/vae_model.pt")

def inspect_vae_weights(model_path: Path):
    """Load and inspect VAE weights, printing key statistics."""
    print("=" * 60)
    print("VAE WEIGHT INSPECTION")
    print("=" * 60)
    print(f"\nModel path: {model_path}")
    print(f"Exists: {model_path.exists()}")

    if not model_path.exists():
        print("[ERROR] Model file not found!")
        return None

    # Load state dict
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        print(f"\n[OK] Loaded state dict successfully")
        print(f"Type: {type(state_dict)}")
        print(f"Number of keys: {len(state_dict)}")
    except Exception as e:
        print(f"[ERROR] Failed to load: {e}")
        return None

    # Print all keys and shapes
    print("\n" + "-" * 40)
    print("STATE DICT KEYS AND SHAPES:")
    print("-" * 40)

    total_params = 0
    for key, value in state_dict.items():
        shape = tuple(value.shape) if hasattr(value, 'shape') else 'N/A'
        num_params = value.numel() if hasattr(value, 'numel') else 0
        total_params += num_params
        print(f"  {key}: {shape} ({num_params:,} params)")

    print(f"\nTotal parameters: {total_params:,}")

    # Print statistics for key layers
    print("\n" + "-" * 40)
    print("KEY LAYER STATISTICS:")
    print("-" * 40)

    key_layers = [
        'encoder.0.weight',  # First encoder layer
        'z_mean.weight',     # Latent mean projection
        'z_log_var.weight',  # Latent log variance projection
        'decoder.0.weight',  # First decoder layer
    ]

    for layer_name in key_layers:
        if layer_name in state_dict:
            weight = state_dict[layer_name]
            print(f"\n{layer_name}:")
            print(f"  Shape: {tuple(weight.shape)}")
            print(f"  Mean: {weight.mean().item():.6f}")
            print(f"  Std: {weight.std().item():.6f}")
            print(f"  Min: {weight.min().item():.6f}")
            print(f"  Max: {weight.max().item():.6f}")
            print(f"  Sum: {weight.sum().item():.6f}")
            print(f"  Abs Sum: {weight.abs().sum().item():.6f}")
            # First few values for fingerprinting
            flat = weight.flatten()
            print(f"  First 5 values: {flat[:5].tolist()}")

    # Generate a unique fingerprint
    print("\n" + "-" * 40)
    print("MODEL FINGERPRINT (for comparison):")
    print("-" * 40)

    # Create fingerprint from multiple layers
    fingerprint_values = []
    for key in ['encoder.0.weight', 'z_mean.weight', 'decoder.0.weight']:
        if key in state_dict:
            w = state_dict[key]
            fingerprint_values.extend([
                w.mean().item(),
                w.std().item(),
                w.sum().item(),
            ])

    fingerprint_str = "_".join([f"{v:.6f}" for v in fingerprint_values[:9]])
    print(f"Fingerprint: {fingerprint_str}")

    return state_dict


def main():
    # Get model path
    script_dir = Path(__file__).parent
    model_path = script_dir / VAE_MODEL_PATH

    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])

    state_dict = inspect_vae_weights(model_path)

    if state_dict:
        print("\n" + "=" * 60)
        print("EXPECTED ARCHITECTURE:")
        print("=" * 60)

        # Infer architecture from state dict
        if 'encoder.0.weight' in state_dict:
            enc_w = state_dict['encoder.0.weight']
            n_genes = enc_w.shape[1]
            first_hidden = enc_w.shape[0]
            print(f"  n_genes: {n_genes}")
            print(f"  first hidden dim: {first_hidden}")

        if 'z_mean.weight' in state_dict:
            z_w = state_dict['z_mean.weight']
            latent_size = z_w.shape[0]
            last_hidden = z_w.shape[1]
            print(f"  latent_size: {latent_size}")
            print(f"  last hidden dim: {last_hidden}")

        print("\n[INFO] Use this fingerprint to verify weights match in generate.py")


if __name__ == "__main__":
    main()
