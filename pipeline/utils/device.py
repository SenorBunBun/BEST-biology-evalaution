"""
Device selection utility.

Picks the best available PyTorch device, preferring CUDA (NVIDIA) when present,
then Apple-Silicon MPS (Metal) on macOS, falling back to CPU. Enabling the MPS
fallback lets any op MPS does not yet implement run on CPU instead of crashing.
"""

import os

# Must be set before the first MPS op dispatch; importing this module early
# (device selection happens at startup) guarantees that.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def get_device():
    """Return the best available torch.device: cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device=None):
    """Human-readable device description for logging."""
    device = device or get_device()
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name()})"
    if device.type == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu"
