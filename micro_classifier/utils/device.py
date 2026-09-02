"""Device selection utility (supports CUDA, Apple Silicon MPS, and CPU)."""

import torch
from typing import Optional


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Resolve the best available compute device.

    Priority:
        1. Explicitly requested device (if provided and valid)
        2. CUDA (NVIDIA GPU)
        3. MPS (Apple Silicon GPU)
        4. CPU

    Args:
        device: Optional device string like "cuda", "mps", "cpu", or None.

    Returns:
        Resolved torch.device.
    """
    if device is not None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{device}' but CUDA is not available."
            )
        if requested.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                f"Requested device '{device}' but MPS is not available."
            )
        return requested

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_name() -> str:
    """Human-readable name of the active device."""
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "Apple Silicon GPU (MPS)"
    return "CPU"
