"""Offline checkpoint loading without importing evaluation or environment code."""

import pathlib

import numpy as np
import torch


def load_checkpoint_dict(checkpoint: pathlib.Path, device: torch.device) -> dict:
    """Safely load an IntrMotiv training checkpoint for offline evaluation.

    Training checkpoints include NumPy scalar metadata. PyTorch 2.6 excludes
    those classes from its default ``weights_only`` allowlist, even though the
    tensor model state itself is safe to load. Retain the weights-only loader
    and allowlist only the exact NumPy scalar and dtype classes found in these
    locally produced checkpoints.
    """
    safe_globals = [np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(safe_globals):
        # CPU audits/evaluation need model and metadata, not eager copies of
        # gigabytes of replay tensors. Torch's private mmap preserves tensor
        # values and safe unpickling while faulting storage pages on demand.
        return torch.load(checkpoint, map_location=device, weights_only=True, mmap=device.type == "cpu")
