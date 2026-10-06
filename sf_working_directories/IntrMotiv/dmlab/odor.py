"""Four smooth spatial odor channels; position is never exposed to the policy."""

from __future__ import annotations

import numpy as np

ODOR_CENTERS = np.asarray(((575, 575), (575, 1525), (1525, 575), (1525, 1525)), dtype=np.float64)
ODOR_SPATIAL_SIGMA = 350.0


def clean_odor(position: np.ndarray) -> np.ndarray:
    xy = np.asarray(position, dtype=np.float64)[:2]
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Odor requires a finite two-dimensional engine position")
    squared = np.sum((ODOR_CENTERS - xy) ** 2, axis=-1)
    return np.exp(-squared / (2 * ODOR_SPATIAL_SIGMA**2)).astype(np.float32)


def odor_observation(position: np.ndarray | None, mode: str, rng: np.random.Generator, noise_std: float) -> np.ndarray:
    if mode == "zero":
        return np.zeros(4, dtype=np.float32)
    if mode != "gaussian4" or position is None or not 0 <= noise_std <= 1:
        raise ValueError("Invalid odor mode, position, or noise scale")
    return clean_odor(position) + rng.normal(0, noise_std, 4).astype(np.float32)
