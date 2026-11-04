"""Gain-invariant spatial concentration from saved occupancy-corrected maps.

These shape measures complement the connected-component dominance score in
``spatial_contract``. They do not classify monofields: permuting spatial bins
leaves them unchanged, including when one compact patch becomes several peaks.
No rollout or runtime dependency is needed, and existing NPZ schemas are intact.
"""
from __future__ import annotations

import numpy as np


def calculate_field_concentration(
    rate_maps: np.ndarray,
    occupancy: np.ndarray,
    *,
    minimum_bin_observations: int = 1,
) -> dict[str, np.ndarray]:
    """Return per-unit effective area, concentration, entropy, and 80%-mass area.

    ``rate_maps`` has shape (rows, columns, units), with nonnegative mean activity
    at each occupied bin. Each supported bin gets equal spatial weight; these
    rates already correct for the number of visits. Unvisited bins are unknown,
    never evidence for zero activity. Increasing the support minimum is an
    explicitly reported sensitivity analysis, not an automatic filter.

    With N supported bins and p_i = rate_i / sum(rate), effective area is
    A_eff = 1 / sum(p_i**2), in bin equivalents. Concentration is
    (N - A_eff) / (N - 1): zero for uniform activity, one for a single-bin peak.
    Entropy concentration is 1 - H(p)/log(N). The 80%-mass area fraction uses
    descending rates and linearly interpolates the final included bin.

    Silent units have undefined shape and return NaN. Fewer than two supported
    bins also yield undefined normalized concentration. Field eligibility and
    unit aggregation are the caller's responsibility, using canonical metadata.
    """
    maps = np.asarray(rate_maps, dtype=np.float64)
    counts = np.asarray(occupancy, dtype=np.float64)
    if maps.ndim != 3 or maps.shape[:2] != counts.shape:
        raise ValueError("rate maps must be (rows, columns, units), matching occupancy")
    if not isinstance(minimum_bin_observations, (int, np.integer)) or minimum_bin_observations < 1:
        raise ValueError("minimum bin observations must be a positive integer")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("occupancy must be finite and nonnegative")
    supported = counts >= minimum_bin_observations
    rates = maps[supported]
    if not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("supported map rates must be finite and nonnegative")
    n_bins, n_units = rates.shape
    outputs = {name: np.full(n_units, np.nan, dtype=np.float64) for name in (
        "effective_area_bins", "effective_area_fraction", "concentration",
        "entropy_concentration", "mass80_area_fraction",
    )}
    outputs["supported_bins"] = np.asarray(n_bins, dtype=np.int64)
    if n_bins == 0:
        return outputs
    maxima = rates.max(axis=0)
    valid = maxima > 0
    if not valid.any():
        return outputs
    # Scaling by the peak first also avoids overflow under large activity gains.
    normalized = rates[:, valid] / maxima[valid]
    probabilities = normalized / normalized.sum(axis=0)
    area = 1. / np.square(probabilities).sum(axis=0)
    outputs["effective_area_bins"][valid] = area
    outputs["effective_area_fraction"][valid] = area / n_bins
    if n_bins >= 2:
        outputs["concentration"][valid] = np.clip((n_bins - area) / (n_bins - 1), 0, 1)
        log_p = np.log(probabilities, out=np.zeros_like(probabilities), where=probabilities > 0)
        entropy = -(probabilities * log_p).sum(axis=0)
        outputs["entropy_concentration"][valid] = np.clip(1 - entropy / np.log(n_bins), 0, 1)
    ordered = np.sort(probabilities, axis=0)[::-1]
    cumulative = ordered.cumsum(axis=0)
    crossing = (cumulative >= .8).argmax(axis=0)
    columns = np.arange(ordered.shape[1])
    prior = np.where(crossing > 0, cumulative[np.maximum(crossing - 1, 0), columns], 0)
    partial_bin = (.8 - prior) / ordered[crossing, columns]
    outputs["mass80_area_fraction"][valid] = (crossing + partial_bin) / n_bins
    return outputs
