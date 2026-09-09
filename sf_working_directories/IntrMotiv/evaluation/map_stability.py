"""Compare occupancy-corrected DG rate maps across retained checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


MIN_PEAK_OCCUPANCY = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def artifact_path(raw_dir: Path, suffix: str) -> Path:
    candidates = list(raw_dir.glob(f"*__{suffix}/place_fields.npz"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one artifact for {suffix}, found {len(candidates)}")
    return candidates[0]


def weighted_pearson(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    weight = weight.astype(np.float64)
    total = weight.sum()
    if total <= 0:
        return np.nan
    x_mean = np.sum(weight * x) / total
    y_mean = np.sum(weight * y) / total
    x_centered = x - x_mean
    y_centered = y - y_mean
    denom = np.sqrt(np.sum(weight * x_centered**2) * np.sum(weight * y_centered**2))
    return float(np.sum(weight * x_centered * y_centered) / denom) if denom > 1e-12 else np.nan


def occupancy_supported_peak(rate_map: np.ndarray, occupancy: np.ndarray) -> tuple[float, float, float]:
    """Return the rate-map peak only among cells sampled at least three times."""
    valid = np.isfinite(rate_map) & (occupancy >= MIN_PEAK_OCCUPANCY)
    if not np.any(valid):
        return np.nan, np.nan, np.nan
    masked = np.where(valid, rate_map, np.nan)
    peak_value = float(np.nanmax(masked))
    if peak_value <= 0:
        return np.nan, np.nan, peak_value
    x_bin, y_bin = np.unravel_index(np.nanargmax(masked), masked.shape)
    return float(x_bin), float(y_bin), peak_value


def main() -> None:
    args = parse_args()
    rows = [row for row in load_manifest(args.manifest) if row["condition"] == args.condition]
    if len(rows) < 2:
        raise ValueError(f"Need at least two checkpoints for {args.condition}")
    rows.sort(key=lambda row: int(row["checkpoint_frames"]))
    artifacts = [(row, np.load(artifact_path(args.input_dir / "raw", row["label_suffix"]), allow_pickle=False)) for row in rows]
    reference_row, reference = artifacts[-1]
    reference_occupancy = reference["occupancy"]
    reference_maps = reference["rate_maps"]
    reference_peaks = [
        occupancy_supported_peak(reference_maps[:, :, unit], reference_occupancy)
        for unit in range(reference_maps.shape[-1])
    ]
    output: list[dict[str, object]] = []
    for row, data in artifacts:
        occupancy = data["occupancy"]
        rate_maps = data["rate_maps"]
        overlap = (occupancy > 0) & (reference_occupancy > 0)
        weights = np.minimum(occupancy[overlap], reference_occupancy[overlap])
        for unit in range(rate_maps.shape[-1]):
            current = rate_maps[:, :, unit][overlap]
            final = reference_maps[:, :, unit][overlap]
            valid = np.isfinite(current) & np.isfinite(final) & (weights > 0)
            peak_x, peak_y, peak_rate = occupancy_supported_peak(rate_maps[:, :, unit], occupancy)
            final_x, final_y, final_rate = reference_peaks[unit]
            peak_shift = (
                float(np.hypot(peak_x - final_x, peak_y - final_y))
                if np.isfinite(peak_x) and np.isfinite(final_x)
                else np.nan
            )
            output.append(
                {
                    "condition": args.condition,
                    "checkpoint_frames": int(row["checkpoint_frames"]),
                    "reference_frames": int(reference_row["checkpoint_frames"]),
                    "dg_unit": unit,
                    "overlap_cells": int(valid.sum()),
                    "shared_occupancy": float(weights[valid].sum()),
                    "weighted_rate_map_correlation": weighted_pearson(current[valid], final[valid], weights[valid]),
                    "peak_x_bin": peak_x,
                    "peak_y_bin": peak_y,
                    "peak_rate": peak_rate,
                    "final_peak_x_bin": final_x,
                    "final_peak_y_bin": final_y,
                    "final_peak_rate": final_rate,
                    "peak_shift_to_final_bins": peak_shift,
                    "peak_shift_to_final_dmlab_units": peak_shift * 100.0 if np.isfinite(peak_shift) else np.nan,
                }
            )
    frame = pd.DataFrame(output)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "per_unit_rate_map_stability.csv", index=False)
    summary = (
        frame.groupby("checkpoint_frames", as_index=False)
        .agg(
            mean_correlation=("weighted_rate_map_correlation", "mean"),
            median_correlation=("weighted_rate_map_correlation", "median"),
            valid_units=("weighted_rate_map_correlation", "count"),
            overlap_cells=("overlap_cells", "first"),
            shared_occupancy=("shared_occupancy", "first"),
            mean_peak_shift_bins=("peak_shift_to_final_bins", "mean"),
            median_peak_shift_bins=("peak_shift_to_final_bins", "median"),
            valid_peak_units=("peak_shift_to_final_bins", "count"),
        )
        .sort_values("checkpoint_frames")
    )
    summary.to_csv(args.out_dir / "rate_map_stability_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
