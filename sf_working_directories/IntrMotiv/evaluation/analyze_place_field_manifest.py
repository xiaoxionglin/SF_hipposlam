"""Derive active-only DG field-diversity metrics from a telemetry manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from hpc_runs.intrmotiv_study.spatial_contract import (
    FIELD_MIN_ACTIVE_BINS,
    FIELD_MIN_ACTIVE_OBSERVATIONS,
    FIELD_MONO_MASS_FRACTION,
    FIELD_THRESHOLD_FRACTIONS,
    _binomial_smooth,
    _component_labels,
)
from sf_working_directories.IntrMotiv.evaluation.summarize_place_fields import summarize_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw/<run>/place_fields.npz")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--terminal-target-frames", type=int, default=100_000_000)
    parser.add_argument("--trajectory-seed", type=int, default=99)
    return parser.parse_args()


def artifact_for_suffix(raw_dir: Path, suffix: str) -> Path:
    matches = list(raw_dir.glob(f"*__{suffix}/place_fields.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one artifact for {suffix}, found {len(matches)}")
    return matches[0]


def pairwise_map_cosine(maps: np.ndarray, occupancy: np.ndarray, units: np.ndarray | None = None) -> float:
    samples = np.nan_to_num(maps[occupancy > 0].T, nan=0.0)
    if units is not None:
        samples = samples[units]
    if len(samples) < 2:
        return np.nan
    norms = np.linalg.norm(samples, axis=1)
    denominator = norms[:, None] * norms[None, :]
    cosine = np.divide(
        samples @ samples.T,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 1e-12,
    )
    values = cosine[np.triu_indices(len(samples), k=1)]
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def peak_statistics(
    maps: np.ndarray,
    units: np.ndarray,
    *,
    require_positive: bool,
) -> tuple[int, float, float]:
    peaks: list[np.ndarray] = []
    for unit in units:
        rate_map = maps[:, :, unit]
        if not np.isfinite(rate_map).any():
            continue
        index = int(np.nanargmax(rate_map))
        if require_positive and float(rate_map.reshape(-1)[index]) <= 0:
            continue
        peaks.append(np.asarray(np.unravel_index(index, rate_map.shape), dtype=np.float64))

    if not peaks:
        return 0, 0.0, np.nan

    peak_array = np.stack(peaks)
    _, counts = np.unique(peak_array, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum() / np.log(len(peaks))) if len(peaks) > 1 else 0.0
    distances = [
        float(np.linalg.norm(peak_array[first] - peak_array[second]))
        for first in range(len(peak_array))
        for second in range(first + 1, len(peak_array))
    ]
    return len(counts), entropy, float(np.mean(distances)) if distances else np.nan


def connected_components_above_half_peak(rate_map: np.ndarray) -> int:
    """Count four-connected positive components at half of a unit's peak."""
    finite = np.isfinite(rate_map)
    if not finite.any():
        return 0
    peak = float(np.nanmax(rate_map))
    if peak <= 0:
        return 0
    active = finite & (rate_map >= 0.5 * peak)
    visited = np.zeros_like(active, dtype=bool)
    components = 0
    for row, column in zip(*np.nonzero(active)):
        if visited[row, column]:
            continue
        components += 1
        stack = [(int(row), int(column))]
        visited[row, column] = True
        while stack:
            current_row, current_column = stack.pop()
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if not (0 <= next_row < active.shape[0] and 0 <= next_column < active.shape[1]):
                    continue
                if active[next_row, next_column] and not visited[next_row, next_column]:
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
    return components


def multilevel_field_structure(
    rate_maps: np.ndarray,
    occupancy: np.ndarray,
    active_fraction: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply the online 8-connected mono-field contract to offline rate maps."""
    maps = np.moveaxis(np.nan_to_num(rate_maps, nan=0.0), -1, 0).astype(np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)
    smooth_occupancy = _binomial_smooth(occupancy[None])[0]
    smooth_sums = _binomial_smooth(maps * occupancy[None])
    smoothed = np.divide(
        smooth_sums,
        smooth_occupancy[None],
        out=np.zeros_like(smooth_sums),
        where=smooth_occupancy[None] > 0,
    )
    smoothed[:, occupancy == 0] = 0
    active_observations = np.rint(np.asarray(active_fraction) * occupancy.sum()).astype(np.int64)
    active_bins = (maps > 0).sum(axis=(1, 2))
    eligible = (active_observations >= FIELD_MIN_ACTIVE_OBSERVATIONS) & (active_bins >= FIELD_MIN_ACTIVE_BINS)
    fractions = np.asarray(FIELD_THRESHOLD_FRACTIONS, dtype=np.float64)
    component_count = np.zeros((len(fractions), maps.shape[0]), dtype=np.int16)
    dominant_mass = np.zeros((len(fractions), maps.shape[0]), dtype=np.float32)
    for unit, unit_map in enumerate(smoothed):
        peak = float(unit_map.max())
        if not eligible[unit] or peak <= 0:
            continue
        for level, fraction in enumerate(fractions):
            labels, count = _component_labels((unit_map >= fraction * peak) & (occupancy > 0))
            component_count[level, unit] = count
            masses = np.asarray(
                [unit_map[labels == component].sum() for component in range(1, count + 1)],
                dtype=np.float64,
            )
            if masses.size and masses.sum() > 0:
                dominant_mass[level, unit] = float(masses.max() / masses.sum())
    mono_score = dominant_mass.min(axis=0)
    mono = eligible & (mono_score >= FIELD_MONO_MASS_FRACTION)
    return {
        "eligible": eligible,
        "component_count": component_count,
        "dominant_mass": dominant_mass,
        "mono_score": mono_score,
        "mono": mono,
    }


def finite_correlation(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 2 or np.std(first[valid]) <= 1e-12 or np.std(second[valid]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(first[valid], second[valid])[0, 1])


def per_unit_rows(item: dict[str, str], artifact: Path) -> list[dict[str, object]]:
    """Join spatial selectivity with optional graph/recruitment state by DG row."""
    data = np.load(artifact, allow_pickle=False)
    rate_maps = data["rate_maps"]
    n_units = rate_maps.shape[-1]
    confidence = data["control_edge_confidence"] if "control_edge_confidence" in data else None
    attempts = data["control_attempts"] if "control_attempts" in data else None
    tctrl = data["control_tctrl"] if "control_tctrl" in data else None
    structure = multilevel_field_structure(rate_maps, data["occupancy"], data["active_fraction"])
    reliable = None
    if confidence is not None and attempts is not None and tctrl is not None:
        reliability = (confidence + 1.0) / (attempts + 2.0)
        reliable = (tctrl > 0) & (confidence >= 0.5) & (reliability >= 0.5)
        np.fill_diagonal(reliable, False)

    output = []
    for unit in range(n_units):
        row = {
            **item,
            "unit": unit,
            "spatial_information_bits": float(data["spatial_information"][unit]),
            "active_fraction": float(data["active_fraction"][unit]),
            "connected_components_half_peak": connected_components_above_half_peak(rate_maps[:, :, unit]),
            "field_eligible": bool(structure["eligible"][unit]),
            "mono_field": bool(structure["mono"][unit]),
            "mono_score": float(structure["mono_score"][unit]),
            "field_components_30pct": int(structure["component_count"][0, unit]),
            "field_components_50pct": int(structure["component_count"][1, unit]),
            "field_components_70pct": int(structure["component_count"][2, unit]),
            "dominant_component_mass_30pct": float(structure["dominant_mass"][0, unit]),
            "dominant_component_mass_50pct": float(structure["dominant_mass"][1, unit]),
            "dominant_component_mass_70pct": float(structure["dominant_mass"][2, unit]),
            "field_spread": float(1.0 - structure["mono_score"][unit]),
        }
        if reliable is not None:
            off_diagonal = np.arange(n_units) != unit
            row.update(
                reliable_out_degree=int(reliable[unit].sum()),
                reliable_in_degree=int(reliable[:, unit].sum()),
                reliable_outgoing_confidence=float(confidence[unit][reliable[unit]].sum()),
                reliable_incoming_confidence=float(confidence[:, unit][reliable[:, unit]].sum()),
                outgoing_attempt_coverage_fraction=float((attempts[unit][off_diagonal] >= 0.5).mean()),
            )
        for output_name in ("birth_support", "recruitment_row_counts"):
            if output_name in data:
                row[output_name] = float(data[output_name][unit])
        output.append(row)
    return output


def derive_row(item: dict[str, str], artifact: Path) -> dict[str, object]:
    data = np.load(artifact, allow_pickle=False)
    occupancy = data["occupancy"]
    rate_maps = data["rate_maps"]
    active_units = np.flatnonzero(data["active_fraction"] > 0)
    threshold_peaks = peak_statistics(rate_maps, active_units, require_positive=True)
    information = data["spatial_information"]
    components = np.asarray(
        [connected_components_above_half_peak(rate_maps[:, :, unit]) for unit in range(rate_maps.shape[-1])]
    )
    structure = multilevel_field_structure(rate_maps, occupancy, data["active_fraction"])
    eligible = structure["eligible"]
    incoming_spread_correlation = np.nan
    if "control_edge_confidence" in data:
        incoming_confidence = np.asarray(data["control_edge_confidence"], dtype=np.float64).sum(axis=0)
        incoming_spread_correlation = finite_correlation(
            incoming_confidence[eligible], (1.0 - structure["mono_score"])[eligible]
        )

    derived: dict[str, object] = {
        **item,
        **summarize_artifact(artifact),
        "active_units": int(len(active_units)),
        "active_unit_mean_si_bits": float(np.mean(information[active_units])) if len(active_units) else 0.0,
        "active_map_cosine_mean": pairwise_map_cosine(rate_maps, occupancy, active_units),
        "active_unique_peak_bins": threshold_peaks[0],
        "active_peak_bin_entropy": threshold_peaks[1],
        "active_pairwise_peak_distance_bins": threshold_peaks[2],
        "active_mean_connected_components_half_peak": (
            float(components[active_units].mean()) if len(active_units) else 0.0
        ),
        "field_eligible_units": int(eligible.sum()),
        "mono_field_fraction": float(structure["mono"][eligible].mean()) if eligible.any() else 0.0,
        "mean_components_30pct": float(structure["component_count"][0, eligible].mean()) if eligible.any() else 0.0,
        "mean_components_50pct": float(structure["component_count"][1, eligible].mean()) if eligible.any() else 0.0,
        "mean_components_70pct": float(structure["component_count"][2, eligible].mean()) if eligible.any() else 0.0,
        "mean_dominant_component_mass": (
            float(structure["dominant_mass"][:, eligible].mean()) if eligible.any() else 0.0
        ),
        "incoming_confidence_field_spread_correlation": incoming_spread_correlation,
    }

    if "pre_threshold_rate_maps" in data:
        pre_threshold_maps = data["pre_threshold_rate_maps"]
        all_units = np.arange(pre_threshold_maps.shape[-1])
        pre_threshold_peaks = peak_statistics(pre_threshold_maps, all_units, require_positive=False)
        derived.update(
            prethreshold_map_cosine_mean=pairwise_map_cosine(pre_threshold_maps, occupancy),
            prethreshold_unique_peak_bins=pre_threshold_peaks[0],
            prethreshold_peak_bin_entropy=pre_threshold_peaks[1],
            prethreshold_pairwise_peak_distance_bins=pre_threshold_peaks[2],
        )
    return derived


def main() -> None:
    args = parse_args()
    with args.manifest.open(encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))

    artifacts = [artifact_for_suffix(args.input_dir / "raw", item["label_suffix"]) for item in manifest]
    rows = [derive_row(item, artifact) for item, artifact in zip(manifest, artifacts)]
    units = [row for item, artifact in zip(manifest, artifacts) for row in per_unit_rows(item, artifact)]
    frame = pd.DataFrame(rows)
    for column in ("seed", "target_frames", "checkpoint_frames"):
        frame[column] = frame[column].astype(int)
    frame = frame.sort_values(["condition", "seed", "checkpoint_frames"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "derived_place_field_metrics.csv", index=False)
    pd.DataFrame(units).to_csv(args.out_dir / "per_unit_place_field_metrics.csv", index=False)

    metric_columns = [
        "visited_cells",
        "mean_spatial_information_bits",
        "active_unit_mean_si_bits",
        "mean_active_fraction",
        "silent_units",
        "active_map_cosine_mean",
        "active_unique_peak_bins",
        "active_peak_bin_entropy",
        "active_pairwise_peak_distance_bins",
        "active_mean_connected_components_half_peak",
        "field_eligible_units",
        "mono_field_fraction",
        "mean_components_30pct",
        "mean_components_50pct",
        "mean_components_70pct",
        "mean_dominant_component_mass",
        "incoming_confidence_field_spread_correlation",
        "prethreshold_map_cosine_mean",
        "prethreshold_unique_peak_bins",
        "prethreshold_peak_bin_entropy",
        "prethreshold_pairwise_peak_distance_bins",
    ]
    terminal = frame[frame["target_frames"] == args.terminal_target_frames]
    aggregate = terminal.groupby("condition")[metric_columns].agg(["mean", "std"])
    aggregate.to_csv(args.out_dir / "terminal_three_seed_aggregate.csv")

    trajectory = frame[frame["seed"] == args.trajectory_seed]
    trajectory.to_csv(args.out_dir / "seed99_checkpoint_trajectory.csv", index=False)
    print(f"Wrote {len(frame)} checkpoint rows and {len(terminal)} terminal rows to {args.out_dir}")


if __name__ == "__main__":
    main()
