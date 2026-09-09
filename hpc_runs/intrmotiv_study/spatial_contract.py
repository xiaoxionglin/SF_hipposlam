"""Shared contract and calculations for compact online spatial telemetry."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


SNAPSHOT_SCHEMA = "intrmotiv/online-spatial/v1"
DEFAULT_GRAIN = 19
DEFAULT_BOUNDS = (100.0, 2000.0, 100.0, 2000.0)
FIELD_THRESHOLD_FRACTIONS = (0.30, 0.50, 0.70)
FIELD_MIN_ACTIVE_OBSERVATIONS = 20
FIELD_MIN_ACTIVE_BINS = 3
FIELD_MONO_MASS_FRACTION = 0.80
SNAPSHOT_REQUIRED_ARRAYS = (
    "pose",
    "dg_activity",
    "actions",
    "dones",
    "segment_id",
    "policy_version",
)
SPATIAL_DETAIL_ARRAYS = (
    "occupancy",
    "rate_maps",
    "smoothed_rate_maps",
    "spatial_information",
    "active_fraction",
    "field_threshold_fractions",
    "field_component_labels",
    "field_component_count",
    "field_dominant_mass_fraction",
    "field_active_observation_count",
    "field_active_bin_count",
    "field_eligible",
    "field_mono_score",
    "field_mono",
    "field_dominant_peak_bin",
    "field_secondary_peak_bin",
    "field_dominant_peak_xy",
    "field_secondary_peak_xy",
    "field_primary_secondary_peak_distance",
    "field_dominant_peak_nearest_neighbor_distance",
)
CONTROL_GRAPH_ARRAYS = (
    "control_node_visits",
    "control_tctrl",
    "control_edge_confidence",
    "control_attempts",
    "control_prospective_attempts",
    "control_prospective_successes",
    "control_prospective_probability_sum",
    "control_prospective_brier_sum",
    "control_prospective_timing_count",
    "control_prospective_timing_sum",
    "control_prospective_predicted_timing_sum",
    "control_prospective_timing_absolute_error_sum",
)


class SpatialContractError(ValueError):
    """Raised when an online spatial payload violates the v1 contract."""


@dataclass(frozen=True)
class SpatialBounds:
    x_min: float = DEFAULT_BOUNDS[0]
    x_max: float = DEFAULT_BOUNDS[1]
    y_min: float = DEFAULT_BOUNDS[2]
    y_max: float = DEFAULT_BOUNDS[3]

    def __post_init__(self) -> None:
        values = np.asarray((self.x_min, self.x_max, self.y_min, self.y_max), dtype=np.float64)
        if not np.isfinite(values).all():
            raise SpatialContractError("spatial bounds must be finite")
        if not self.x_min < self.x_max or not self.y_min < self.y_max:
            raise SpatialContractError("spatial bounds require min < max on both axes")

    def as_array(self) -> np.ndarray:
        return np.asarray((self.x_min, self.x_max, self.y_min, self.y_max), dtype=np.float32)


def _aligned_arrays(
    pose: np.ndarray,
    dg_activity: np.ndarray,
    actions: np.ndarray,
    dones: np.ndarray,
    segment_id: np.ndarray,
    policy_version: np.ndarray,
) -> tuple[np.ndarray, ...]:
    pose = np.asarray(pose, dtype=np.float32)
    dg_activity = np.asarray(dg_activity, dtype=np.float32)
    actions = np.asarray(actions)
    dones = np.asarray(dones, dtype=np.bool_).reshape(-1)
    segment_id = np.asarray(segment_id, dtype=np.int64).reshape(-1)
    policy_version = np.asarray(policy_version, dtype=np.int64).reshape(-1)
    if pose.ndim != 2 or pose.shape[1] != 3:
        raise SpatialContractError(f"pose must have shape [N, 3], got {pose.shape}")
    if dg_activity.ndim != 2:
        raise SpatialContractError(f"dg_activity must have shape [N, units], got {dg_activity.shape}")
    n = pose.shape[0]
    if n == 0:
        raise SpatialContractError("spatial telemetry cannot be empty")
    if actions.ndim == 0 or actions.shape[0] != n:
        raise SpatialContractError("actions must have N rows")
    for name, value in (
        ("dg_activity", dg_activity),
        ("dones", dones),
        ("segment_id", segment_id),
        ("policy_version", policy_version),
    ):
        if value.shape[0] != n:
            raise SpatialContractError(f"{name} has {value.shape[0]} rows; expected {n}")
    if not np.isfinite(pose).all():
        raise SpatialContractError("pose contains non-finite values")
    if not np.isfinite(dg_activity).all():
        raise SpatialContractError("dg_activity contains non-finite values")
    if (dg_activity < 0).any():
        raise SpatialContractError("thresholded DG activity must be non-negative")
    return pose, dg_activity, actions, dones, segment_id, policy_version


def spatial_rate_maps(
    pose: np.ndarray,
    dg_activity: np.ndarray,
    bounds: SpatialBounds = SpatialBounds(),
    grain: int = DEFAULT_GRAIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return occupancy-corrected unit maps, occupancy, and in-bounds mask."""

    pose = np.asarray(pose, dtype=np.float32)
    activity = np.asarray(dg_activity, dtype=np.float32)
    if grain <= 0:
        raise SpatialContractError("grain must be positive")
    if pose.ndim != 2 or pose.shape[1] != 3 or activity.ndim != 2 or activity.shape[0] != pose.shape[0]:
        raise SpatialContractError("pose and DG activity must be aligned [N, 3] and [N, units] arrays")
    if not np.isfinite(pose).all() or not np.isfinite(activity).all():
        raise SpatialContractError("pose and DG activity must be finite")

    x = pose[:, 0]
    y = pose[:, 1]
    in_bounds = (
        (x >= bounds.x_min)
        & (x <= bounds.x_max)
        & (y >= bounds.y_min)
        & (y <= bounds.y_max)
    )
    occupancy = np.zeros((grain, grain), dtype=np.int64)
    rate_sums = np.zeros((activity.shape[1], grain, grain), dtype=np.float64)
    if in_bounds.any():
        x_bin = np.floor((x[in_bounds] - bounds.x_min) / (bounds.x_max - bounds.x_min) * grain).astype(int)
        y_bin = np.floor((y[in_bounds] - bounds.y_min) / (bounds.y_max - bounds.y_min) * grain).astype(int)
        x_bin = np.clip(x_bin, 0, grain - 1)
        y_bin = np.clip(y_bin, 0, grain - 1)
        np.add.at(occupancy, (y_bin, x_bin), 1)
        bounded_activity = activity[in_bounds]
        for unit in range(activity.shape[1]):
            np.add.at(rate_sums[unit], (y_bin, x_bin), bounded_activity[:, unit])

    rate_maps = np.divide(
        rate_sums,
        occupancy[None, :, :],
        out=np.zeros_like(rate_sums),
        where=occupancy[None, :, :] > 0,
    ).astype(np.float32)
    return rate_maps, occupancy, in_bounds


def _spatial_information(rate_map: np.ndarray, occupancy: np.ndarray) -> float:
    total = float(occupancy.sum())
    if total <= 0:
        return 0.0
    probability = occupancy.astype(np.float64) / total
    mean_rate = float(np.sum(rate_map * probability))
    if mean_rate <= 0:
        return 0.0
    positive = (rate_map > 0) & (probability > 0)
    ratio = rate_map[positive] / mean_rate
    return float(np.sum(rate_map[positive] * probability[positive] * np.log2(ratio)))


def _binomial_smooth(array: np.ndarray) -> np.ndarray:
    """Apply a zero-padded separable [1, 2, 1] / 4 filter on the last two axes."""

    value = np.asarray(array, dtype=np.float64)
    padded = np.pad(value, ((0, 0),) * (value.ndim - 2) + ((1, 1), (1, 1)))
    result = np.zeros_like(value, dtype=np.float64)
    weights = (1.0, 2.0, 1.0)
    for row, row_weight in enumerate(weights):
        for column, column_weight in enumerate(weights):
            result += row_weight * column_weight * padded[
                ..., row : row + value.shape[-2], column : column + value.shape[-1]
            ]
    return result / 16.0


def _component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label an eight-connected two-dimensional boolean mask."""

    mask = np.asarray(mask, dtype=np.bool_)
    labels = np.zeros(mask.shape, dtype=np.int16)
    next_label = 0
    height, width = mask.shape
    for row, column in np.argwhere(mask):
        if labels[row, column]:
            continue
        next_label += 1
        labels[row, column] = next_label
        stack = [(int(row), int(column))]
        while stack:
            current_row, current_column = stack.pop()
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    if row_delta == 0 and column_delta == 0:
                        continue
                    neighbor_row = current_row + row_delta
                    neighbor_column = current_column + column_delta
                    if not (0 <= neighbor_row < height and 0 <= neighbor_column < width):
                        continue
                    if mask[neighbor_row, neighbor_column] and not labels[neighbor_row, neighbor_column]:
                        labels[neighbor_row, neighbor_column] = next_label
                        stack.append((neighbor_row, neighbor_column))
    return labels, next_label


def _bin_centers(bins: np.ndarray, bounds: SpatialBounds, grain: int) -> np.ndarray:
    bins = np.asarray(bins, dtype=np.float64)
    result = np.full(bins.shape, np.nan, dtype=np.float32)
    valid = (bins[:, 0] >= 0) & (bins[:, 1] >= 0)
    if valid.any():
        result[valid, 0] = bounds.x_min + (bins[valid, 0] + 0.5) * (
            bounds.x_max - bounds.x_min
        ) / grain
        result[valid, 1] = bounds.y_min + (bins[valid, 1] + 0.5) * (
            bounds.y_max - bounds.y_min
        ) / grain
    return result


def calculate_place_field_details(
    pose: np.ndarray,
    dg_activity: np.ndarray,
    bounds: SpatialBounds = SpatialBounds(),
    grain: int = DEFAULT_GRAIN,
    threshold_fractions: tuple[float, ...] = FIELD_THRESHOLD_FRACTIONS,
    min_active_observations: int = FIELD_MIN_ACTIVE_OBSERVATIONS,
    min_active_bins: int = FIELD_MIN_ACTIVE_BINS,
    mono_mass_fraction: float = FIELD_MONO_MASS_FRACTION,
) -> dict[str, np.ndarray]:
    """Return evaluator-compatible maps and multilevel DG field diagnostics."""

    pose = np.asarray(pose, dtype=np.float32)
    activity = np.asarray(dg_activity, dtype=np.float32)
    if min_active_observations <= 0 or min_active_bins <= 0:
        raise SpatialContractError("place-field eligibility minima must be positive")
    fractions = np.asarray(threshold_fractions, dtype=np.float32)
    if fractions.ndim != 1 or fractions.size == 0 or not np.isfinite(fractions).all():
        raise SpatialContractError("field threshold fractions must be a finite nonempty sequence")
    if (fractions <= 0).any() or (fractions >= 1).any() or np.any(np.diff(fractions) <= 0):
        raise SpatialContractError("field threshold fractions must be increasing values in (0, 1)")
    if not 0 < mono_mass_fraction <= 1:
        raise SpatialContractError("mono-field mass fraction must be in (0, 1]")

    rate_maps, occupancy, in_bounds = spatial_rate_maps(pose, activity, bounds, grain)
    n_units = activity.shape[1]
    rate_sums = rate_maps.astype(np.float64) * occupancy[None, :, :]
    smooth_occupancy = _binomial_smooth(occupancy[None, :, :])[0]
    smooth_sums = _binomial_smooth(rate_sums)
    smoothed = np.divide(
        smooth_sums,
        smooth_occupancy[None, :, :],
        out=np.zeros_like(smooth_sums),
        where=smooth_occupancy[None, :, :] > 0,
    )
    smoothed[:, occupancy == 0] = 0

    active_counts = (activity[in_bounds] > 0).sum(axis=0).astype(np.int32)
    active_bin_maps, _, _ = spatial_rate_maps(pose, (activity > 0).astype(np.float32), bounds, grain)
    active_bins = (active_bin_maps > 0).sum(axis=(1, 2)).astype(np.int16)
    eligible = (active_counts >= min_active_observations) & (active_bins >= min_active_bins)
    active_fraction = (activity > 0).mean(axis=0).astype(np.float32) if pose.shape[0] else np.zeros(n_units, np.float32)
    information = np.asarray(
        [_spatial_information(rate_maps[unit], occupancy) for unit in range(n_units)], dtype=np.float32
    )

    labels = np.zeros((fractions.size, grain, grain, n_units), dtype=np.int16)
    component_count = np.zeros((fractions.size, n_units), dtype=np.int16)
    dominant_mass_fraction = np.zeros((fractions.size, n_units), dtype=np.float32)
    peak_bins = np.full((n_units, 2), -1, dtype=np.int16)
    secondary_bins = np.full((n_units, 2), -1, dtype=np.int16)

    for unit in range(n_units):
        unit_map = smoothed[unit]
        peak = float(unit_map.max())
        if not eligible[unit] or peak <= 0:
            continue
        threshold_components: list[list[tuple[float, int]]] = []
        for threshold_index, fraction in enumerate(fractions):
            unit_labels, count = _component_labels((unit_map >= float(fraction) * peak) & (occupancy > 0))
            labels[threshold_index, :, :, unit] = unit_labels
            component_count[threshold_index, unit] = count
            components: list[tuple[float, int]] = []
            for component in range(1, count + 1):
                mass = float(unit_map[unit_labels == component].sum())
                components.append((mass, component))
            components.sort(reverse=True)
            threshold_components.append(components)
            total_mass = sum(value for value, _ in components)
            if total_mass > 0:
                dominant_mass_fraction[threshold_index, unit] = components[0][0] / total_mass

        middle = int(np.argmin(np.abs(fractions - 0.5)))
        for destination, rank in ((peak_bins, 0), (secondary_bins, 1)):
            if len(threshold_components[middle]) <= rank:
                continue
            component = threshold_components[middle][rank][1]
            component_mask = labels[middle, :, :, unit] == component
            masked = np.where(component_mask, unit_map, -np.inf)
            peak_row, peak_column = np.unravel_index(int(np.argmax(masked)), masked.shape)
            destination[unit] = (peak_column, peak_row)

    mono_score = dominant_mass_fraction.min(axis=0)
    mono_field = eligible & (mono_score >= float(mono_mass_fraction))
    peak_xy = _bin_centers(peak_bins, bounds, grain)
    secondary_xy = _bin_centers(secondary_bins, bounds, grain)
    primary_secondary_distance = np.linalg.norm(peak_xy - secondary_xy, axis=1).astype(np.float32)
    primary_secondary_distance[~np.isfinite(secondary_xy).all(axis=1)] = np.nan
    nearest_neighbor = np.full(n_units, np.nan, dtype=np.float32)
    valid_peaks = eligible & np.isfinite(peak_xy).all(axis=1)
    valid_indices = np.flatnonzero(valid_peaks)
    if valid_indices.size >= 2:
        coordinates = peak_xy[valid_indices].astype(np.float64)
        distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        nearest_neighbor[valid_indices] = distances.min(axis=1).astype(np.float32)

    return {
        "occupancy": occupancy.astype(np.int32),
        "rate_maps": np.moveaxis(rate_maps, 0, -1).astype(np.float32),
        "smoothed_rate_maps": np.moveaxis(smoothed, 0, -1).astype(np.float32),
        "spatial_information": information,
        "active_fraction": active_fraction,
        "field_threshold_fractions": fractions,
        "field_component_labels": labels,
        "field_component_count": component_count,
        "field_dominant_mass_fraction": dominant_mass_fraction,
        "field_active_observation_count": active_counts,
        "field_active_bin_count": active_bins,
        "field_eligible": eligible,
        "field_mono_score": mono_score.astype(np.float32),
        "field_mono": mono_field,
        "field_dominant_peak_bin": peak_bins,
        "field_secondary_peak_bin": secondary_bins,
        "field_dominant_peak_xy": peak_xy,
        "field_secondary_peak_xy": secondary_xy,
        "field_primary_secondary_peak_distance": primary_secondary_distance,
        "field_dominant_peak_nearest_neighbor_distance": nearest_neighbor,
        "field_min_active_observations": np.asarray(min_active_observations, dtype=np.int32),
        "field_min_active_bins": np.asarray(min_active_bins, dtype=np.int16),
        "field_mono_mass_fraction": np.asarray(mono_mass_fraction, dtype=np.float32),
    }


def reliable_graph_adjacency(
    tctrl: np.ndarray,
    confidence: np.ndarray,
    attempts: np.ndarray,
    confidence_threshold: float = 0.5,
    reliability_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return directed reliable edges and their beta-posterior mean reliability."""

    tctrl = np.asarray(tctrl, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    attempts = np.asarray(attempts, dtype=np.float64)
    if tctrl.ndim != 2 or tctrl.shape[0] != tctrl.shape[1]:
        raise SpatialContractError("control graph arrays must be square matrices")
    if confidence.shape != tctrl.shape or attempts.shape != tctrl.shape:
        raise SpatialContractError("control graph arrays must have matching dimensions")
    if confidence_threshold < 0 or not 0 < reliability_threshold <= 1:
        raise SpatialContractError("invalid graph reliability thresholds")
    reliability = (confidence + 1.0) / (attempts + 2.0)
    adjacency = (
        (tctrl > 0)
        & (confidence >= float(confidence_threshold))
        & (reliability >= float(reliability_threshold))
    )
    adjacency = adjacency.astype(np.bool_, copy=True)
    np.fill_diagonal(adjacency, False)
    return adjacency, reliability.astype(np.float32)


def _shortest_path_hops(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=np.bool_)
    n_nodes = adjacency.shape[0]
    distances = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    distances[adjacency] = 1.0
    np.fill_diagonal(distances, 0.0)
    for intermediate in range(n_nodes):
        distances = np.minimum(
            distances,
            distances[:, intermediate, None] + distances[None, intermediate, :],
        )
    return distances


def _component_sizes(adjacency: np.ndarray) -> list[int]:
    adjacency = np.asarray(adjacency, dtype=np.bool_)
    remaining = set(range(adjacency.shape[0]))
    sizes: list[int] = []
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = set(np.flatnonzero(adjacency[node]).tolist()) & remaining
            remaining -= neighbors
            stack.extend(neighbors)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _undirected_clustering(adjacency: np.ndarray) -> float:
    adjacency = np.asarray(adjacency, dtype=np.bool_)
    values: list[float] = []
    for node in range(adjacency.shape[0]):
        neighbors = np.flatnonzero(adjacency[node])
        degree = neighbors.size
        if degree < 2:
            values.append(0.0)
            continue
        edges = int(adjacency[np.ix_(neighbors, neighbors)].sum() // 2)
        values.append(2.0 * edges / float(degree * (degree - 1)))
    return float(np.mean(values)) if values else 0.0


def _global_efficiency(adjacency: np.ndarray) -> tuple[float, np.ndarray]:
    distances = _shortest_path_hops(adjacency)
    n_nodes = adjacency.shape[0]
    mask = np.isfinite(distances) & ~np.eye(n_nodes, dtype=np.bool_)
    inverse = np.zeros_like(distances)
    inverse[mask] = 1.0 / distances[mask]
    denominator = max(1, n_nodes * (n_nodes - 1))
    return float(inverse.sum() / denominator), distances


def _density_matched_references(n_nodes: int, edge_count: int, samples: int = 64) -> tuple[np.ndarray, list[np.ndarray]]:
    possible = [(i, j) for distance in range(1, n_nodes) for i in range(n_nodes)
                for j in ((i + distance) % n_nodes,) if i < j]
    possible = list(dict.fromkeys(possible))
    possible.sort(key=lambda edge: (min((edge[1] - edge[0]) % n_nodes, (edge[0] - edge[1]) % n_nodes), edge))
    lattice = np.zeros((n_nodes, n_nodes), dtype=np.bool_)
    for source, target in possible[:edge_count]:
        lattice[source, target] = lattice[target, source] = True
    all_edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    rng = np.random.default_rng(7919 + n_nodes * 257 + edge_count)
    random_graphs: list[np.ndarray] = []
    for _ in range(samples):
        graph = np.zeros((n_nodes, n_nodes), dtype=np.bool_)
        if edge_count:
            selected = rng.choice(len(all_edges), size=edge_count, replace=False)
            for index in np.atleast_1d(selected):
                source, target = all_edges[int(index)]
                graph[source, target] = graph[target, source] = True
        random_graphs.append(graph)
    return lattice, random_graphs


def _small_world_propensity(undirected: np.ndarray) -> float:
    n_nodes = undirected.shape[0]
    edge_count = int(np.triu(undirected, 1).sum())
    if n_nodes < 3 or edge_count == 0:
        return 0.0
    lattice, random_graphs = _density_matched_references(n_nodes, edge_count)
    clustering = _undirected_clustering(undirected)
    lattice_clustering = _undirected_clustering(lattice)
    random_clustering = float(np.mean([_undirected_clustering(graph) for graph in random_graphs]))
    efficiency, _ = _global_efficiency(undirected)
    lattice_efficiency, _ = _global_efficiency(lattice)
    random_efficiency = float(np.mean([_global_efficiency(graph)[0] for graph in random_graphs]))
    length = 1.0 / efficiency if efficiency > 0 else np.inf
    lattice_length = 1.0 / lattice_efficiency if lattice_efficiency > 0 else np.inf
    random_length = 1.0 / random_efficiency if random_efficiency > 0 else np.inf
    clustering_denominator = lattice_clustering - random_clustering
    length_denominator = lattice_length - random_length
    delta_clustering = 0.0 if abs(clustering_denominator) < 1e-12 else (
        lattice_clustering - clustering
    ) / clustering_denominator
    delta_length = 0.0 if not np.isfinite(length) or abs(length_denominator) < 1e-12 else (
        length - random_length
    ) / length_denominator
    delta_clustering = float(np.clip(delta_clustering, 0.0, 1.0))
    delta_length = float(np.clip(delta_length, 0.0, 1.0))
    return float(np.clip(1.0 - np.sqrt((delta_clustering ** 2 + delta_length ** 2) / 2.0), 0.0, 1.0))


def calculate_graph_diagnostics(
    tctrl: np.ndarray,
    confidence: np.ndarray,
    attempts: np.ndarray,
    prospective_attempts: np.ndarray | None = None,
    prospective_successes: np.ndarray | None = None,
    field_eligible: np.ndarray | None = None,
    dominant_peak_xy: np.ndarray | None = None,
    confidence_threshold: float = 0.5,
    reliability_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Calculate deterministic structural, prospective, and spatial graph diagnostics."""

    adjacency, reliability = reliable_graph_adjacency(
        tctrl, confidence, attempts, confidence_threshold, reliability_threshold
    )
    n_nodes = adjacency.shape[0]
    efficiency, distances = _global_efficiency(adjacency)
    reachable = np.isfinite(distances) & ~np.eye(n_nodes, dtype=np.bool_)
    reachable_hops = distances[reachable]
    reach = np.isfinite(distances)
    strong_sizes = [int((reach[node] & reach[:, node]).sum()) for node in range(n_nodes)]
    undirected = adjacency | adjacency.T
    weak_sizes = _component_sizes(undirected)
    directed_edges = int(adjacency.sum())
    degrees = adjacency.sum(axis=0) + adjacency.sum(axis=1)
    edge_peak_distance = np.full((n_nodes, n_nodes), np.nan, dtype=np.float32)
    eligible = np.zeros(n_nodes, dtype=np.bool_) if field_eligible is None else np.asarray(field_eligible, dtype=np.bool_)
    peaks = np.full((n_nodes, 2), np.nan, dtype=np.float32) if dominant_peak_xy is None else np.asarray(
        dominant_peak_xy, dtype=np.float32
    )
    if eligible.shape != (n_nodes,) or peaks.shape != (n_nodes, 2):
        raise SpatialContractError("graph nodes must align exactly with eligible mono-field diagnostics")
    valid_nodes = eligible & np.isfinite(peaks).all(axis=1)
    endpoint_valid = valid_nodes[:, None] & valid_nodes[None, :]
    if endpoint_valid.any():
        edge_peak_distance[endpoint_valid] = np.linalg.norm(
            peaks[:, None, :] - peaks[None, :, :], axis=-1
        )[endpoint_valid]
    valid_reliable = adjacency & endpoint_valid

    prospective_attempts = np.zeros_like(np.asarray(attempts, dtype=np.float64)) if prospective_attempts is None else np.asarray(
        prospective_attempts, dtype=np.float64
    )
    prospective_successes = np.zeros_like(prospective_attempts) if prospective_successes is None else np.asarray(
        prospective_successes, dtype=np.float64
    )
    if prospective_attempts.shape != adjacency.shape or prospective_successes.shape != adjacency.shape:
        raise SpatialContractError("prospective graph accumulators must align with the graph")
    prospective_total = float(prospective_attempts.sum())
    prospective_success_total = float(prospective_successes.sum())
    prospective_rate = prospective_success_total / prospective_total if prospective_total > 0 else 0.0
    endpoint_fraction = float(valid_reliable.sum() / directed_edges) if directed_edges else 0.0
    edge_distances = edge_peak_distance[valid_reliable]
    tctrl_values = np.asarray(tctrl, dtype=np.float64)[valid_reliable]
    correlation = 0.0
    if edge_distances.size >= 2 and np.std(edge_distances) > 0 and np.std(tctrl_values) > 0:
        correlation = float(np.corrcoef(edge_distances, tctrl_values)[0, 1])
    hops_encoded = np.where(np.isfinite(distances), distances, -1).astype(np.int16)
    return {
        "graph_reliable_adjacency": adjacency,
        "graph_edge_reliability": reliability,
        "graph_shortest_path_hops": hops_encoded,
        "graph_reliable_edge_count": np.asarray(directed_edges, dtype=np.int32),
        "graph_reliable_edge_density": np.asarray(directed_edges / max(1, n_nodes * (n_nodes - 1)), dtype=np.float32),
        "graph_largest_weak_component_size": np.asarray(weak_sizes[0] if weak_sizes else 0, dtype=np.int16),
        "graph_largest_strong_component_size": np.asarray(max(strong_sizes, default=0), dtype=np.int16),
        "graph_reachable_pair_fraction": np.asarray(reachable.sum() / max(1, n_nodes * (n_nodes - 1)), dtype=np.float32),
        "graph_mean_reachable_shortest_path_hops": np.asarray(reachable_hops.mean() if reachable_hops.size else 0.0, dtype=np.float32),
        "graph_median_reachable_shortest_path_hops": np.asarray(np.median(reachable_hops) if reachable_hops.size else 0.0, dtype=np.float32),
        "graph_reliable_global_efficiency": np.asarray(efficiency, dtype=np.float32),
        "graph_undirected_clustering": np.asarray(_undirected_clustering(undirected), dtype=np.float32),
        "graph_directed_reciprocity": np.asarray((adjacency & adjacency.T).sum() / max(1, directed_edges), dtype=np.float32),
        "graph_small_world_propensity": np.asarray(_small_world_propensity(undirected), dtype=np.float32),
        "graph_max_total_degree_fraction": np.asarray(degrees.max(initial=0) / max(1, 2 * directed_edges), dtype=np.float32),
        "graph_degree_herfindahl": np.asarray(np.square(degrees / max(1, degrees.sum())).sum(), dtype=np.float32),
        "graph_spatial_endpoint_valid_fraction": np.asarray(endpoint_fraction, dtype=np.float32),
        "graph_reliable_edge_peak_distance": edge_peak_distance,
        "graph_reliable_edge_peak_distance_mean": np.asarray(edge_distances.mean() if edge_distances.size else 0.0, dtype=np.float32),
        "graph_tctrl_peak_distance_correlation": np.asarray(correlation, dtype=np.float32),
        "graph_tctrl_peak_distance_pair_count": np.asarray(edge_distances.size, dtype=np.int32),
        "graph_prospective_attempt_count": np.asarray(prospective_total, dtype=np.float32),
        "graph_prospective_success_count": np.asarray(prospective_success_total, dtype=np.float32),
        "graph_prospective_success_fraction": np.asarray(prospective_rate, dtype=np.float32),
        "graph_grounded_controllability": np.asarray(prospective_rate * endpoint_fraction, dtype=np.float32),
    }


def calculate_spatial_metrics(
    pose: np.ndarray,
    dg_activity: np.ndarray,
    dones: np.ndarray,
    segment_id: np.ndarray,
    bounds: SpatialBounds = SpatialBounds(),
    grain: int = DEFAULT_GRAIN,
    stationary_distance: float = 1.0,
) -> dict[str, float]:
    """Calculate monitoring metrics from exact behavior-time pose and DG activity."""

    if stationary_distance < 0 or not np.isfinite(stationary_distance):
        raise SpatialContractError("stationary_distance must be finite and non-negative")
    pose = np.asarray(pose, dtype=np.float32)
    activity = np.asarray(dg_activity, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.bool_).reshape(-1)
    segment_id = np.asarray(segment_id, dtype=np.int64).reshape(-1)
    if pose.ndim != 2 or activity.ndim != 2 or pose.shape[0] != activity.shape[0]:
        raise SpatialContractError("pose and DG activity arrays are not aligned")
    if dones.shape[0] != pose.shape[0] or segment_id.shape[0] != pose.shape[0]:
        raise SpatialContractError("trajectory arrays are not aligned")
    if not np.isfinite(pose).all() or not np.isfinite(activity).all():
        raise SpatialContractError("pose and DG activity must be finite")

    details = calculate_place_field_details(pose, activity, bounds, grain)
    rate_maps = np.moveaxis(details["rate_maps"], -1, 0)
    occupancy = details["occupancy"]
    _, _, in_bounds = spatial_rate_maps(pose, activity, bounds, grain)
    active = (activity > 0).any(axis=0)
    n_units = int(activity.shape[1])
    n_active = int(active.sum())
    active_maps = rate_maps[active]
    information = details["spatial_information"][active]

    occupied = occupancy.reshape(-1) > 0
    map_cosine = 0.0
    if n_active >= 2 and occupied.any():
        vectors = active_maps.reshape(n_active, -1)[:, occupied].astype(np.float64)
        norms = np.linalg.norm(vectors, axis=1)
        usable = norms > 0
        vectors = vectors[usable]
        norms = norms[usable]
        if vectors.shape[0] >= 2:
            similarity = (vectors @ vectors.T) / np.outer(norms, norms)
            upper = similarity[np.triu_indices(similarity.shape[0], k=1)]
            map_cosine = float(upper.mean()) if upper.size else 0.0

    unique_peaks = 0
    if n_active:
        flat_active_maps = active_maps.reshape(n_active, -1)
        positive_maps = flat_active_maps.max(axis=1) > 0
        if positive_maps.any():
            unique_peaks = int(np.unique(flat_active_maps[positive_maps].argmax(axis=1)).size)

    transition = (segment_id[1:] == segment_id[:-1]) & ~dones[:-1]
    delta_xy = pose[1:, :2] - pose[:-1, :2]
    step_distance = np.linalg.norm(delta_xy, axis=1)[transition]
    yaw_delta = ((pose[1:, 2] - pose[:-1, 2] + 180.0) % 360.0) - 180.0
    yaw_delta = np.abs(yaw_delta[transition])

    total_path = float(step_distance.sum())
    total_displacement = 0.0
    for segment in np.unique(segment_id):
        indices = np.flatnonzero(segment_id == segment)
        if indices.size >= 2:
            total_displacement += float(np.linalg.norm(pose[indices[-1], :2] - pose[indices[0], :2]))

    n = int(pose.shape[0])
    return {
        "valid_sample_count": float(n),
        "in_bounds_fraction": float(in_bounds.mean()) if n else 0.0,
        "visited_cell_fraction": float((occupancy > 0).sum() / occupancy.size),
        "active_unit_fraction": float(n_active / n_units) if n_units else 0.0,
        "silent_unit_fraction": float(1.0 - n_active / n_units) if n_units else 0.0,
        "active_unit_mean_spatial_information": float(np.mean(information)) if information.size else 0.0,
        "active_only_map_cosine": map_cosine,
        "unique_active_peak_bins": float(unique_peaks),
        "mean_physical_step_distance": float(step_distance.mean()) if step_distance.size else 0.0,
        "stationary_step_fraction": float((step_distance <= stationary_distance).mean()) if step_distance.size else 0.0,
        "path_efficiency": float(total_displacement / total_path) if total_path > 0 else 0.0,
        "mean_absolute_circular_yaw_change": float(yaw_delta.mean()) if yaw_delta.size else 0.0,
        "mono_field_unit_fraction": float(details["field_mono"].sum() / details["field_eligible"].sum())
        if details["field_eligible"].any() else 0.0,
        "mean_primary_secondary_peak_distance": float(np.nanmean(details["field_primary_secondary_peak_distance"]))
        if np.isfinite(details["field_primary_secondary_peak_distance"]).any() else 0.0,
        "median_dominant_peak_nearest_neighbor_distance": float(
            np.nanmedian(details["field_dominant_peak_nearest_neighbor_distance"])
        ) if np.isfinite(details["field_dominant_peak_nearest_neighbor_distance"]).any() else 0.0,
    }


class OnlineSpatialWindow:
    """A bounded latest-N buffer that preserves rollout and episode segments."""

    def __init__(self, limit: int):
        if limit <= 0:
            raise SpatialContractError("window limit must be positive")
        self.limit = int(limit)
        self._next_segment = 0
        self._size = 0
        self._write_index = 0
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self._size

    def append_rollouts(
        self,
        pose: np.ndarray,
        dg_activity: np.ndarray,
        actions: np.ndarray,
        dones: np.ndarray,
        policy_version: np.ndarray,
        valids: np.ndarray,
        max_segment_jump_distance: float | None = None,
    ) -> int:
        pose = np.asarray(pose, dtype=np.float32)
        activity = np.asarray(dg_activity, dtype=np.float32)
        actions = np.asarray(actions)
        dones = np.asarray(dones, dtype=np.bool_)
        versions = np.asarray(policy_version)
        valids = np.asarray(valids, dtype=np.bool_)
        if max_segment_jump_distance is not None and (
            not np.isfinite(max_segment_jump_distance) or max_segment_jump_distance <= 0
        ):
            raise SpatialContractError("maximum segment jump distance must be finite and positive")
        if pose.ndim != 3 or pose.shape[-1] != 3 or activity.ndim != 3:
            raise SpatialContractError("rollout pose/activity must have shapes [B,T,3] and [B,T,units]")
        leading = pose.shape[:2]
        for name, value in (
            ("dg_activity", activity),
            ("actions", actions),
            ("dones", dones),
            ("policy_version", versions),
            ("valids", valids),
        ):
            if value.shape[:2] != leading:
                raise SpatialContractError(f"{name} rollout shape {value.shape[:2]} != {leading}")

        selected: dict[str, list[np.ndarray | Any]] = {
            "pose": [], "dg_activity": [], "actions": [], "dones": [],
            "segment_id": [], "policy_version": [],
        }
        for row in range(leading[0]):
            segment = self._next_segment
            self._next_segment += 1
            previous_valid = False
            previous_pose: np.ndarray | None = None
            for step in range(leading[1]):
                if not valids[row, step] or not np.isfinite(pose[row, step]).all():
                    previous_valid = False
                    previous_pose = None
                    continue
                discontinuity = (
                    previous_valid
                    and max_segment_jump_distance is not None
                    and float(np.linalg.norm(pose[row, step, :2] - previous_pose[:2]))
                    > max_segment_jump_distance
                )
                if (not previous_valid and step > 0) or discontinuity:
                    # Gaps, terminals, and implausible unmarked relocations
                    # never inherit the preceding trajectory segment.
                    segment = self._next_segment
                    self._next_segment += 1
                selected["pose"].append(pose[row, step])
                selected["dg_activity"].append(activity[row, step])
                selected["actions"].append(actions[row, step])
                selected["dones"].append(bool(dones[row, step]))
                selected["segment_id"].append(segment)
                selected["policy_version"].append(int(round(float(versions[row, step]))))
                previous_valid = True
                previous_pose = pose[row, step]
                if dones[row, step]:
                    segment = self._next_segment
                    self._next_segment += 1
                    previous_valid = False
                    previous_pose = None

        if not selected["pose"]:
            return 0
        incoming = {
            "pose": np.asarray(selected["pose"], dtype=np.float32),
            "dg_activity": np.asarray(selected["dg_activity"], dtype=np.float32),
            "actions": np.asarray(selected["actions"]),
            "dones": np.asarray(selected["dones"], dtype=np.bool_),
            "segment_id": np.asarray(selected["segment_id"], dtype=np.int64),
            "policy_version": np.asarray(selected["policy_version"], dtype=np.int64),
        }
        incoming_size = int(incoming["pose"].shape[0])
        if not self._arrays:
            self._arrays = {
                key: np.empty((self.limit, *value.shape[1:]), dtype=value.dtype)
                for key, value in incoming.items()
            }
        else:
            for key, value in incoming.items():
                expected = self._arrays[key].shape[1:]
                if value.shape[1:] != expected:
                    raise SpatialContractError(
                        f"{key} trailing shape {value.shape[1:]} != buffered shape {expected}"
                    )

        if incoming_size >= self.limit:
            for key, value in incoming.items():
                self._arrays[key][...] = value[-self.limit :]
            self._size = self.limit
            self._write_index = 0
            return len(selected["pose"])

        first = min(incoming_size, self.limit - self._write_index)
        second = incoming_size - first
        for key, value in incoming.items():
            self._arrays[key][self._write_index : self._write_index + first] = value[:first]
            if second:
                self._arrays[key][:second] = value[first:]
        self._write_index = (self._write_index + incoming_size) % self.limit
        self._size = min(self.limit, self._size + incoming_size)
        return len(selected["pose"])

    def arrays(self, max_samples: int | None = None) -> dict[str, np.ndarray]:
        """Return the latest samples in chronological order."""

        if max_samples is not None and max_samples <= 0:
            raise SpatialContractError("maximum returned samples must be positive")
        if not self._arrays:
            return {}
        size = self._size if max_samples is None else min(self._size, int(max_samples))
        start = (self._write_index - size) % self.limit
        if start + size <= self.limit:
            return {key: value[start : start + size].copy() for key, value in self._arrays.items()}
        first = self.limit - start
        return {
            key: np.concatenate((value[start:], value[: size - first]), axis=0)
            for key, value in self._arrays.items()
        }


def validate_snapshot_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema = str(np.asarray(payload.get("schema", "")).item())
    if schema != SNAPSHOT_SCHEMA:
        raise SpatialContractError(f"schema must be {SNAPSHOT_SCHEMA!r}, got {schema!r}")
    missing = [key for key in SNAPSHOT_REQUIRED_ARRAYS if key not in payload]
    if missing:
        raise SpatialContractError(f"snapshot is missing arrays: {missing}")
    arrays = _aligned_arrays(*(payload[key] for key in SNAPSHOT_REQUIRED_ARRAYS))
    result = dict(payload)
    for key, value in zip(SNAPSHOT_REQUIRED_ARRAYS, arrays):
        result[key] = value
    required_metadata = (
        "schema_version", "target_env_steps", "actual_env_steps", "window_limit", "window_start_env_steps",
        "window_end_env_steps", "policy_id", "run_name", "experiment_identity", "environment", "frameskip",
        "grain", "bounds",
    )
    absent = [key for key in required_metadata if key not in payload]
    if absent:
        raise SpatialContractError(f"snapshot is missing metadata: {absent}")
    bounds_array = np.asarray(payload["bounds"], dtype=np.float32).reshape(-1)
    if bounds_array.shape != (4,):
        raise SpatialContractError("bounds must contain x_min, x_max, y_min, y_max")
    SpatialBounds(*[float(item) for item in bounds_array])
    if int(np.asarray(payload["grain"]).item()) <= 0:
        raise SpatialContractError("grain must be positive")
    if int(np.asarray(payload["schema_version"]).item()) != 1:
        raise SpatialContractError("schema_version must be 1")
    if int(np.asarray(payload["policy_id"]).item()) < 0:
        raise SpatialContractError("policy_id must be non-negative")
    if int(np.asarray(payload["frameskip"]).item()) <= 0:
        raise SpatialContractError("frameskip must be positive")
    if int(np.asarray(payload["window_limit"]).item()) != result["pose"].shape[0]:
        raise SpatialContractError("snapshot sample count must equal window_limit")
    if "scalar_window_limit" in payload:
        scalar_window_limit = int(np.asarray(payload["scalar_window_limit"]).item())
        if scalar_window_limit <= 0 or scalar_window_limit > result["pose"].shape[0]:
            raise SpatialContractError("scalar_window_limit must be within the snapshot window")
    if int(np.asarray(payload["target_env_steps"]).item()) <= 0:
        raise SpatialContractError("target_env_steps must be positive")
    if int(np.asarray(payload["actual_env_steps"]).item()) < int(np.asarray(payload["target_env_steps"]).item()):
        raise SpatialContractError("actual_env_steps cannot precede target_env_steps")
    window_start = int(np.asarray(payload["window_start_env_steps"]).item())
    window_end = int(np.asarray(payload["window_end_env_steps"]).item())
    actual = int(np.asarray(payload["actual_env_steps"]).item())
    if window_start < 0 or window_start > window_end or window_end != actual:
        raise SpatialContractError("window frame limits must satisfy 0 <= start <= end == actual")
    for key in ("run_name", "experiment_identity", "environment"):
        if not str(np.asarray(payload[key]).item()).strip():
            raise SpatialContractError(f"{key} must be nonempty")

    detail_presence = [key in payload for key in SPATIAL_DETAIL_ARRAYS]
    if any(detail_presence) and not all(detail_presence):
        missing_details = [key for key in SPATIAL_DETAIL_ARRAYS if key not in payload]
        raise SpatialContractError(f"cached spatial details are incomplete: {missing_details}")
    if all(detail_presence):
        grain = int(np.asarray(payload["grain"]).item())
        units = result["dg_activity"].shape[1]
        thresholds = np.asarray(payload["field_threshold_fractions"]).size
        expected_shapes = {
            "occupancy": (grain, grain),
            "rate_maps": (grain, grain, units),
            "smoothed_rate_maps": (grain, grain, units),
            "spatial_information": (units,),
            "active_fraction": (units,),
            "field_component_labels": (thresholds, grain, grain, units),
            "field_component_count": (thresholds, units),
            "field_dominant_mass_fraction": (thresholds, units),
            "field_active_observation_count": (units,),
            "field_active_bin_count": (units,),
            "field_eligible": (units,),
            "field_mono_score": (units,),
            "field_mono": (units,),
            "field_dominant_peak_bin": (units, 2),
            "field_secondary_peak_bin": (units, 2),
            "field_dominant_peak_xy": (units, 2),
            "field_secondary_peak_xy": (units, 2),
            "field_primary_secondary_peak_distance": (units,),
            "field_dominant_peak_nearest_neighbor_distance": (units,),
        }
        for key, expected_shape in expected_shapes.items():
            if np.asarray(payload[key]).shape != expected_shape:
                raise SpatialContractError(
                    f"cached {key} has shape {np.asarray(payload[key]).shape}; expected {expected_shape}"
                )

    graph_presence = [key in payload for key in CONTROL_GRAPH_ARRAYS]
    if any(graph_presence) and not all(graph_presence):
        missing_graph = [key for key in CONTROL_GRAPH_ARRAYS if key not in payload]
        raise SpatialContractError(f"control graph snapshot is incomplete: {missing_graph}")
    if all(graph_presence):
        nodes = result["dg_activity"].shape[1]
        matrix_shape = (nodes, nodes)
        if np.asarray(payload["control_node_visits"]).shape != (nodes,):
            raise SpatialContractError("control graph node count does not match DG units")
        for key in CONTROL_GRAPH_ARRAYS[1:]:
            if np.asarray(payload[key]).shape != matrix_shape:
                raise SpatialContractError(f"{key} must have shape {matrix_shape}")
        if "control_representation_generation" not in payload:
            raise SpatialContractError("control graph generation is required")
        for key in (
            "control_passive_confidence", "control_passive_time", "control_passive_path_length",
            "control_passive_dx", "control_passive_dy", "control_passive_dtheta_sin",
            "control_passive_dtheta_cos",
        ):
            if key not in payload or np.asarray(payload[key]).shape != matrix_shape:
                raise SpatialContractError(f"complete control graph snapshot requires {key} with shape {matrix_shape}")
        for key in ("control_frontier_attempts", "control_frontier_discoveries", "control_pose_valid"):
            if key not in payload or np.asarray(payload[key]).shape != (nodes,):
                raise SpatialContractError(f"complete control graph snapshot requires {key} with {nodes} values")
        if "control_landmark_pose" not in payload or np.asarray(payload["control_landmark_pose"]).shape != (nodes, 3):
            raise SpatialContractError("control_landmark_pose must have shape [DG units, 3]")
        if "passive_recruitment_generation" in payload and (
            int(np.asarray(payload["passive_recruitment_generation"]).item())
            != int(np.asarray(payload["control_representation_generation"]).item())
        ):
            raise SpatialContractError("control and passive graph generations do not match")
    nodes = result["dg_activity"].shape[1]
    passive_keys = (
        "passive_recruitment_confidence", "passive_recruitment_elapsed",
        "passive_recruitment_birth_support", "passive_recruitment_generation",
    )
    passive_presence = [key in payload for key in passive_keys]
    if any(passive_presence) and not all(passive_presence):
        raise SpatialContractError("passive recruitment graph snapshot is incomplete")
    if all(passive_presence):
        if np.asarray(payload["passive_recruitment_confidence"]).shape != (nodes, nodes):
            raise SpatialContractError("passive recruitment confidence has the wrong graph dimensions")
        if np.asarray(payload["passive_recruitment_elapsed"]).shape != (nodes, nodes):
            raise SpatialContractError("passive recruitment elapsed has the wrong graph dimensions")
        if np.asarray(payload["passive_recruitment_birth_support"]).shape != (nodes,):
            raise SpatialContractError("passive recruitment birth support must align with DG units")
    if "dg_recruitment_row_counts" in payload and np.asarray(payload["dg_recruitment_row_counts"]).shape != (nodes,):
        raise SpatialContractError("DG recruitment row counts must align with DG units")
    return result


def load_spatial_snapshot(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return validate_snapshot_payload({key: data[key] for key in data.files})


def write_spatial_snapshot_atomic(output_dir: Path, payload: Mapping[str, Any]) -> tuple[Path, bool]:
    """Write a compressed target/actual snapshot atomically without replacement."""

    validated = validate_snapshot_payload(payload)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = int(np.asarray(validated["target_env_steps"]).item())
    actual = int(np.asarray(validated["actual_env_steps"]).item())
    run_name = str(np.asarray(validated["run_name"]).item())
    policy_id = int(np.asarray(validated["policy_id"]).item())
    existing = sorted(output_dir.glob(f"snapshot_target_{target:012d}_actual_*.npz"))
    if existing:
        for path in existing:
            prior = load_spatial_snapshot(path)
            if str(np.asarray(prior["run_name"]).item()) != run_name or int(np.asarray(prior["policy_id"]).item()) != policy_id:
                raise SpatialContractError(f"existing target has conflicting identity: {path}")
        return existing[0], False

    destination = output_dir / f"snapshot_target_{target:012d}_actual_{actual:012d}.npz"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".online-spatial-", suffix=".npz", dir=output_dir, delete=False) as handle:
            temp_path = Path(handle.name)
            np.savez_compressed(handle, **validated)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, destination)
        except FileExistsError:
            prior = load_spatial_snapshot(destination)
            if str(np.asarray(prior["run_name"]).item()) != run_name or int(np.asarray(prior["policy_id"]).item()) != policy_id:
                raise SpatialContractError(f"concurrent snapshot has conflicting identity: {destination}")
            return destination, False
        return destination, True
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
