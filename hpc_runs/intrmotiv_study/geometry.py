"""Verified map identity and privileged, traversability-aware diagnostics.

Entity maps use top-to-bottom rows; telemetry arrays use increasing world y.
These helpers never supply model inputs or training rewards.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

GEOMETRY_SCHEMA_V1 = "intrmotiv/map-geometry/v1"
GEOMETRY_SCHEMA_V2 = "intrmotiv/map-geometry/v2"
GEOMETRY_SCHEMA = GEOMETRY_SCHEMA_V1

_DIRECTIONS = (
    ("north", -1, 0),
    ("east", 0, 1),
    ("south", 1, 0),
    ("west", 0, -1),
)
_DECALS = tuple(f"decal/lab_games/dec_img_style01_{index:03d}" for index in range(1, 11))
_COLORS = (
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4",
    "#42D4F4", "#F032E6", "#BFEF45", "#FABED4", "#469990",
)


def entity_record(entity: str, *, map_seed: int, wall_removal_probability: float) -> dict:
    rows = entity.splitlines()
    height = len(rows)
    width = len(rows[0]) if rows else 0
    if height < 5 or width < 5 or height % 2 == 0 or width % 2 == 0:
        raise ValueError("Expected odd entity dimensions of at least 5x5")
    if any(len(row) != width for row in rows):
        raise ValueError("Entity map must be rectangular")
    grid = np.array([list(row) for row in rows])
    if not np.isin(grid, ["*", " ", "P"]).all():
        raise ValueError("Only wall, floor and spawn entities are allowed")
    if not all((edge == "*").all() for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])):
        raise ValueError("Map boundary must remain closed")
    floor = grid != "*"
    cells = [tuple(cell) for cell in np.argwhere(floor)]
    adjacency = {
        cell: [
            p
            for p in ((cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1))
            if floor[p]
        ]
        for cell in cells
    }
    distances = []
    for start in cells:
        seen = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen[nxt] = seen[node] + 1
                    queue.append(nxt)
        if len(seen) != len(cells):
            raise ValueError("Disconnected accessible floor")
        distances.extend(distance for target, distance in seen.items() if target > start)
    degree = np.array([len(adjacency[cell]) for cell in cells])
    return dict(
        schema=GEOMETRY_SCHEMA,
        map_seed=int(map_seed),
        wall_removal_probability=float(wall_removal_probability),
        entity_layer=entity,
        sha256=sha256(entity.encode()).hexdigest(),
        accessible_mask=np.flipud(floor[1:-1, 1:-1]).astype(int).tolist(),
        spawn_cells_rc=np.argwhere(grid == "P").tolist(),
        bounds=[100.0, float((width - 1) * 100), 100.0, float((height - 1) * 100)],
        cell_size=100.0,
        coordinate_contract="mask[y_bin,x_bin]; x=floor(x_world/100)-1; y=floor(y_world/100)-1",
        accessible_cells=len(cells),
        corridor_fraction=float(np.mean(degree == 2)),
        dead_ends=int(np.sum(degree == 1)),
        junctions=int(np.sum(degree >= 3)),
        shortest_path_histogram=np.bincount(distances).tolist(),
        shortest_path_quantiles=np.quantile(distances, [0, 0.25, 0.5, 0.75, 1]).tolist(),
    )


def _portable_shuffle(values, seed):
    """Fisher-Yates shuffle shared with the Lua level (32-bit LCG)."""
    result = list(values)
    state = int(seed) & 0xFFFFFFFF
    for index in range(len(result) - 1, 0, -1):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        selected = state % (index + 1)
        result[index], result[selected] = result[selected], result[index]
    return result


def cue_sites(entity: str, *, cue_layout_seed: int) -> list[dict]:
    """Choose 20 fixed, non-overlapping visible wall sites.

    Orientations name the direction from the adjacent floor cell toward the
    wall. Distinct floor and wall cells avoid ambiguous co-located cues.
    """
    rows = entity.splitlines()
    grid = np.array([list(row) for row in rows])
    candidates = []
    for wall_row in range(len(rows)):
        for wall_col in range(len(rows[0])):
            if grid[wall_row, wall_col] != "*":
                continue
            for direction, floor_dr, floor_dc in _DIRECTIONS:
                floor_row, floor_col = wall_row - floor_dr, wall_col - floor_dc
                if (
                    0 <= floor_row < len(rows)
                    and 0 <= floor_col < len(rows[0])
                    and grid[floor_row, floor_col] != "*"
                ):
                    candidates.append((wall_row, wall_col, floor_row, floor_col, direction))
    chosen = []
    used_walls, used_floors = set(), set()
    for candidate in _portable_shuffle(candidates, cue_layout_seed):
        wall = candidate[:2]
        floor = candidate[2:4]
        if wall in used_walls or floor in used_floors:
            continue
        chosen.append(candidate)
        used_walls.add(wall)
        used_floors.add(floor)
        if len(chosen) == 20:
            break
    if len(chosen) != 20:
        raise ValueError("Map does not contain 20 distinct visible cue sites")

    sites = []
    for index, (wall_row, wall_col, floor_row, floor_col, direction) in enumerate(chosen):
        cue_type = "decal" if index < 10 else "color"
        type_index = index if index < 10 else index - 10
        sites.append({
            "cue_id": f"{'D' if cue_type == 'decal' else 'C'}{type_index + 1:02d}",
            "cue_type": cue_type,
            "asset": _DECALS[type_index] if cue_type == "decal" else _COLORS[type_index],
            "wall_rc": [wall_row, wall_col],
            "floor_rc": [floor_row, floor_col],
            "floor_yx": [len(rows) - 2 - floor_row, floor_col - 1],
            "orientation": direction,
        })
    return sites


def landmark_entity_record(
    entity: str,
    *,
    map_seed: int,
    wall_removal_probability: float,
    cue_layout_seed: int,
    cue_mode: str,
) -> dict:
    """Create a verified v2 record for the landmark-rich/control maze."""
    if cue_mode not in {"rich", "none"}:
        raise ValueError("cue_mode must be 'rich' or 'none'")
    base = entity_record(
        entity,
        map_seed=map_seed,
        wall_removal_probability=wall_removal_probability,
    )
    sites = cue_sites(entity, cue_layout_seed=cue_layout_seed)
    identity = {
        "entity_sha256": base["sha256"],
        "cue_layout_seed": int(cue_layout_seed),
        "cue_mode": cue_mode,
        "cue_sites": sites,
    }
    return {
        **base,
        "schema": GEOMETRY_SCHEMA_V2,
        "entity_shape": [len(entity.splitlines()), len(entity.splitlines()[0])],
        "cue_layout_seed": int(cue_layout_seed),
        "cue_mode": cue_mode,
        "cue_sites": sites,
        "cue_layout_sha256": sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "neutral_wall_rgb": [112, 112, 112],
    }


@lru_cache(maxsize=16)
def load_geometry(manifest: str, seed: int, opening: float) -> dict:
    records = json.loads(Path(manifest).read_text())["maps"]
    matches = [r for r in records if r["map_seed"] == seed and r["wall_removal_probability"] == opening]
    if len(matches) != 1:
        raise ValueError("Geometry manifest must contain exactly one matching map")
    record = matches[0]
    verified = entity_record(record["entity_layer"], map_seed=seed, wall_removal_probability=opening)
    if verified != record:
        raise ValueError("Geometry manifest differs from its verified entity map")
    return record


@lru_cache(maxsize=8)
def load_landmark_geometry(
    manifest: str,
    seed: int,
    opening: float,
    rows: int,
    cols: int,
    cue_layout_seed: int,
    cue_mode: str,
) -> dict:
    records = json.loads(Path(manifest).read_text())["maps"]
    matches = [
        record for record in records
        if record["map_seed"] == seed
        and record["wall_removal_probability"] == opening
        and record.get("entity_shape") == [rows, cols]
        and record.get("cue_layout_seed") == cue_layout_seed
        and record.get("cue_mode") == cue_mode
    ]
    if len(matches) != 1:
        raise ValueError("Landmark geometry manifest must contain exactly one matching map")
    record = matches[0]
    verified = landmark_entity_record(
        record["entity_layer"], map_seed=seed,
        wall_removal_probability=opening,
        cue_layout_seed=cue_layout_seed, cue_mode=cue_mode
    )
    if verified != record:
        raise ValueError("Landmark geometry manifest differs from its verified source")
    return record


def geometry_from_config(cfg):
    environment = getattr(cfg, "env", None)
    if environment not in {"corridor_geometry_noreward", "easy_landmark_maze_noreward"}:
        return None
    if environment == "corridor_geometry_noreward":
        manifest = getattr(cfg, "dmlab_geometry_manifest", "") or str(
            Path(__file__).resolve().parents[1] / "studies/assets/corridor_geometry/maps.json"
        )
        return load_geometry(manifest, int(cfg.dmlab_map_seed), float(cfg.dmlab_wall_removal_probability))
    manifest = getattr(cfg, "dmlab_geometry_manifest", "") or str(
        Path(__file__).resolve().parents[1] / "studies/assets/easy_landmark_maze/maps.json"
    )
    return load_landmark_geometry(
        manifest,
        int(cfg.dmlab_map_seed),
        float(cfg.dmlab_wall_removal_probability),
        int(cfg.dmlab_map_rows),
        int(cfg.dmlab_map_cols),
        int(cfg.dmlab_cue_layout_seed),
        str(cfg.dmlab_landmark_cues),
    )


def verify_entity(entity, record):
    if isinstance(entity, bytes):
        entity = entity.decode()
    if not isinstance(entity, str) or sha256(entity.encode()).hexdigest() != record["sha256"]:
        raise ValueError("Loaded DMLab geometry does not match the archived map SHA-256")


def verify_cue_manifest(manifest, record):
    """Verify the native Lua cue manifest without exposing it to the policy."""
    if isinstance(manifest, bytes):
        manifest = manifest.decode()
    if not isinstance(manifest, str):
        raise ValueError("DMLab cue manifest must be a string")
    lines = manifest.splitlines()
    shape = record["entity_shape"]
    if len(lines) != 25 or lines[:5] != [
        "schema\teasy-landmark-maze/v2",
        f"shape\t{shape[0]}\t{shape[1]}",
        f"mode\t{record['cue_mode']}",
        f"seed\t{record['cue_layout_seed']}",
        "cue_id\ttype\tasset\twall_row\twall_col\tfloor_row\tfloor_col\torientation",
    ]:
        raise ValueError("DMLab cue manifest header differs from the archived contract")
    observed = []
    entity_height = len(record["entity_layer"].splitlines())
    for line in lines[5:]:
        fields = line.split("\t")
        if len(fields) != 8:
            raise ValueError("Malformed DMLab cue manifest row")
        observed.append({
            "cue_id": fields[0],
            "cue_type": fields[1],
            "asset": fields[2],
            "wall_rc": [int(fields[3]), int(fields[4])],
            "floor_rc": [int(fields[5]), int(fields[6])],
            "floor_yx": [entity_height - 2 - int(fields[5]), int(fields[6]) - 1],
            "orientation": fields[7],
        })
    if observed != record["cue_sites"]:
        raise ValueError("Loaded DMLab cue layout differs from the archived contract")


def geometry_payload(record):
    if record is None:
        return {}
    payload = {
        "geometry_schema": np.asarray(record.get("schema", GEOMETRY_SCHEMA_V1)),
        "geometry_sha256": np.asarray(record["sha256"]),
        "geometry_entity_layer": np.asarray(record["entity_layer"]),
        "geometry_map_seed": np.asarray(record["map_seed"]),
        "geometry_wall_removal_probability": np.asarray(record["wall_removal_probability"]),
        "geometry_accessible_mask": np.asarray(record["accessible_mask"], dtype=bool),
        "geometry_bounds": np.asarray(record["bounds"]),
        "geometry_cell_size": np.asarray(record["cell_size"]),
        "geometry_coordinate_contract": np.asarray(record["coordinate_contract"]),
    }
    if record.get("schema") == GEOMETRY_SCHEMA_V2:
        sites = record["cue_sites"]
        payload.update({
            "geometry_entity_shape": np.asarray(record["entity_shape"], dtype=np.int16),
            "geometry_cue_layout_seed": np.asarray(record["cue_layout_seed"]),
            "geometry_cue_mode": np.asarray(record["cue_mode"]),
            "geometry_cue_layout_sha256": np.asarray(record["cue_layout_sha256"]),
            "geometry_cue_ids": np.asarray([site["cue_id"] for site in sites]),
            "geometry_cue_types": np.asarray([site["cue_type"] for site in sites]),
            "geometry_cue_assets": np.asarray([site["asset"] for site in sites]),
            "geometry_cue_wall_rc": np.asarray([site["wall_rc"] for site in sites], dtype=np.int16),
            "geometry_cue_floor_yx": np.asarray([site["floor_yx"] for site in sites], dtype=np.int16),
            "geometry_cue_orientations": np.asarray([site["orientation"] for site in sites]),
            "geometry_cue_rendered": np.full(len(sites), record["cue_mode"] == "rich", dtype=bool),
        })
    return payload


def geometry_record_from_payload(payload):
    """Reconstruct and validate a geometry record embedded in an NPZ payload."""
    if "geometry_schema" not in payload:
        return None
    validate_geometry_payload(payload)
    schema = str(np.asarray(payload["geometry_schema"]).item())
    entity = str(np.asarray(payload["geometry_entity_layer"]).item())
    map_seed = int(np.asarray(payload["geometry_map_seed"]).item())
    if schema == GEOMETRY_SCHEMA_V1:
        return entity_record(
            entity,
            map_seed=map_seed,
            wall_removal_probability=float(
                np.asarray(payload["geometry_wall_removal_probability"]).item()
            ),
        )
    return landmark_entity_record(
        entity,
        map_seed=map_seed,
        wall_removal_probability=float(
            np.asarray(payload["geometry_wall_removal_probability"]).item()
        ),
        cue_layout_seed=int(np.asarray(payload["geometry_cue_layout_seed"]).item()),
        cue_mode=str(np.asarray(payload["geometry_cue_mode"]).item()),
    )


def accessible_cell(position, record):
    xy = np.asarray(position, dtype=float)[:2]
    if not np.isfinite(xy).all():
        return None
    x, y = np.floor(xy / record["cell_size"]).astype(int) - 1
    mask = record["accessible_mask"]
    if not (0 <= y < len(mask) and 0 <= x < len(mask[0])) or not mask[y][x]:
        return None
    return int(y), int(x)


class AccessibleCoverage:
    """Decision-weighted episode coverage; missing terminal pose holds coverage."""

    def __init__(self, record):
        self.record = record
        self.visited = set()
        self.steps = 0
        self.area_sum = 0
        self.invalid_pose_steps = 0

    def step(self, position):
        cell = accessible_cell(position, self.record) if position is not None else None
        if cell is None:
            self.invalid_pose_steps += 1
        else:
            self.visited.add(cell)
        self.steps += 1
        self.area_sum += len(self.visited)

    def metrics(self):
        area = self.record["accessible_cells"]
        return dict(
            accessible_coverage_fraction=len(self.visited) / area,
            accessible_coverage_auc=self.area_sum / max(1, self.steps) / area,
            accessible_unique_cells=float(len(self.visited)),
            geometry_invalid_pose_steps=float(self.invalid_pose_steps),
        )


def traversable_field_components(rate_maps, occupancy, mask):
    """Unsmooth, four-neighbor components at half of each observed unit peak.

    Separate names preserve the historical smoothed-field metrics. Walls and
    unvisited floor never bridge components, including diagonal contacts.
    """
    rate_maps = np.asarray(rate_maps)
    mask = np.asarray(mask, bool)
    if rate_maps.shape[-2:] != mask.shape or np.shape(occupancy) != mask.shape:
        raise ValueError("Geometry and field grid shapes differ")
    counts = []
    for field in rate_maps:
        peak = field[mask & (occupancy > 0)].max(initial=0)
        remaining = set(map(tuple, np.argwhere(mask & (occupancy > 0) & (field > 0) & (field >= 0.5 * peak))))
        count = 0
        while remaining:
            count += 1
            queue = [remaining.pop()]
            while queue:
                y, x = queue.pop()
                for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if p in remaining:
                        remaining.remove(p)
                        queue.append(p)
        counts.append(count)
    return {"geometry_field_components_half_peak": np.asarray(counts, dtype=np.int32)}


def _grid_distances(mask, starts):
    mask = np.asarray(mask, dtype=bool)
    result = np.full((len(starts), *mask.shape), np.inf, dtype=np.float32)
    for index, start in enumerate(starts):
        start = tuple(int(value) for value in start)
        if not mask[start]:
            raise ValueError("Cue site is not on accessible floor")
        result[index][start] = 0
        queue = deque([start])
        while queue:
            y, x = queue.popleft()
            for nxt in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= nxt[0] < mask.shape[0] and 0 <= nxt[1] < mask.shape[1]
                        and mask[nxt] and not np.isfinite(result[index][nxt])):
                    result[index][nxt] = result[index][y, x] + 1
                    queue.append(nxt)
    return result


def _minimum_cost_assignment(cost):
    """Return an exact rectangular assignment without an optional SciPy dependency.

    This is the shortest-augmenting-path Hungarian algorithm.  The smaller
    dimension is fully matched, matching ``linear_sum_assignment`` semantics.
    Stable column scanning makes ties deterministic across hosts.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Assignment cost must be a matrix")
    original_rows, original_cols = matrix.shape
    if not original_rows or not original_cols:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    if not np.isfinite(matrix).all():
        raise ValueError("Assignment cost must be finite")
    transposed = original_rows > original_cols
    if transposed:
        matrix = matrix.T
    rows, cols = matrix.shape
    row_potential = np.zeros(rows + 1, dtype=np.float64)
    col_potential = np.zeros(cols + 1, dtype=np.float64)
    matched_row = np.zeros(cols + 1, dtype=np.int64)
    predecessor = np.zeros(cols + 1, dtype=np.int64)
    for row in range(1, rows + 1):
        matched_row[0] = row
        minimum = np.full(cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(cols + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, cols + 1):
                if used[candidate]:
                    continue
                reduced = (
                    matrix[current_row - 1, candidate - 1]
                    - row_potential[current_row]
                    - col_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            if not np.isfinite(delta):
                raise ValueError("Assignment has no finite completion")
            for candidate in range(cols + 1):
                if used[candidate]:
                    row_potential[matched_row[candidate]] += delta
                    col_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    pairs = [(int(matched_row[column] - 1), column - 1)
             for column in range(1, cols + 1) if matched_row[column]]
    if transposed:
        pairs = [(column, row) for row, column in pairs]
    pairs.sort()
    return tuple(np.asarray(values, dtype=int) for values in zip(*pairs))


def cue_spatial_metrics(rate_maps, occupancy, record):
    """Capacity-aware cue/field alignment for one spatial snapshot."""
    if record.get("schema") != GEOMETRY_SCHEMA_V2:
        return {}, []

    maps = np.asarray(rate_maps, dtype=np.float32)
    occupancy = np.asarray(occupancy)
    mask = np.asarray(record["accessible_mask"], dtype=bool)
    if maps.ndim != 3 or maps.shape[1:] != mask.shape or occupancy.shape != mask.shape:
        raise ValueError("Cue metrics require unit-first cell-aligned rate maps")
    cue_yx = np.asarray([site["floor_yx"] for site in record["cue_sites"]], dtype=int)
    visited = occupancy[cue_yx[:, 0], cue_yx[:, 1]] > 0
    valid_bins = mask & (occupancy > 0)
    active_units = np.flatnonzero(np.max(np.where(valid_bins[None], maps, 0), axis=(1, 2)) > 0)
    peaks = []
    for unit in active_units:
        peak_flat = int(np.argmax(np.where(valid_bins, maps[unit], -np.inf)))
        peaks.append(np.unravel_index(peak_flat, mask.shape))
    cue_distances = _grid_distances(mask, cue_yx)
    cost = np.asarray([[cue_distances[cue][peak] for unit, peak in enumerate(peaks)]
                       for cue in range(len(cue_yx))], dtype=np.float32)
    assignments = []
    matched_within_one = 0
    assigned_cues, assigned_units = set(), set()
    if cost.size:
        cue_indices, unit_indices = _minimum_cost_assignment(cost)
        for cue_index, active_index in zip(cue_indices.tolist(), unit_indices.tolist()):
            unit = int(active_units[active_index])
            distance = float(cost[cue_index, active_index])
            assigned_cues.add(cue_index)
            assigned_units.add(unit)
            matched_within_one += int(distance <= 1 and visited[cue_index])
            site = record["cue_sites"][cue_index]
            assignments.append({
                "cue_id": site["cue_id"],
                "cue_type": site["cue_type"],
                "cue_visited": int(visited[cue_index]),
                "unit_id": unit,
                "peak_distance_cells": distance,
                "within_one_cell": int(distance <= 1),
                "cue_activation": float(maps[unit, cue_yx[cue_index, 0], cue_yx[cue_index, 1]]),
            })
    for cue_index, site in enumerate(record["cue_sites"]):
        if cue_index not in assigned_cues:
            assignments.append({
                "cue_id": site["cue_id"], "cue_type": site["cue_type"],
                "cue_visited": int(visited[cue_index]), "unit_id": -1,
                "peak_distance_cells": float("nan"), "within_one_cell": 0,
                "cue_activation": float("nan"),
            })
    for unit in active_units:
        if int(unit) not in assigned_units:
            assignments.append({
                "cue_id": "", "cue_type": "unmatched_unit", "cue_visited": 0,
                "unit_id": int(unit), "peak_distance_cells": float("nan"),
                "within_one_cell": 0, "cue_activation": float("nan"),
            })
    nearest = [float(np.min(cue_distances[:, peak[0], peak[1]])) for peak in peaks]
    denominator = min(len(cue_yx), len(active_units), int(visited.sum()))
    summary = {
        "cue_site_visit_fraction": float(visited.mean()),
        "cue_active_unit_count": int(len(active_units)),
        "cue_peak_nearest_distance_mean": float(np.mean(nearest)) if nearest else float("nan"),
        "cue_peak_match_count": int(matched_within_one),
        "cue_peak_coverage_fraction": matched_within_one / len(cue_yx),
        "cue_peak_capacity_normalized_coverage": matched_within_one / denominator if denominator else float("nan"),
    }
    for cue_type in ("decal", "color"):
        selected = np.asarray([site["cue_type"] == cue_type for site in record["cue_sites"]])
        summary[f"cue_{cue_type}_site_visit_fraction"] = float(visited[selected].mean())
        summary[f"cue_{cue_type}_peak_match_count"] = int(sum(
            row["within_one_cell"] and row["cue_visited"] and row["cue_type"] == cue_type
            for row in assignments
        ))
    assignments.sort(key=lambda row: (row["cue_id"] == "", row["cue_id"], row["unit_id"]))
    return summary, assignments


def validate_geometry_payload(payload):
    """Validate optional geometry fields without changing legacy NPZ contracts."""
    keys = [key for key in payload if key.startswith("geometry_")]
    if not keys:
        return
    required = (
        "geometry_schema",
        "geometry_sha256",
        "geometry_entity_layer",
        "geometry_map_seed",
        "geometry_wall_removal_probability",
        "geometry_accessible_mask",
        "geometry_bounds",
        "geometry_cell_size",
        "geometry_coordinate_contract",
    )
    if any(key not in payload for key in required):
        raise ValueError("Incomplete optional geometry payload")
    schema = str(np.asarray(payload["geometry_schema"]).item())
    if schema == GEOMETRY_SCHEMA_V1:
        record = entity_record(
            str(np.asarray(payload["geometry_entity_layer"]).item()),
            map_seed=int(np.asarray(payload["geometry_map_seed"]).item()),
            wall_removal_probability=float(np.asarray(payload["geometry_wall_removal_probability"]).item()),
        )
    elif schema == GEOMETRY_SCHEMA_V2:
        cue_required = (
            "geometry_entity_shape",
            "geometry_cue_layout_seed", "geometry_cue_mode", "geometry_cue_layout_sha256",
            "geometry_cue_ids", "geometry_cue_types", "geometry_cue_assets",
            "geometry_cue_wall_rc", "geometry_cue_floor_yx", "geometry_cue_orientations",
            "geometry_cue_rendered",
        )
        if any(key not in payload for key in cue_required):
            raise ValueError("Incomplete optional cue geometry payload")
        record = landmark_entity_record(
            str(np.asarray(payload["geometry_entity_layer"]).item()),
            map_seed=int(np.asarray(payload["geometry_map_seed"]).item()),
            wall_removal_probability=float(
                np.asarray(payload["geometry_wall_removal_probability"]).item()
            ),
            cue_layout_seed=int(np.asarray(payload["geometry_cue_layout_seed"]).item()),
            cue_mode=str(np.asarray(payload["geometry_cue_mode"]).item()),
        )
    else:
        raise ValueError(f"Unsupported geometry schema: {schema}")
    expected = geometry_payload(record)
    for key, value in expected.items():
        if not np.array_equal(payload[key], value):
            raise ValueError(f"Geometry payload disagrees with entity map: {key}")
    if "bounds" in payload and not np.array_equal(payload["bounds"], expected["geometry_bounds"]):
        raise ValueError("Spatial bounds differ from geometry bounds")
    accessible_shape = np.asarray(record["accessible_mask"]).shape
    if len(accessible_shape) != 2 or accessible_shape[0] != accessible_shape[1]:
        raise ValueError(
            "The scalar spatial grain contract requires a square accessible mask"
        )
    if "grain" in payload and int(np.asarray(payload["grain"])) != accessible_shape[0]:
        raise ValueError(
            "Geometry requires a cell-aligned "
            f"{accessible_shape[0]}x{accessible_shape[1]} spatial grid"
        )
