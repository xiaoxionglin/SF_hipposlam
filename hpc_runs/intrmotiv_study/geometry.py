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

GEOMETRY_SCHEMA = "intrmotiv/map-geometry/v1"


def entity_record(entity: str, *, map_seed: int, wall_removal_probability: float) -> dict:
    rows = entity.splitlines()
    if len(rows) != 21 or any(len(row) != 21 for row in rows):
        raise ValueError("Expected a 21x21 entity map")
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
        bounds=[100.0, 2000.0, 100.0, 2000.0],
        cell_size=100.0,
        coordinate_contract="mask[y_bin,x_bin]; x=floor(x_world/100)-1; y=floor(y_world/100)-1",
        accessible_cells=len(cells),
        corridor_fraction=float(np.mean(degree == 2)),
        dead_ends=int(np.sum(degree == 1)),
        junctions=int(np.sum(degree >= 3)),
        shortest_path_histogram=np.bincount(distances).tolist(),
        shortest_path_quantiles=np.quantile(distances, [0, 0.25, 0.5, 0.75, 1]).tolist(),
    )


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


def geometry_from_config(cfg):
    if getattr(cfg, "env", None) != "corridor_geometry_noreward":
        return None
    manifest = getattr(cfg, "dmlab_geometry_manifest", "") or str(
        Path(__file__).resolve().parents[1] / "studies/assets/corridor_geometry/maps.json"
    )
    return load_geometry(manifest, int(cfg.dmlab_map_seed), float(cfg.dmlab_wall_removal_probability))


def verify_entity(entity, record):
    if isinstance(entity, bytes):
        entity = entity.decode()
    if not isinstance(entity, str) or sha256(entity.encode()).hexdigest() != record["sha256"]:
        raise ValueError("Loaded DMLab geometry does not match the archived map SHA-256")


def geometry_payload(record):
    if record is None:
        return {}
    return {
        "geometry_schema": np.asarray(GEOMETRY_SCHEMA),
        "geometry_sha256": np.asarray(record["sha256"]),
        "geometry_entity_layer": np.asarray(record["entity_layer"]),
        "geometry_map_seed": np.asarray(record["map_seed"]),
        "geometry_wall_removal_probability": np.asarray(record["wall_removal_probability"]),
        "geometry_accessible_mask": np.asarray(record["accessible_mask"], dtype=bool),
        "geometry_bounds": np.asarray(record["bounds"]),
        "geometry_cell_size": np.asarray(record["cell_size"]),
        "geometry_coordinate_contract": np.asarray(record["coordinate_contract"]),
    }


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
    record = entity_record(
        str(np.asarray(payload["geometry_entity_layer"]).item()),
        map_seed=int(np.asarray(payload["geometry_map_seed"]).item()),
        wall_removal_probability=float(np.asarray(payload["geometry_wall_removal_probability"]).item()),
    )
    expected = geometry_payload(record)
    for key, value in expected.items():
        if not np.array_equal(payload[key], value):
            raise ValueError(f"Geometry payload disagrees with entity map: {key}")
    if "bounds" in payload and not np.array_equal(payload["bounds"], expected["geometry_bounds"]):
        raise ValueError("Spatial bounds differ from geometry bounds")
    if "grain" in payload and int(np.asarray(payload["grain"])) != 19:
        raise ValueError("Geometry requires a cell-aligned 19x19 spatial grid")
