"""Fail-closed runtime, geometry, and cue checks for the six-run preflight."""

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hpc_runs.audit_full_system_controller_preflight import audit as audit_controller
from hpc_runs.audit_navigation8_algorithm_screen_preflight import _events
from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.geometry import geometry_from_config
from hpc_runs.intrmotiv_study.spatial_contract import load_spatial_snapshot


def audit(spec, jobs, train_root):
    study = load_study(spec)
    result = audit_controller(spec, jobs, train_root, required_frames=2_000_000)
    by_name = {row["run"]: row for row in result["runs"]}
    root = Path(train_root)
    for job in csv.DictReader(Path(jobs).open(), delimiter="\t"):
        row = by_name[job["experiment"].removeprefix("00_")]
        directory = root / job["train_root"] / job["experiment"]
        cfg = json.loads((directory / "config.json").read_text())
        geometry = geometry_from_config(SimpleNamespace(**cfg))
        entity_rows = geometry["entity_layer"].splitlines()
        if geometry.get("entity_shape") != [11, 11] or any(len(line) != 11 for line in entity_rows):
            row["errors"].append("geometry is not an exact native 11x11 entity layer")
        if np.asarray(geometry["accessible_mask"]).shape != (9, 9):
            row["errors"].append("geometry does not expose a 9x9 accessible mask")
        if any(marker in geometry["entity_layer"] for marker in ("G", "A")):
            row["errors"].append("geometry contains a goal or pickup entity")
        if cfg.get("dmlab_map_rows") != 11 or cfg.get("dmlab_map_cols") != 11:
            row["errors"].append("runtime map dimensions are not 11x11")
        if cfg.get("online_spatial_grid_grain") != 9:
            row["errors"].append("online spatial grid is not 9x9")
        expected_bounds = [100.0, 1000.0, 100.0, 1000.0]
        configured_bounds = [
            cfg.get("online_spatial_x_min"),
            cfg.get("online_spatial_x_max"),
            cfg.get("online_spatial_y_min"),
            cfg.get("online_spatial_y_max"),
        ]
        if configured_bounds != expected_bounds:
            row["errors"].append("online spatial bounds differ from native geometry")
        if cfg["with_pos_obs"] or cfg.get("hrl_landmark_geometry", "none") != "none":
            row["errors"].append("privileged policy input enabled")
        if len(geometry["cue_sites"]) != 20:
            row["errors"].append("cue manifest does not contain exactly 20 sites")
        expected_rendered = cfg["dmlab_landmark_cues"] == "rich"
        events = _events(directory)
        prefix = "policy_stats/avg_z_00_easy_landmark_maze_noreward_"
        measurements = {}
        for suffix in (
            "dmlab_raw_score",
            "len",
            "accessible_coverage_auc",
            "accessible_coverage_fraction",
            "geometry_invalid_pose_steps",
        ):
            values = events.get(prefix + suffix, [])
            if not values:
                row["errors"].append("missing episode measurement " + suffix)
                continue
            latest = max(values, key=lambda event: (event.step, event.wall_time))
            value = float(latest.value)
            measurements[suffix] = {"step": latest.step, "value": value}
            if not math.isfinite(value):
                row["errors"].append("nonfinite episode measurement " + suffix)
            if suffix in {"dmlab_raw_score", "geometry_invalid_pose_steps"} and value != 0:
                row["errors"].append("nonzero final " + suffix)
            if suffix == "len" and value != 7200:
                row["errors"].append("episode timing is not 7200 frames")
            if suffix.startswith("accessible_coverage") and not 0 <= value <= 1:
                row["errors"].append("coverage outside [0, 1]")
        for tag in (
            "intrmotiv/reward/environment_mean",
            "intrmotiv/reward/environment_nonzero_fraction",
        ):
            if not events.get(tag) or any(float(event.value) != 0 for event in events[tag]):
                row["errors"].append("missing or nonzero external reward: " + tag)
        snapshots = root / "analysis/online_spatial" / study.batch_name / row["run"] / "policy_00"
        observed_snapshots = 0
        for path in snapshots.glob("*.npz"):
            observed_snapshots += 1
            payload = load_spatial_snapshot(path)
            rendered = np.asarray(payload.get("geometry_cue_rendered", []), dtype=bool)
            if rendered.size != 20 or bool(rendered.all()) != expected_rendered:
                row["errors"].append("snapshot cue rendering contract mismatch")
            if str(payload["geometry_cue_layout_sha256"].item()) != geometry["cue_layout_sha256"]:
                row["errors"].append("snapshot cue layout differs from run configuration")
            mask = payload["geometry_accessible_mask"].astype(bool)
            if np.asarray(payload["occupancy"])[~mask].sum() != 0:
                row["errors"].append("snapshot contains occupied wall cells")
        if observed_snapshots != 2:
            row["errors"].append(f"expected two spatial snapshots, found {observed_snapshots}")
        row["geometry_sha256"] = geometry["sha256"]
        row["cue_layout_sha256"] = geometry["cue_layout_sha256"]
        row["episode_measurements"] = measurements
        row["passed"] = not row["errors"]
    result["passed"] = all(row["passed"] for row in result["runs"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    parser.add_argument("jobs")
    parser.add_argument("train_root")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.study, args.jobs, args.train_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["passed"] else 1)
