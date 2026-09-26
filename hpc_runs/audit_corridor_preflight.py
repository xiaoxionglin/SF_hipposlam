"""Geometry/coverage gates layered on the canonical controller runtime audit."""

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
        if cfg["with_pos_obs"] or cfg.get("hrl_landmark_geometry", "none") != "none":
            row["errors"].append("privileged policy input enabled")
        events = _events(directory)
        prefix = "policy_stats/avg_z_00_corridor_geometry_noreward_"
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
            measurements[suffix] = dict(step=latest.step, value=value)
            if not math.isfinite(value):
                row["errors"].append("nonfinite episode measurement " + suffix)
            if suffix in {"dmlab_raw_score", "geometry_invalid_pose_steps"} and value != 0:
                row["errors"].append("nonzero final " + suffix)
            if suffix == "len" and value != 7200:
                row["errors"].append("episode timing is not 7200 frames")
            if suffix.startswith("accessible_coverage") and not 0 <= value <= 1:
                row["errors"].append("coverage outside [0, 1]")
        for tag in ("intrmotiv/reward/environment_mean", "intrmotiv/reward/environment_nonzero_fraction"):
            if not events.get(tag) or any(float(event.value) != 0 for event in events[tag]):
                row["errors"].append("missing or nonzero external reward: " + tag)
        snapshots = root / "analysis/online_spatial" / study.batch_name / row["run"] / "policy_00"
        for path in snapshots.glob("*.npz"):
            payload = load_spatial_snapshot(path)
            if str(payload["geometry_sha256"].item()) != geometry["sha256"]:
                row["errors"].append("snapshot geometry differs from run configuration")
            mask = payload["geometry_accessible_mask"].astype(bool)
            if np.asarray(payload["occupancy"])[~mask].sum() != 0:
                row["errors"].append("snapshot contains occupied wall cells")
        row["geometry_sha256"] = geometry["sha256"]
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
