"""Fail-closed runtime audit for the six navigation8 algorithm preflights."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.spec import SpecError


def _events(run_dir: Path) -> dict[str, list]:
    values: dict[str, list] = {}
    for directory in sorted({path.parent for path in run_dir.rglob("events.out.tfevents.*")}):
        accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 100_000}).Reload()
        for tag in accumulator.Tags().get("scalars", []):
            values.setdefault(tag, []).extend(accumulator.Scalars(tag))
    return values


def _checkpoint_frames(run_dir: Path) -> list[int]:
    frames = []
    for path in run_dir.glob("checkpoint_p*/checkpoint_*.pth"):
        match = re.search(r"_(\d+)\.pth$", path.name)
        if match:
            frames.append(int(match.group(1)))
    return frames


def audit(study, jobs_tsv: Path, train_root: Path, analysis_root: Path, required_frames: int) -> dict:
    with jobs_tsv.open(newline="", encoding="utf-8") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    expected = {run.name: run for run in study.expand_runs()}
    if len(jobs) != len(expected):
        raise SpecError(f"submission has {len(jobs)} rows; expected {len(expected)}")

    records = []
    for job in jobs:
        experiment = job["experiment"]
        run_name = experiment[3:] if experiment.startswith("00_") else experiment
        if run_name not in expected:
            raise SpecError(f"unexpected run {run_name!r}")
        run = expected[run_name]
        run_dir = train_root / job["train_root"] / experiment
        errors = []

        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        required_config = {
            "dmlab_navigation_action_set": True,
            "dmlab_reduced_action_set": False,
            "dmlab_extended_action_set": False,
            "env_frameskip": 4,
            "with_pos_obs": False,
        }
        for key, value in required_config.items():
            if config.get(key) != value:
                errors.append(f"config {key}={config.get(key)!r}; expected {value!r}")

        values = _events(run_dir) if run_dir.is_dir() else {}
        frame_events = values.get("train/env_steps", [])
        max_frames = max((float(event.value) for event in frame_events), default=math.nan)
        if not math.isfinite(max_frames) or max_frames < required_frames:
            errors.append(f"max env frames {max_frames!r} is below {required_frames}")
        for tag, events in values.items():
            if "loss" in tag.lower() or "gradient" in tag.lower():
                if not events or any(not math.isfinite(float(event.value)) for event in events):
                    errors.append(f"nonfinite or empty learning scalar {tag}")

        checkpoints = _checkpoint_frames(run_dir) if run_dir.is_dir() else []
        if max(checkpoints, default=0) < required_frames:
            errors.append("terminal checkpoint is missing or below the required frame count")

        spatial_dir = analysis_root / study.batch_name / run.name / "policy_00"
        observed_targets = {
            int(match.group(1))
            for path in spatial_dir.glob("snapshot_target_*_actual_*.npz")
            if (match := re.search(r"snapshot_target_(\d+)_actual_", path.name))
        }
        required_targets = set(study.telemetry["target_frames"])
        if not required_targets.issubset(observed_targets):
            errors.append(f"missing spatial targets {sorted(required_targets - observed_targets)}")

        stdout = Path(job["stdout"].replace("%j", job.get("job_id", "")))
        stderr = Path(job["stderr"].replace("%j", job.get("job_id", "")))
        log_text = "\n".join(path.read_text(errors="replace") for path in (stdout, stderr) if path.is_file())
        if "using eight-action navigation set!" not in log_text:
            errors.append("runtime did not log navigation action-set selection")
        if "Traceback (most recent call last)" in log_text or "RuntimeError:" in log_text:
            errors.append("runtime log contains an exception")

        records.append(
            {
                "run_name": run.name,
                "job_id": job.get("job_id"),
                "max_env_frames": max_frames,
                "checkpoint_frames": max(checkpoints, default=0),
                "spatial_targets": sorted(observed_targets),
                "pass": not errors,
                "errors": errors,
            }
        )

    return {
        **study.provenance(),
        "protocol": "navigation8-algorithm-screen-preflight-v1",
        "required_frames": required_frames,
        "expected_runs": len(expected),
        "passed_runs": sum(record["pass"] for record in records),
        "passed": len(records) == len(expected) and all(record["pass"] for record in records),
        "runs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("jobs_tsv", type=Path)
    parser.add_argument("train_root", type=Path)
    parser.add_argument("analysis_root", type=Path)
    parser.add_argument("--required-frames", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(load_study(args.study), args.jobs_tsv, args.train_root, args.analysis_root, args.required_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
