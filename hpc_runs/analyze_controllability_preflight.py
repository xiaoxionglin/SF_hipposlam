"""Evaluate runtime gates for the 16-cell controllability preflight."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.spec import SpecError


STEP_TAG = "train/env_steps"
MODE_TAGS = {
    "free": "intrmotiv/hrl/mode/free_fraction",
    "goal": "intrmotiv/hrl/mode/goal_fraction",
    "probe": "intrmotiv/hrl/mode/probe_fraction",
}
REPLAY_TAG = "intrmotiv/hrl/behavior_replay_mismatch"
BRANCH_TAGS = {
    "goal": (
        "intrmotiv/hrl/branch/goal_policy_loss",
        "intrmotiv/hrl/branch/goal_value_loss",
    ),
    "free": (
        "intrmotiv/hrl/branch/free_policy_loss",
        "intrmotiv/hrl/branch/free_value_loss",
    ),
}


def _values(accumulator, tag: str) -> list[float]:
    return [float(event.value) for event in accumulator.Scalars(tag)]


def _active(values: list[float], tolerance: float = 1e-8) -> bool:
    return any(math.isfinite(value) and abs(value) > tolerance for value in values)


def analyze(study, jobs_tsv: Path, train_root: Path) -> dict:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as error:
        raise RuntimeError("TensorBoard is required for preflight analysis") from error

    with jobs_tsv.open(newline="", encoding="utf-8") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    expected = {run.name: run for run in study.expand_runs() if run.seed == 99}
    if len(jobs) != len(expected):
        raise SpecError(f"preflight has {len(jobs)} rows; expected {len(expected)}")

    results = []
    all_failures: list[str] = []
    for job in jobs:
        experiment = job["experiment"]
        if not experiment.startswith("00_PF_"):
            raise SpecError(f"unexpected preflight experiment {experiment!r}")
        run_name = experiment[len("00_PF_"):]
        if run_name not in expected:
            raise SpecError(f"unexpected preflight run {run_name!r}")
        run = expected[run_name]
        # Sample Factory's launcher assigns a per-experiment train root and
        # the training entry point creates the concrete experiment directory
        # one level below it.
        run_dir = train_root / job["train_root"] / experiment
        summary_dir = run_dir / ".summary" / "0"
        config_path = run_dir / "config.json"
        failures: list[str] = []
        if not summary_dir.is_dir():
            failures.append("missing TensorBoard summary")
        if not config_path.is_file():
            failures.append("missing config.json")
        if failures:
            results.append({"run_name": run_name, "run_dir": str(run_dir), "failures": failures})
            all_failures.extend(f"{run_name}: {failure}" for failure in failures)
            continue

        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_config = {
            "train_for_env_steps": 1_000_000,
            "seed": 99,
            "hrl_target_timing": "immediate",
            "hrl_manager_mode": "control_graph",
            "hrl_behavior_mode_condition": True,
            "hrl_goal_conditioning": "target_trace",
        }
        for key, value in expected_config.items():
            if config.get(key) != value:
                failures.append(f"config {key}={config.get(key)!r}, expected {value!r}")
        expected_factor_config = {
            "dg_ca3_temporal_exclusion_coeff": 1.0 if run.factors["representation"] == "X1" else 0.0,
            "hrl_exploration_policy": run.factors["exploration_policy"],
            "hrl_edge_exploration": run.factors["manager_objective"] == "edge_ucb",
            "hrl_landmark_geometry": run.factors["geometry"],
        }
        for key, value in expected_factor_config.items():
            if config.get(key) != value:
                failures.append(f"config {key}={config.get(key)!r}, expected {value!r}")

        accumulator = EventAccumulator(str(summary_dir), size_guidance={"scalars": 50_000})
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        required = {STEP_TAG, REPLAY_TAG, *MODE_TAGS.values()}
        required.update(tag for tags in BRANCH_TAGS.values() for tag in tags)
        missing = sorted(required - available)
        if missing:
            failures.append(f"missing scalar tags: {missing}")

        max_env_steps = math.nan
        mode_max = {mode: math.nan for mode in MODE_TAGS}
        replay_max = math.nan
        finite_loss_tags = 0
        branch_active = {branch: False for branch in BRANCH_TAGS}
        if STEP_TAG in available:
            max_env_steps = max(_values(accumulator, STEP_TAG), default=math.nan)
            if not math.isfinite(max_env_steps) or max_env_steps < 1_000_000:
                failures.append(f"max env steps {max_env_steps!r} is below 1M")
        loss_tags = sorted(tag for tag in available if "loss" in tag.lower())
        for tag in loss_tags:
            values = _values(accumulator, tag)
            if not values or not all(math.isfinite(value) for value in values):
                failures.append(f"nonfinite or empty loss scalar {tag}")
            else:
                finite_loss_tags += 1
        if not loss_tags:
            failures.append("no loss scalars")
        if REPLAY_TAG in available:
            replay_values = _values(accumulator, REPLAY_TAG)
            replay_max = max((abs(value) for value in replay_values), default=math.nan)
            if not math.isfinite(replay_max) or replay_max > 1e-6:
                failures.append(f"behavior replay mismatch max is {replay_max!r}")
        for mode, tag in MODE_TAGS.items():
            if tag in available:
                values = _values(accumulator, tag)
                mode_max[mode] = max(values, default=math.nan)
        if not math.isfinite(mode_max["free"]) or mode_max["free"] <= 0:
            failures.append("manager mode free had no activity")
        edge_cell = run.factors["manager_objective"] == "edge_ucb"
        if edge_cell and (not math.isfinite(mode_max["probe"]) or mode_max["probe"] <= 0):
            failures.append("edge-aware cell had no EDGE_PROBE activity")
        if not edge_cell and math.isfinite(mode_max["probe"]) and mode_max["probe"] > 1e-8:
            failures.append(f"node-only cell unexpectedly probed ({mode_max['probe']})")
        for branch, tags in BRANCH_TAGS.items():
            if all(tag in available for tag in tags):
                branch_active[branch] = any(_active(_values(accumulator, tag)) for tag in tags)
        if not branch_active["free"]:
            failures.append("free branch produced no nonzero loss")
        # EDGE_PROBE is goal-directed, so edge-aware cells must exercise the
        # goal branch even if no multi-hop NAVIGATE option is available yet.
        if edge_cell and not branch_active["goal"]:
            failures.append("edge-aware cell's goal branch produced no nonzero loss")

        result = {
            "run_name": run_name,
            "condition": run.condition,
            **run.factors,
            "run_dir": str(run_dir),
            "max_env_steps": max_env_steps,
            "finite_loss_tags": finite_loss_tags,
            "behavior_replay_mismatch_max": replay_max,
            **{f"mode_{mode}_max": value for mode, value in mode_max.items()},
            **{f"{branch}_branch_active": value for branch, value in branch_active.items()},
            "pass": not failures,
            "failures": failures,
        }
        results.append(result)
        all_failures.extend(f"{run_name}: {failure}" for failure in failures)

    passed = sum(bool(result.get("pass")) for result in results)
    mode_means = {}
    for mode in MODE_TAGS:
        values = [
            result[f"mode_{mode}_max"]
            for result in results
            if math.isfinite(result.get(f"mode_{mode}_max", math.nan))
        ]
        mode_means[mode] = fmean(values) if values else math.nan
    return {
        **study.provenance(),
        "protocol": "controllability-preflight-v1",
        "jobs_tsv": str(jobs_tsv),
        "expected_cells": len(expected),
        "passed_cells": passed,
        "all_cells_pass": passed == len(expected),
        "branch_gradient_isolation_test": "test_controllability_policy_branches.py",
        "mode_max_means": mode_means,
        "failures": all_failures,
        "runs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("jobs_tsv", type=Path)
    parser.add_argument("train_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        payload = analyze(load_study(args.study), args.jobs_tsv, args.train_root)
    except (OSError, RuntimeError, SpecError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"preflight cells passed: {payload['passed_cells']}/{payload['expected_cells']}; "
        f"report: {args.output}"
    )
    if not payload["all_cells_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
