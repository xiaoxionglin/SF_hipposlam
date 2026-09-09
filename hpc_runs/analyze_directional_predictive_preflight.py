"""Evaluate runtime gates for the 18-cell directional/predictive preflight."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.spec import SpecError


STEP_TAG = "train/env_steps"
REPLAY_TAG = "intrmotiv/hrl/behavior_replay_mismatch"
TARGET_VALID_TAG = "intrmotiv/hrl/goal_condition/target_valid_fraction"
FILM_NORM_TAG = "intrmotiv/hrl/goal_condition/film_modulation_norm_mean"
RECRUITMENT_TAG = "intrmotiv/dg/recruitment/total"
BAD_ASSIGNMENT_TAG = "intrmotiv/dg/recruitment/bad_source_assignments_per_rollout"
PRED_ASSIGNMENT_TAG = "intrmotiv/dg/recruitment/predictive_assignments_per_rollout"
FULLY_TESTED_TAG = "intrmotiv/dg/recruitment/fully_tested_count"
PRED_EVENT_TAG = "intrmotiv/dg/recruitment/predictive_event_count"
ACTIVE_TARGET_TAG = "intrmotiv/hrl/active_target_fraction"
EXPLORATION_TAG = "intrmotiv/hrl/exploration/selection_fraction"
FRONTIER_TAG = "intrmotiv/hrl/frontier/selection_rate"


def _values(accumulator, tag: str) -> list[float]:
    return [float(event.value) for event in accumulator.Scalars(tag)]


def _finite_max(accumulator, tag: str) -> float:
    values = _values(accumulator, tag)
    return max(values, default=math.nan)


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
    all_failures = []
    for job in jobs:
        experiment = job["experiment"]
        if not experiment.startswith("00_PF_"):
            raise SpecError(f"unexpected preflight experiment {experiment!r}")
        run_name = experiment[len("00_PF_"):]
        run = expected[run_name]
        run_dir = train_root / job["train_root"] / experiment
        summary_dir = run_dir / ".summary" / "0"
        config_path = run_dir / "config.json"
        failures = []
        if not summary_dir.is_dir():
            failures.append("missing TensorBoard summary")
        if not config_path.is_file():
            failures.append("missing config.json")
        if failures:
            results.append({"run_name": run_name, "run_dir": str(run_dir), "pass": False, "failures": failures})
            all_failures.extend(f"{run_name}: {failure}" for failure in failures)
            continue

        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_config = {
            "train_for_env_steps": 5_000_000,
            "seed": 99,
            "hrl_target_timing": "immediate",
            "dg_recruitment_victim_rule": run.factors["recruitment"],
            "hrl_goal_conditioning": run.factors["goal_conditioning"],
            "dg_recruitment_attempt_threshold": 0.5,
            "dg_recruitment_redundancy_max_steps": 4,
            "hrl_fast_weight_half_life_options": 5000,
        }
        for key, value in expected_config.items():
            if config.get(key) != value:
                failures.append(f"config {key}={config.get(key)!r}, expected {value!r}")

        accumulator = EventAccumulator(str(summary_dir), size_guidance={"scalars": 100_000})
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        required = {
            STEP_TAG,
            REPLAY_TAG,
            TARGET_VALID_TAG,
            RECRUITMENT_TAG,
            FULLY_TESTED_TAG,
            PRED_EVENT_TAG,
            ACTIVE_TARGET_TAG,
        }
        missing = sorted(required - available)
        if missing:
            failures.append(f"missing scalar tags: {missing}")

        max_steps = _finite_max(accumulator, STEP_TAG) if STEP_TAG in available else math.nan
        if not math.isfinite(max_steps) or max_steps < 5_000_000:
            failures.append(f"max env steps {max_steps!r} is below 5M")
        loss_tags = sorted(tag for tag in available if "loss" in tag.lower())
        if not loss_tags:
            failures.append("no loss scalars")
        for tag in loss_tags:
            values = _values(accumulator, tag)
            if not values or not all(math.isfinite(value) for value in values):
                failures.append(f"nonfinite or empty loss scalar {tag}")

        replay_max = max((abs(value) for value in _values(accumulator, REPLAY_TAG)), default=math.nan) if REPLAY_TAG in available else math.nan
        if not math.isfinite(replay_max) or replay_max > 1e-6:
            failures.append(f"behavior replay mismatch max is {replay_max!r}")
        target_valid_max = _finite_max(accumulator, TARGET_VALID_TAG) if TARGET_VALID_TAG in available else math.nan
        if not math.isfinite(target_valid_max) or target_valid_max <= 0:
            failures.append("goal-directed samples never carried a valid replay target")

        if run.factors["goal_conditioning"] == "target_id_film":
            if FILM_NORM_TAG not in available or _finite_max(accumulator, FILM_NORM_TAG) <= 0:
                failures.append("FiLM modulation remained zero")
        recruitment_max = _finite_max(accumulator, RECRUITMENT_TAG) if RECRUITMENT_TAG in available else math.nan
        if run.factors["recruitment"] == "monitor" and recruitment_max != 0:
            failures.append(f"MON recruitment total is {recruitment_max!r}, expected zero")
        if run.factors["recruitment"] == "directional" and BAD_ASSIGNMENT_TAG in available:
            bad_assignment_max = _finite_max(accumulator, BAD_ASSIGNMENT_TAG)
            fully_tested_max = _finite_max(accumulator, FULLY_TESTED_TAG)
            if bad_assignment_max > 0 and fully_tested_max <= 0:
                failures.append("DIR assigned a bad source before any source was fully tested")
        if run.factors["recruitment"] in ("monitor", "predictive"):
            pred_event_max = _finite_max(accumulator, PRED_EVENT_TAG) if PRED_EVENT_TAG in available else math.nan
            if not math.isfinite(pred_event_max) or pred_event_max <= 0:
                failures.append("batch-local PRED extraction had no completed contextual events")

        active_target_max = _finite_max(accumulator, ACTIVE_TARGET_TAG) if ACTIVE_TARGET_TAG in available else math.nan
        if not math.isfinite(active_target_max) or active_target_max <= 0:
            failures.append("goal manager had no target activity")
        if run.base == "C13":
            if EXPLORATION_TAG not in available or _finite_max(accumulator, EXPLORATION_TAG) <= 0:
                failures.append("C13-like exploration branch had no activity")
        if run.base == "C15":
            if FRONTIER_TAG not in available or _finite_max(accumulator, FRONTIER_TAG) <= 0:
                failures.append("C15 frontier manager had no selection activity")

        result = {
            "run_name": run_name,
            "condition": run.condition,
            "base": run.base,
            **run.factors,
            "run_dir": str(run_dir),
            "max_env_steps": max_steps,
            "behavior_replay_mismatch_max": replay_max,
            "target_valid_max": target_valid_max,
            "recruitment_total_max": recruitment_max,
            "pass": not failures,
            "failures": failures,
        }
        results.append(result)
        all_failures.extend(f"{run_name}: {failure}" for failure in failures)

    passed = sum(bool(result.get("pass")) for result in results)
    return {
        **study.provenance(),
        "protocol": "directional-predictive-preflight-v1",
        "jobs_tsv": str(jobs_tsv),
        "expected_cells": len(expected),
        "passed_cells": passed,
        "all_cells_pass": passed == len(expected),
        "film_gradient_test": "test_controllability_policy_branches.py",
        "directional_eligibility_test": "test_graph_stabilized_recruitment.py",
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
    except (OSError, RuntimeError, SpecError, ValueError, KeyError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"preflight cells passed: {payload['passed_cells']}/{payload['expected_cells']}; report: {args.output}")
    if not payload["all_cells_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
