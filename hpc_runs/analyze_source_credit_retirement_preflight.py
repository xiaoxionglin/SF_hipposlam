"""Evaluate runtime gates for the ten-cell source-credit preflight."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.spec import SpecError


STEP = "train/env_steps"
REPLAY = "intrmotiv/hrl/behavior_replay_mismatch"
TARGET_VALID = "intrmotiv/hrl/goal_condition/target_valid_fraction"
FRONTIER = "intrmotiv/hrl/frontier/selection_rate"
PRED_EVENTS = "intrmotiv/dg/recruitment/predictive_event_count"
PRED_ELIGIBLE = "intrmotiv/dg/recruitment/predictive_eligible_count"
FULLY_TESTED = "intrmotiv/dg/recruitment/fully_tested_count"
RECRUITMENT = "intrmotiv/dg/recruitment/total"
BAD_ASSIGNMENT = "intrmotiv/dg/recruitment/bad_source_assignments_per_rollout"
PRED_ASSIGNMENT = "intrmotiv/dg/recruitment/predictive_assignments_per_rollout"
TOTAL_EVENTS = "intrmotiv/encoder/credit/total_events"
CREDITED_EVENTS = "intrmotiv/encoder/credit/credited_events"
ARRIVAL_LOSS = "intrmotiv/encoder/credit/arrival_loss"
SOURCE_LOSS = "intrmotiv/encoder/credit/source_loss"


def _values(accumulator, tag: str) -> list[float]:
    return [float(event.value) for event in accumulator.Scalars(tag)]


def _maximum(accumulator, tag: str) -> float:
    return max(_values(accumulator, tag), default=math.nan)


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
        prefix = "00_PF_"
        if not experiment.startswith(prefix):
            raise SpecError(f"unexpected preflight experiment {experiment!r}")
        run_name = experiment[len(prefix):]
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
            results.append({"run_name": run_name, "pass": False, "failures": failures})
            all_failures.extend(f"{run_name}: {failure}" for failure in failures)
            continue

        config = json.loads(config_path.read_text(encoding="utf-8"))
        retirement = run.factors["retirement"]
        expected_gate = "open" if retirement.endswith("open") else "silent"
        expected_rule = "monitor" if retirement == "monitor" else retirement.split("_", 1)[0]
        expected_rule = {"dir": "directional", "pred": "predictive"}.get(expected_rule, expected_rule)
        expected_config = {
            "train_for_env_steps": 5_000_000,
            "seed": 99,
            "encoder_reward_recipient": run.factors["encoder_credit"],
            "encoder_reward_require_local_predecessor": True,
            "dg_recruitment_victim_rule": expected_rule,
            "dg_recruitment_endpoint_gate": expected_gate,
            "dg_recruitment_pred_min_context_attempts": 4.0,
            "hrl_goal_conditioning": "target_id_film",
            "hrl_target_timing": "immediate",
        }
        for key, value in expected_config.items():
            if config.get(key) != value:
                failures.append(f"config {key}={config.get(key)!r}, expected {value!r}")

        accumulator = EventAccumulator(str(summary_dir), size_guidance={"scalars": 100_000})
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        required = {
            STEP, REPLAY, TARGET_VALID, FRONTIER, PRED_EVENTS, PRED_ELIGIBLE,
            FULLY_TESTED, RECRUITMENT, TOTAL_EVENTS, CREDITED_EVENTS,
            ARRIVAL_LOSS, SOURCE_LOSS,
        }
        missing = sorted(required - available)
        if missing:
            failures.append(f"missing scalar tags: {missing}")

        max_steps = _maximum(accumulator, STEP) if STEP in available else math.nan
        if not math.isfinite(max_steps) or max_steps < 5_000_000:
            failures.append(f"max env steps {max_steps!r} is below 5M")
        loss_tags = sorted(tag for tag in available if "loss" in tag.lower())
        if not loss_tags:
            failures.append("no loss scalars")
        for tag in loss_tags:
            values = _values(accumulator, tag)
            if not values or not all(math.isfinite(value) for value in values):
                failures.append(f"nonfinite or empty loss scalar {tag}")

        replay_max = max((abs(value) for value in _values(accumulator, REPLAY)), default=math.nan) if REPLAY in available else math.nan
        if not math.isfinite(replay_max) or replay_max > 1e-6:
            failures.append(f"behavior replay mismatch max is {replay_max!r}")
        for tag, label in ((TARGET_VALID, "goal target"), (FRONTIER, "frontier manager"), (PRED_EVENTS, "PRED evidence")):
            if tag not in available or _maximum(accumulator, tag) <= 0:
                failures.append(f"{label} had no activity")

        total = sum(_values(accumulator, TOTAL_EVENTS)) if TOTAL_EVENTS in available else 0.0
        credited = sum(_values(accumulator, CREDITED_EVENTS)) if CREDITED_EVENTS in available else 0.0
        match_fraction = credited / total if total > 0 else 0.0
        selected_loss = SOURCE_LOSS if run.factors["encoder_credit"] == "source" else ARRIVAL_LOSS
        excluded_loss = ARRIVAL_LOSS if selected_loss == SOURCE_LOSS else SOURCE_LOSS
        if selected_loss not in available or max((abs(v) for v in _values(accumulator, selected_loss)), default=0.0) == 0:
            failures.append("selected encoder-credit branch had zero loss")
        if excluded_loss not in available or max((abs(v) for v in _values(accumulator, excluded_loss)), default=math.inf) > 1e-8:
            failures.append("unselected encoder-credit branch was nonzero")

        recruitment_max = _maximum(accumulator, RECRUITMENT) if RECRUITMENT in available else math.nan
        if retirement == "monitor" and recruitment_max != 0:
            failures.append(f"MON recruitment total is {recruitment_max!r}, expected zero")
        if retirement.startswith("dir") and BAD_ASSIGNMENT in available:
            if _maximum(accumulator, BAD_ASSIGNMENT) > 0 and _maximum(accumulator, FULLY_TESTED) <= 0:
                failures.append("DIR assigned before complete outgoing-pair coverage")
        if retirement.startswith("pred") and PRED_ASSIGNMENT in available:
            if _maximum(accumulator, PRED_ASSIGNMENT) > 0 and _maximum(accumulator, PRED_ELIGIBLE) <= 0:
                failures.append("PRED assigned before persistent evidence was eligible")

        result = {
            "run_name": run_name,
            "condition": run.condition,
            **run.factors,
            "run_dir": str(run_dir),
            "max_env_steps": max_steps,
            "encoder_total_events": total,
            "encoder_credited_events": credited,
            "match_fraction": match_fraction,
            "behavior_replay_mismatch_max": replay_max,
            "recruitment_total_max": recruitment_max,
            "pass": not failures,
            "failures": failures,
        }
        results.append(result)
        all_failures.extend(f"{run_name}: {failure}" for failure in failures)

    pooled_total = sum(float(result.get("encoder_total_events", 0.0)) for result in results)
    pooled_credited = sum(float(result.get("encoder_credited_events", 0.0)) for result in results)
    pooled_match_fraction = pooled_credited / pooled_total if pooled_total > 0 else 0.0
    if pooled_match_fraction < 0.5:
        all_failures.append(
            "pooled within-rollout predecessor match fraction is "
            f"{pooled_match_fraction:.3f}, below 0.5"
        )
    passed = sum(bool(result.get("pass")) for result in results)
    return {
        **study.provenance(),
        "protocol": "source-credit-retirement-preflight-v1",
        "jobs_tsv": str(jobs_tsv),
        "expected_cells": len(expected),
        "passed_cells": passed,
        "all_cells_pass": passed == len(expected) and not all_failures,
        "pooled_encoder_total_events": pooled_total,
        "pooled_encoder_credited_events": pooled_credited,
        "pooled_match_fraction": pooled_match_fraction,
        "source_credit_gradient_test": "test_source_credit_retirement.py",
        "retirement_eligibility_test": "test_graph_stabilized_recruitment.py",
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
