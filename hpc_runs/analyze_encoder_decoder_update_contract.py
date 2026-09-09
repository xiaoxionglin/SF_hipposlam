"""Audit runtime invariants for the DG-controller update-contract batch."""

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
RECRUITMENT = "intrmotiv/dg/recruitment/total"
RESET_TOTAL = "intrmotiv/dg/recruitment/goal_adapter_reset_total"
SCHEDULED_COUNT = "intrmotiv/encoder/credit/scheduled_count"
APPLIED_COUNT = "intrmotiv/encoder/credit/applied_count"
SCHEDULED_MASS = "intrmotiv/encoder/credit/scheduled_reward_mass"
APPLIED_MASS = "intrmotiv/encoder/credit/applied_reward_mass"
REPLAY_MATCH = "intrmotiv/encoder/credit/replay_match"
DG_FORWARD = "intrmotiv/dg/update_contract/forward_count"
DG_STATS_UPDATE = "intrmotiv/dg/update_contract/running_stats_update_count"
ARRIVAL_LOSS = "intrmotiv/encoder/credit/arrival_loss"
SOURCE_LOSS = "intrmotiv/encoder/credit/source_loss"


def _values(accumulator, tag: str) -> list[float]:
    return [float(event.value) for event in accumulator.Scalars(tag)]


def _maximum(accumulator, tag: str) -> float:
    return max(_values(accumulator, tag), default=math.nan)


def analyze(study, jobs_tsv: Path, train_root: Path, minimum_steps: int) -> dict:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as error:
        raise RuntimeError("TensorBoard is required for contract analysis") from error

    with jobs_tsv.open(newline="", encoding="utf-8") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    expected = {run.name: run for run in study.expand_runs()}
    if len(jobs) != len(expected):
        raise SpecError(f"submission has {len(jobs)} rows; expected {len(expected)}")

    results = []
    failures = []
    for job in jobs:
        experiment = job["experiment"]
        if not experiment.startswith("00_"):
            raise SpecError(f"unexpected experiment {experiment!r}")
        run_name = experiment[3:]
        run = expected[run_name]
        run_dir = train_root / job["train_root"] / experiment
        summary_dir = run_dir / ".summary" / "0"
        run_failures = []
        if not summary_dir.is_dir():
            run_failures.append("missing TensorBoard summary")
            results.append({"run_name": run_name, "pass": False, "failures": run_failures})
            failures.extend(f"{run_name}: {failure}" for failure in run_failures)
            continue

        accumulator = EventAccumulator(str(summary_dir), size_guidance={"scalars": 100_000})
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        required = {
            STEP, REPLAY, TARGET_VALID, FRONTIER, PRED_EVENTS, RECRUITMENT,
            RESET_TOTAL, SCHEDULED_COUNT, APPLIED_COUNT, SCHEDULED_MASS,
            APPLIED_MASS, REPLAY_MATCH, DG_FORWARD, DG_STATS_UPDATE,
            ARRIVAL_LOSS, SOURCE_LOSS,
        }
        missing = sorted(required - available)
        if missing:
            run_failures.append(f"missing scalar tags: {missing}")

        max_steps = _maximum(accumulator, STEP) if STEP in available else math.nan
        if not math.isfinite(max_steps) or max_steps < minimum_steps:
            run_failures.append(f"max env steps {max_steps!r} is below {minimum_steps}")
        for tag in sorted(tag for tag in available if "loss" in tag.lower()):
            values = _values(accumulator, tag)
            if not values or not all(math.isfinite(value) for value in values):
                run_failures.append(f"nonfinite or empty loss scalar {tag}")

        replay_max = (
            max((abs(value) for value in _values(accumulator, REPLAY)), default=math.nan)
            if REPLAY in available else math.nan
        )
        if not math.isfinite(replay_max) or replay_max > 1e-6:
            run_failures.append(f"behavior replay mismatch max is {replay_max!r}")
        for tag, label in (
            (TARGET_VALID, "goal target"),
            (FRONTIER, "frontier manager"),
            (PRED_EVENTS, "PRED evidence"),
        ):
            if tag not in available or _maximum(accumulator, tag) <= 0:
                run_failures.append(f"{label} had no activity")

        for tag, label in ((DG_FORWARD, "DG forwards"), (DG_STATS_UPDATE, "DG stat updates")):
            values = _values(accumulator, tag) if tag in available else []
            if not values or any(abs(value - 1.0) > 1e-6 for value in values):
                run_failures.append(f"{label} were not exactly one on every recorded minibatch")

        scheduled = sum(_values(accumulator, SCHEDULED_COUNT)) if SCHEDULED_COUNT in available else 0.0
        applied = sum(_values(accumulator, APPLIED_COUNT)) if APPLIED_COUNT in available else 0.0
        match = applied / scheduled if scheduled > 0 else 0.0
        if match < 0.90:
            run_failures.append(f"credited-row replay match {match:.3f} is below 0.90")
        scheduled_mass = sum(_values(accumulator, SCHEDULED_MASS)) if SCHEDULED_MASS in available else 0.0
        applied_mass = sum(_values(accumulator, APPLIED_MASS)) if APPLIED_MASS in available else 0.0
        if scheduled_mass <= 0 or applied_mass <= 0:
            run_failures.append("scheduled or applied encoder reward mass was zero")

        selected = SOURCE_LOSS if run.factors["encoder_credit"] == "source" else ARRIVAL_LOSS
        excluded = ARRIVAL_LOSS if selected == SOURCE_LOSS else SOURCE_LOSS
        if selected not in available or max((abs(v) for v in _values(accumulator, selected)), default=0.0) == 0:
            run_failures.append("selected encoder-credit branch had zero loss")
        if excluded not in available or max((abs(v) for v in _values(accumulator, excluded)), default=math.inf) > 1e-8:
            run_failures.append("unselected encoder-credit branch was nonzero")

        recruitment = _values(accumulator, RECRUITMENT) if RECRUITMENT in available else []
        resets = _values(accumulator, RESET_TOTAL) if RESET_TOTAL in available else []
        recruitment_final = recruitment[-1] if recruitment else math.nan
        reset_final = resets[-1] if resets else math.nan
        if not math.isfinite(recruitment_final) or abs(recruitment_final - reset_final) > 1e-6:
            run_failures.append(
                f"recruitment total {recruitment_final!r} != FiLM reset total {reset_final!r}"
            )
        if run.factors["retirement"] == "monitor" and recruitment_final != 0:
            run_failures.append(f"MON recruitment total is {recruitment_final!r}, expected zero")

        result = {
            "run_name": run_name,
            "condition": run.condition,
            **run.factors,
            "run_dir": str(run_dir),
            "max_env_steps": max_steps,
            "scheduled_credit_count": scheduled,
            "applied_credit_count": applied,
            "credited_row_replay_match": match,
            "scheduled_reward_mass": scheduled_mass,
            "applied_reward_mass": applied_mass,
            "recruitment_total": recruitment_final,
            "goal_adapter_reset_total": reset_final,
            "pass": not run_failures,
            "failures": run_failures,
        }
        results.append(result)
        failures.extend(f"{run_name}: {failure}" for failure in run_failures)

    pooled_scheduled = sum(float(row.get("scheduled_credit_count", 0.0)) for row in results)
    pooled_applied = sum(float(row.get("applied_credit_count", 0.0)) for row in results)
    pooled_match = pooled_applied / pooled_scheduled if pooled_scheduled > 0 else 0.0
    if pooled_match < 0.95:
        failures.append(f"pooled credited-row replay match {pooled_match:.3f} is below 0.95")
    passed = sum(bool(row.get("pass")) for row in results)
    return {
        **study.provenance(),
        "protocol": "encoder-decoder-update-contract-v1",
        "minimum_steps": minimum_steps,
        "expected_cells": len(expected),
        "passed_cells": passed,
        "all_cells_pass": passed == len(expected) and not failures,
        "pooled_scheduled_credit_count": pooled_scheduled,
        "pooled_applied_credit_count": pooled_applied,
        "pooled_credited_row_replay_match": pooled_match,
        "static_contract_test": "sf_working_directories/IntrMotiv/tests/test_update_contract.py",
        "failures": failures,
        "runs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("jobs_tsv", type=Path)
    parser.add_argument("train_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-steps", type=int, default=5_000_000)
    args = parser.parse_args()
    try:
        payload = analyze(load_study(args.study), args.jobs_tsv, args.train_root, args.minimum_steps)
    except (OSError, RuntimeError, SpecError, ValueError, KeyError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"contract cells passed: {payload['passed_cells']}/{payload['expected_cells']}; report: {args.output}")
    if not payload["all_cells_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
