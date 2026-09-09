"""Apply implementation-health gates to the 27-cell CPD preflight."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.spec import SpecError


TAGS = {
    "steps": "train/env_steps",
    "replay": "intrmotiv/hrl/behavior_replay_mismatch",
    "forward": "intrmotiv/dg/update_contract/forward_count",
    "ppo_grad": "intrmotiv/dg/gradient/ppo_norm",
    "encoder_grad": "intrmotiv/dg/gradient/encoder_norm",
    "adapter_grad": "intrmotiv/dg/context/adapter_gradient_norm",
    "modulation": "intrmotiv/dg/context/modulation_abs_mean",
    "prediction_loss": "intrmotiv/dg/prediction/loss",
    "prediction_applied": "intrmotiv/dg/prediction/applied_count",
    "prediction_validation": "intrmotiv/dg/prediction/validation_count",
    "prediction_match": "intrmotiv/dg/prediction/replay_match",
    "correct": "intrmotiv/hrl/control/correct_count",
    "wrong": "intrmotiv/hrl/control/wrong_count",
    "timeout": "intrmotiv/hrl/control/timeout_count",
    "command_entropy": "intrmotiv/hrl/control/normalized_command_entropy",
    "pair_coverage": "intrmotiv/hrl/control/observed_pair_coverage",
    "silent_fraction": "intrmotiv/dg/silent_unit_fraction",
    "arrival_loss": "intrmotiv/encoder/credit/arrival_loss",
}


def _values(acc, tag):
    return [float(event.value) for event in acc.Scalars(tag)]


def analyze(study, jobs_tsv: Path, train_root: Path, minimum_steps: int) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    with jobs_tsv.open(newline="", encoding="utf-8") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    expected = {run.name: run for run in study.expand_runs()}
    if len(jobs) != len(expected):
        raise SpecError(f"submission has {len(jobs)} rows; expected {len(expected)}")

    results, failures = [], []
    for job in jobs:
        experiment = job["experiment"]
        run_name = experiment[3:] if experiment.startswith("00_") else experiment
        if run_name not in expected:
            raise SpecError(f"unexpected run {run_name!r}")
        run = expected[run_name]
        summary = train_root / job["train_root"] / experiment / ".summary" / "0"
        run_failures = []
        values = {}
        if not summary.is_dir():
            run_failures.append("missing TensorBoard summary")
        else:
            acc = EventAccumulator(str(summary), size_guidance={"scalars": 100_000})
            acc.Reload()
            available = set(acc.Tags().get("scalars", []))
            missing = [tag for tag in TAGS.values() if tag not in available]
            if missing:
                run_failures.append(f"missing scalar tags: {missing}")
            values = {name: _values(acc, tag) for name, tag in TAGS.items() if tag in available}
            for tag in sorted(tag for tag in available if "loss" in tag.lower()):
                vals = _values(acc, tag)
                if not vals or not all(math.isfinite(v) for v in vals):
                    run_failures.append(f"nonfinite or empty loss scalar {tag}")

        max_steps = max(values.get("steps", [math.nan]))
        if not math.isfinite(max_steps) or max_steps < minimum_steps:
            run_failures.append(f"max env steps {max_steps!r} is below {minimum_steps}")
        if max((abs(v) for v in values.get("replay", [math.inf])), default=math.inf) > 1e-6:
            run_failures.append("behavior replay mismatch is nonzero")
        if any(abs(v - 1.0) > 1e-6 for v in values.get("forward", [math.nan])):
            run_failures.append("recorded DG forward count is not exactly one")
        if max(values.get("ppo_grad", [0.0])) <= 0:
            run_failures.append("JOINT PPO-to-DG gradient never became nonzero")
        if max(values.get("encoder_grad", [0.0])) <= 0:
            run_failures.append("ARR encoder-to-DG gradient never became nonzero")
        if max((abs(v) for v in values.get("arrival_loss", [])), default=0.0) <= 0:
            run_failures.append("ARR credit branch had zero loss")
        for outcome in ("correct", "wrong", "timeout"):
            if sum(values.get(outcome, [])) <= 0:
                run_failures.append(f"FIRST produced no {outcome} events")

        feedback = run.metadata["cell_feedback"]
        predictor = run.metadata["cell_predictor"]
        if feedback == "none":
            if max((abs(v) for v in values.get("adapter_grad", [])), default=0.0) > 1e-10:
                run_failures.append("no-feedback control has a nonzero adapter gradient")
        else:
            if max(values.get("adapter_grad", [0.0])) <= 0:
                run_failures.append("feedback adapter never received a gradient")
            if max(values.get("modulation", [0.0])) <= 0:
                run_failures.append("feedback modulation never left identity")

        if predictor == "none":
            if sum(values.get("prediction_applied", [])) > 1e-8:
                run_failures.append("NONE predictor cell applied prediction events")
        else:
            if sum(values.get("prediction_applied", [])) <= 0:
                run_failures.append("predictor received no within-recurrence events")
            if max(values.get("prediction_loss", [0.0])) <= 0:
                run_failures.append("predictor loss never became positive")
            if sum(values.get("prediction_validation", [])) <= 0:
                run_failures.append("predictor validation split received no events")
            matches = [v for v in values.get("prediction_match", []) if math.isfinite(v)]
            if not matches or max(matches) <= 0:
                run_failures.append("predictor replay match is always zero")

        row = {
            "run_name": run_name,
            **run.factors,
            **run.metadata,
            "max_env_steps": max_steps,
            "pass": not run_failures,
            "failures": run_failures,
        }
        results.append(row)
        failures.extend(f"{run_name}: {item}" for item in run_failures)

    return {
        **study.provenance(),
        "protocol": "ca3-feedback-predictive-dg-preflight-v1",
        "minimum_steps": minimum_steps,
        "expected_cells": len(expected),
        "passed_cells": sum(row["pass"] for row in results),
        "all_cells_pass": not failures,
        "failures": failures,
        "runs": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("jobs_tsv", type=Path)
    parser.add_argument("train_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-steps", type=int, default=5_000_000)
    args = parser.parse_args()
    payload = analyze(load_study(args.study), args.jobs_tsv, args.train_root, args.minimum_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"preflight cells passed: {payload['passed_cells']}/{payload['expected_cells']}")
    if not payload["all_cells_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
