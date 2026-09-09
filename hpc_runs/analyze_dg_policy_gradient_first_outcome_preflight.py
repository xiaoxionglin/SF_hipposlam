"""Apply the declared runtime gates to the eight-run DGP preflight."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.spec import SpecError


TAGS = {
    "steps": "train/env_steps",
    "replay": "intrmotiv/hrl/behavior_replay_mismatch",
    "forward": "intrmotiv/dg/update_contract/forward_count",
    "ppo_grad": "intrmotiv/dg/gradient/ppo_norm",
    "encoder_grad": "intrmotiv/dg/gradient/encoder_norm",
    "correct": "intrmotiv/hrl/control/correct_count",
    "wrong": "intrmotiv/hrl/control/wrong_count",
    "timeout": "intrmotiv/hrl/control/timeout_count",
    "command_entropy": "intrmotiv/hrl/control/normalized_command_entropy",
    "pair_coverage": "intrmotiv/hrl/control/observed_pair_coverage",
    "silent_fraction": "intrmotiv/dg/silent_unit_fraction",
    "node_coverage": "intrmotiv/hrl/node_coverage_fraction",
    "recruitment": "intrmotiv/dg/recruitment/total",
    "arrival_loss": "intrmotiv/encoder/credit/arrival_loss",
}


def _values(acc, tag):
    return [float(event.value) for event in acc.Scalars(tag)]


def _tail(values, n=10):
    return values[-min(n, len(values)) :]


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
        if not summary.is_dir():
            run_failures.append("missing TensorBoard summary")
            values = {}
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

        ppo = values.get("ppo_grad", [])
        if run.factors["ppo_dg_gradient"] == "stop":
            if not ppo or max(abs(v) for v in ppo) > 1e-10:
                run_failures.append("STOP PPO-to-DG gradient is nonzero")
        elif not ppo or max(ppo) <= 0:
            run_failures.append("JOINT PPO-to-DG gradient never became nonzero")
        if max(values.get("encoder_grad", [0.0])) <= 0:
            run_failures.append("ARR encoder-to-DG gradient never became nonzero")

        if run.factors["worker_outcome"] == "first_distinct":
            for event in ("correct", "wrong", "timeout"):
                if sum(values.get(event, [])) <= 0:
                    run_failures.append(f"FIRST produced no {event} events")
        entropy = fmean(_tail(values.get("command_entropy", [0.0])))
        coverage = fmean(_tail(values.get("pair_coverage", [0.0])))
        silent_units = 16.0 * fmean(_tail(values.get("silent_fraction", [1.0])))
        node_coverage = values.get("node_coverage", [0.0])[-1]
        if entropy < 0.9:
            run_failures.append(f"terminal normalized command entropy {entropy:.3f} is below 0.9")
        if coverage < 0.8:
            run_failures.append(f"terminal observed-pair coverage {coverage:.3f} is below 0.8")
        if silent_units > 1.0 + 1e-6 or node_coverage < 1.0 - 1e-6:
            run_failures.append(
                f"DG activity gate failed: mean silent units={silent_units:.3f}, node coverage={node_coverage:.3f}"
            )
        recruitment = values.get("recruitment", [math.nan])[-1]
        if not math.isfinite(recruitment) or abs(recruitment) > 1e-6:
            run_failures.append(f"MON recruitment total is {recruitment!r}, expected zero")
        if max((abs(v) for v in values.get("arrival_loss", [])), default=0.0) <= 0:
            run_failures.append("ARR credit branch had zero loss")

        row = {"run_name": run_name, **run.factors, "max_env_steps": max_steps,
               "terminal_command_entropy": entropy, "terminal_pair_coverage": coverage,
               "terminal_mean_silent_units": silent_units, "terminal_node_coverage": node_coverage,
               "pass": not run_failures, "failures": run_failures}
        results.append(row)
        failures.extend(f"{run_name}: {item}" for item in run_failures)

    return {**study.provenance(), "protocol": "dg-policy-gradient-first-outcome-preflight-v1",
            "minimum_steps": minimum_steps, "expected_cells": len(expected),
            "passed_cells": sum(row["pass"] for row in results),
            "all_cells_pass": not failures, "static_contract_tests_required": [
                "test_update_contract.py", "test_hrl_controllable_graph.py", "test_topological_frontier.py"],
            "failures": failures, "runs": results}


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
