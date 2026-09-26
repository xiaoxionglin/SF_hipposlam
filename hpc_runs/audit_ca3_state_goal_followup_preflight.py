"""Mechanical release gate for the four-cell CA3 state-goal follow-up."""

import argparse
import csv
import json
import math
from pathlib import Path

from hpc_runs.audit_ca3_predictive_active_goals_preflight import audit as audit_parent
from hpc_runs.audit_navigation8_algorithm_screen_preflight import _events

FINITE_TAGS = (
    "intrmotiv/ca3_readout/total_loss",
    "intrmotiv/ca3_readout/var_loss",
    "intrmotiv/ca3_readout/cov_loss",
    "intrmotiv/ca3_readout/latent_std_mean",
    "intrmotiv/ca3_readout/latent_std_min",
    "intrmotiv/controller/positive_similarity_q10",
    "intrmotiv/controller/positive_similarity_q50",
    "intrmotiv/controller/positive_similarity_q90",
    "intrmotiv/controller/background_similarity_q50",
    "intrmotiv/controller/background_similarity_q90",
    "intrmotiv/controller/background_similarity_q99",
    "intrmotiv/controller/background_above_threshold_fraction",
    "intrmotiv/controller/active_anchor_collision_fraction",
    "intrmotiv/controller/context_raw_multi_activation",
    "intrmotiv/controller/context_accepted_events",
    "intrmotiv/controller/context_zero_match",
    "intrmotiv/controller/context_multi_match",
    "intrmotiv/controller/her_contextual_candidates",
    "intrmotiv/controller/her_contextual_positive_hits",
    "intrmotiv/controller/her_contextual_wrong_context",
)


def _latest(events, tag):
    values = events.get(tag, ())
    if not values:
        return None
    return float(max(values, key=lambda event: (event.step, event.wall_time)).value)


def audit(study, jobs, train_root, reload_certificate_root, telemetry_root):
    # This follow-up intentionally introduces no terminal raw-CA3 transport.
    # Contextual terminal HER targets are rejected, while ordinary terminal
    # presence and every other stored-replay invariant remain audited.
    result = audit_parent(
        study,
        jobs,
        train_root,
        reload_certificate_root,
        telemetry_root,
        require_certified_terminal_successor=False,
    )
    rows = {row["run"]: row for row in result["runs"]}
    for job in csv.DictReader(Path(jobs).open(), delimiter="\t"):
        run = job["experiment"].removeprefix("00_")
        row = rows[run]
        directory = Path(train_root) / job["train_root"] / job["experiment"]
        cfg = json.loads((directory / "config.json").read_text())
        events = _events(directory)
        metrics = {}
        for tag in FINITE_TAGS:
            value = _latest(events, tag)
            metrics[tag] = value
            if value is None or not math.isfinite(value):
                row["errors"].append("missing or nonfinite follow-up scalar " + tag)
        if (metrics["intrmotiv/controller/her_contextual_candidates"] or 0) <= 0:
            row["errors"].append("contextual HER candidate path was not exercised")
        if (metrics["intrmotiv/controller/her_contextual_positive_hits"] or 0) <= 0:
            row["errors"].append("contextual HER positive-hit path was not exercised")
        if (metrics["intrmotiv/controller/her_contextual_wrong_context"] or 0) <= 0:
            row["errors"].append("contextual HER wrong-context path was not exercised")
        if (metrics["intrmotiv/controller/context_raw_multi_activation"] or 0) <= 0:
            row["errors"].append("multi-activation recognition path was not exercised")
        if cfg.get("ca3_graph_anchor_mode") == "ema":
            tag = "intrmotiv/controller/anchor_refinement_attempts"
            value = _latest(events, tag)
            metrics[tag] = value
            if value is None or value <= 0:
                row["errors"].append("EMA comparison path was not exercised")
        row["followup_metrics"] = metrics
        row["passed"] = not row["errors"]
    result["passed"] = all(row["passed"] for row in result["runs"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    parser.add_argument("jobs")
    parser.add_argument("train_root")
    parser.add_argument("--reload-certificate-root", required=True)
    parser.add_argument("--telemetry-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(
        args.study,
        args.jobs,
        args.train_root,
        args.reload_certificate_root,
        args.telemetry_root,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["passed"] else 1)
