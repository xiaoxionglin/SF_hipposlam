#!/usr/bin/env python3
"""Summarize the latest flat-baseline and persistence-comparison batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

METRICS = {
    "distance": ("train/distance_metric", "intrmotiv/distance/mean"),
    "dg_density": ("train/dg_density", "intrmotiv/dg/density"),
    "dg_silent_fraction": ("train/dg_silent_unit_frac", "intrmotiv/dg/silent_unit_fraction"),
    "dg_multi_fraction": ("train/dg_multi_activation_rate", "intrmotiv/dg/multi_activation_fraction"),
    "intrinsic_mean": ("train/intrinsic_reward_mean", "intrmotiv/reward/intrinsic_mean"),
    "intrinsic_nonzero_fraction": (
        "train/intrinsic_reward_nonzero_frac",
        "intrmotiv/reward/intrinsic_nonzero_fraction",
    ),
    "advantage_mean": ("train/reward_for_advantage_mean", "intrmotiv/reward/advantage_mean"),
    "coverage_auc": (
        "policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc",
        "intrmotiv/exploration/window/coverage_auc",
    ),
    "coverage_unique_cells": (
        "policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_unique_cells",
        "intrmotiv/exploration/window/coverage_unique_cells",
    ),
    "active_target_fraction": ("train/hrl_active_target_frac", "intrmotiv/hrl/active_target_fraction"),
    "target_hit_rate": ("train/hrl_target_hit_rate", "intrmotiv/hrl/target_hit_rate"),
    "tctrl_update_rate": ("train/hrl_tctrl_update_rate", "intrmotiv/hrl/tctrl_update_rate"),
    "option_timeout_rate": ("train/hrl_option_timeout_rate", "intrmotiv/hrl/option_timeout_rate"),
    "known_edge_fraction": ("train/hrl_known_edge_fraction", "intrmotiv/hrl/known_edge_fraction"),
    "node_visit_weight_mean": ("train/hrl_node_visit_weight_mean", "intrmotiv/hrl/node_visit_weight_mean"),
    "window_length_policy_steps": ("intrmotiv/exploration/window/length_policy_steps",),
}


def get_metric(accumulator: EventAccumulator, tags: tuple[str, ...], low: int, high: int) -> tuple[float, str]:
    available = set(accumulator.Tags().get("scalars", []))
    for tag in tags:
        if tag not in available:
            continue
        values = [event.value for event in accumulator.Scalars(tag) if low <= event.step <= high]
        if values:
            return float(np.mean(values)), tag
    return float("nan"), ""


def parse_event(path: Path, batch: str) -> dict[str, object]:
    # The run event files contain many per-update diagnostics. Retaining the
    # most recent samples is sufficient for the terminal analysis window and
    # avoids materializing every tag from long CPU runs.
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 20_000})
    accumulator.Reload()
    scalar_tags = accumulator.Tags().get("scalars", [])
    step_events = accumulator.Scalars("train/env_steps") if "train/env_steps" in scalar_tags else []
    if step_events:
        max_step = max(event.step for event in step_events)
    else:
        tracked = [tag for tags in METRICS.values() for tag in tags if tag in scalar_tags]
        max_step = max((event.step for tag in tracked for event in accumulator.Scalars(tag)), default=0)
    width = min(10_000_000, max(1_000_000, max_step // 5))
    row: dict[str, object] = {
        "batch": batch,
        "run": path.parents[2].name,
        "max_step": max_step,
        "terminal_window_low": max(0, max_step - width),
        "terminal_window_high": max_step,
    }
    for name, tags in METRICS.items():
        value, source_tag = get_metric(accumulator, tags, row["terminal_window_low"], max_step)
        row[name] = value
        row[f"{name}__tag"] = source_tag
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("flat_batch", type=Path)
    parser.add_argument("persistence_batch", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    paths = [("flat_fixed", path) for path in sorted(args.flat_batch.glob("*/*/.summary/0/events.out.tfevents.*"))] + [
        ("persistence", path) for path in sorted(args.persistence_batch.glob("*/*/.summary/0/events.out.tfevents.*"))
    ]
    rows = [parse_event(path, batch) for batch, path in paths]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No TensorBoard event files found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "per_run_terminal.csv", index=False)

    def family(run: str) -> str:
        run = run.removeprefix("00_")
        if run.startswith("GHRL_"):
            return "global_fixed_hrl"
        if run.startswith("LHRL_"):
            return "stream_long_hrl"
        if run.startswith("FLATLONG_"):
            return "flat_long"
        return "flat_fixed"

    frame["family"] = frame.run.map(family)
    numeric = ["max_step", *METRICS]
    aggregate = frame.groupby(["batch", "family"])[numeric].agg(["mean", "std", "count"])
    aggregate.columns = ["__".join(column) for column in aggregate.columns]
    aggregate.reset_index().to_csv(args.output_dir / "family_terminal_summary.csv", index=False)

    summary = {
        "runs": int(len(frame)),
        "by_batch": frame.groupby("batch").size().to_dict(),
        "by_family": frame.groupby("family").size().to_dict(),
        "max_step": {
            family_name: {
                "min": int(group.max_step.min()),
                "median": int(group.max_step.median()),
                "max": int(group.max_step.max()),
            }
            for family_name, group in frame.groupby("family")
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
