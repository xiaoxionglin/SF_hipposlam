#!/usr/bin/env python3
"""Aggregate TensorBoard scalars from the first controllable-graph HRL batch."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUN_RE = re.compile(
    r"B1_F(?P<F>\d+)_L(?P<L>\d+)_T(?P<threshold>\d+)_ER"
    r"(?P<reward_method>punish|encourage|mean)_S(?P<seed>\d+)"
)

METRICS = [
    "perf/_fps",
    "perf/_sample_throughput",
    "train/entropy",
    "train/version_diff_avg",
    "train/distance_metric",
    "train/reward_for_advantage_mean",
    "train/reward_for_advantage_nonzero_frac",
    "train/intrinsic_reward_mean",
    "train/intrinsic_reward_nonzero_frac",
    "train/hrl_active_target_frac",
    "train/hrl_source_frac",
    "train/hrl_target_hit_rate",
    "train/hrl_tctrl_update_rate",
    "train/hrl_option_reset_rate",
    "train/hrl_option_timeout_rate",
    "train/hrl_option_success_fraction",
    "train/hrl_learned_deadline_fraction",
    "train/hrl_selected_deadline_mean",
    "train/hrl_elapsed_on_hit_mean",
    "train/hrl_elapsed_on_timeout_mean",
    "train/hrl_node_coverage_fraction",
    "train/hrl_selected_target_visit_mean",
    "train/hrl_known_edge_fraction",
    "train/hrl_known_controllability_time_mean",
    "train/dg_density",
    "train/dg_multi_activation_rate",
    "train/dg_silent_unit_frac",
    "train/encoder_loss",
    "train/decoder_loss",
    "train/batch_reward_loss",
    "train/encoder_punishment",
    "len/len",
    "len/len_min",
    "len/len_max",
]

WINDOWS = {
    "early_5_15m": (5_000_000, 15_000_000),
    "common_50_60m": (50_000_000, 60_000_000),
    "mature_80_90m": (80_000_000, 90_000_000),
}


def scalar_mean(accumulator: EventAccumulator, tag: str, low: int, high: int) -> tuple[float, int]:
    if tag not in accumulator.Tags().get("scalars", []):
        return np.nan, 0
    values = [event.value for event in accumulator.Scalars(tag) if low <= event.step <= high]
    if not values:
        return np.nan, 0
    return float(np.mean(values)), len(values)


def parse_run(event_path: Path) -> list[dict[str, object]]:
    match = RUN_RE.search(str(event_path))
    if match is None:
        return []

    metadata: dict[str, object] = match.groupdict()
    metadata["F"] = int(metadata["F"])
    metadata["L"] = int(metadata["L"])
    metadata["threshold"] = int(metadata["threshold"]) / 100.0
    metadata["seed"] = int(metadata["seed"])
    metadata["run"] = match.group(0)

    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    all_events = [event for tag in accumulator.Tags().get("scalars", []) for event in accumulator.Scalars(tag)]
    max_step = max((event.step for event in all_events), default=0)

    rows = []
    for window_name, (low, high) in WINDOWS.items():
        row = dict(metadata)
        row.update(window=window_name, window_low=low, window_high=high, max_step=max_step)
        for tag in METRICS:
            value, count = scalar_mean(accumulator, tag, low, high)
            row[tag] = value
            row[f"{tag}__n"] = count
        rows.append(row)

    # A per-run terminal window is useful for diagnosing jobs that have not reached 80M.
    low = max(0, max_step - 10_000_000)
    row = dict(metadata)
    row.update(window="terminal_10m", window_low=low, window_high=max_step, max_step=max_step)
    for tag in METRICS:
        value, count = scalar_mean(accumulator, tag, low, max_step)
        row[tag] = value
        row[f"{tag}__n"] = count
    rows.append(row)
    return rows


def aggregate(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metric_columns = [metric for metric in METRICS if metric in frame.columns]
    grouped = frame.groupby(group_columns, dropna=False)[metric_columns]
    means = grouped.mean().add_suffix("__mean")
    stds = grouped.std().add_suffix("__std")
    counts = grouped.count().add_suffix("__count")
    return means.join(stds).join(counts).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    event_paths = sorted(args.batch_dir.glob("*/*/.summary/0/events.out.tfevents.*"))
    rows = [row for path in event_paths for row in parse_run(path)]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"No matching runs found under {args.batch_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "per_run_windows.csv", index=False)

    aggregate(frame, ["window"]).to_csv(args.output_dir / "aggregate_overall.csv", index=False)
    aggregate(frame, ["window", "reward_method"]).to_csv(
        args.output_dir / "aggregate_reward_method.csv", index=False
    )
    aggregate(frame, ["window", "L"]).to_csv(args.output_dir / "aggregate_sequence_length.csv", index=False)
    aggregate(frame, ["window", "threshold"]).to_csv(
        args.output_dir / "aggregate_threshold.csv", index=False
    )
    aggregate(frame, ["window", "L", "threshold", "reward_method"]).to_csv(
        args.output_dir / "aggregate_cells.csv", index=False
    )

    terminal = frame[frame.window == "terminal_10m"]
    summary = {
        "event_files": len(event_paths),
        "runs": int(terminal.run.nunique()),
        "max_step_min": int(terminal.max_step.min()),
        "max_step_median": int(terminal.max_step.median()),
        "max_step_max": int(terminal.max_step.max()),
        "runs_at_80m": int((terminal.max_step >= 80_000_000).sum()),
        "runs_at_90m": int((terminal.max_step >= 90_000_000).sum()),
    }
    (args.output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
