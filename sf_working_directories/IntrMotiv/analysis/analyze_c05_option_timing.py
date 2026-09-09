#!/usr/bin/env python3
"""Lightweight TensorBoard analysis of C05 option timing versus C03."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRICS = {
    "target_hit_rate": "intrmotiv/hrl/target_hit_rate",
    "timeout_rate": "intrmotiv/hrl/option_timeout_rate",
    "option_success_fraction": "intrmotiv/hrl/option_success_fraction",
    "elapsed_on_hit": "intrmotiv/hrl/elapsed_on_hit_mean",
    "elapsed_on_timeout": "intrmotiv/hrl/elapsed_on_timeout_mean",
    "target_selected_deadline": "intrmotiv/hrl/target_selected_deadline_mean",
    "selected_deadline_positive": "intrmotiv/hrl/selected_deadline_positive_mean",
    "learned_deadline_fraction": "intrmotiv/hrl/learned_deadline_fraction",
    "known_controllability_time": "intrmotiv/hrl/known_controllability_time_mean",
    "known_edge_fraction": "intrmotiv/hrl/known_edge_fraction",
    "target_hit_lift": "intrmotiv/hrl/target_hit_lift",
    "action_sensitivity": "intrmotiv/hrl/goal_condition/action_sensitivity",
    "target_valid_fraction": "intrmotiv/hrl/goal_condition/target_valid_fraction",
    "selected_target_visits": "intrmotiv/hrl/selected_target_visit_mean",
}


def mean_in_window(acc: EventAccumulator, tag: str, low: int, high: int) -> tuple[float, int]:
    if tag not in acc.Tags().get("scalars", []):
        return float("nan"), 0
    values = [event.value for event in acc.Scalars(tag) if low <= event.step <= high]
    return (float(np.mean(values)), len(values)) if values else (float("nan"), 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bin-width", type=int, default=10_000_000)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    pattern = re.compile(r"00_CCR_C(?P<cell>03|05)_.*_S(?P<seed>8|99|123)$")
    paths = sorted(args.batch_root.glob("*/*/.summary/0/events.out.tfevents.*"))
    for path in paths:
        run = path.parents[2].name
        match = pattern.match(run)
        if match is None:
            continue
        cell, seed = int(match.group("cell")), int(match.group("seed"))
        acc = EventAccumulator(str(path), size_guidance={"scalars": 30_000})
        acc.Reload()
        max_step = max(event.step for event in acc.Scalars("train/env_steps"))
        lows = list(range(0, max_step, args.bin_width))
        if len(lows) > 1 and max_step - lows[-1] < args.bin_width // 2:
            lows.pop()
        for index, low in enumerate(lows):
            high = lows[index + 1] if index + 1 < len(lows) else max_step
            row: dict[str, object] = {
                "condition": f"C{cell:02d}",
                "seed": seed,
                "window_low": low,
                "window_high": high,
                "terminal": int(high == max_step),
            }
            for name, tag in METRICS.items():
                row[name], row[f"{name}__n"] = mean_in_window(acc, tag, low, high)
            rows.append(row)

    if len({(row["condition"], row["seed"]) for row in rows}) != 6:
        raise SystemExit("Expected three C03 and three C05 runs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} ten-million-frame window rows to {args.output}")


if __name__ == "__main__":
    main()
