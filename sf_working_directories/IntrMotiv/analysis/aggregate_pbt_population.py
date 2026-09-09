#!/usr/bin/env python3
"""Average Sample Factory policy streams as one PBT population."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAGS = (
    "train/intrinsic_reward_mean",
    "train/reward_for_advantage_mean",
    "train/hrl_active_target_frac",
    "train/hrl_source_frac",
    "train/hrl_target_hit_rate",
    "train/hrl_tctrl_update_rate",
    "train/hrl_option_reset_rate",
    "train/hrl_option_timeout_rate",
    "train/hrl_node_coverage_fraction",
    "train/dg_density",
    "train/dg_multi_activation_rate",
    "train/dg_silent_unit_frac",
    "train/encoder_loss",
    "train/decoder_loss",
    "train/iterative_phase",
)


def read_policy_events(path: Path) -> dict[str, list[tuple[int, float]]]:
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags().get("scalars", [])
    }


def latest_before(events: list[tuple[int, float]], step: int) -> float | None:
    value = None
    for event_step, event_value in events:
        if event_step > step:
            break
        value = event_value
    return value


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_policy_streams(batch_dir: Path) -> dict[str, dict[str, dict[str, list[tuple[int, float]]]]]:
    streams: dict[str, dict[str, dict[str, list[tuple[int, float]]]]] = defaultdict(dict)
    for event_path in sorted(batch_dir.glob("**/.summary/*/events.out.tfevents.*")):
        summary_index = event_path.parts.index(".summary")
        run_name = event_path.parts[summary_index - 1]
        policy_id = event_path.parts[summary_index + 1]
        streams[run_name][policy_id] = read_policy_events(event_path)
    return streams


def population_rows(
    streams: dict[str, dict[str, dict[str, list[tuple[int, float]]]]], tags: tuple[str, ...], step_interval: int
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    latest_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    run_summary = []

    for run_name, policies in sorted(streams.items()):
        max_steps = {
            policy: max((step for events in values.values() for step, _ in events), default=0)
            for policy, values in policies.items()
        }
        common_step = min(max_steps.values(), default=0)
        for tag in tags:
            values = [
                latest_before(policies[policy].get(tag, []), common_step)
                for policy in sorted(policies)
            ]
            values = [value for value in values if value is not None and math.isfinite(value)]
            if values:
                latest_rows.append(
                    {
                        "run": run_name,
                        "common_step": common_step,
                        "tag": tag,
                        "mean": mean(values),
                        "std": pstdev(values) if len(values) > 1 else 0.0,
                        "n_policies": len(values),
                    }
                )

        for step in range(step_interval, common_step + 1, step_interval):
            for tag in tags:
                values = [
                    latest_before(policies[policy].get(tag, []), step)
                    for policy in sorted(policies)
                ]
                values = [value for value in values if value is not None and math.isfinite(value)]
                if values:
                    curve_rows.append(
                        {
                            "run": run_name,
                            "step": step,
                            "tag": tag,
                            "mean": mean(values),
                            "std": pstdev(values) if len(values) > 1 else 0.0,
                            "n_policies": len(values),
                        }
                    )

        run_summary.append(
            {
                "run": run_name,
                "policy_count": len(policies),
                "policy_ids": sorted(policies),
                "common_step": common_step,
                "min_policy_step": min(max_steps.values(), default=0),
                "max_policy_step": max(max_steps.values(), default=0),
            }
        )

    summary = {
        "runs": len(run_summary),
        "runs_with_four_policies": sum(row["policy_count"] == 4 for row in run_summary),
        "run_summary": run_summary,
    }
    return latest_rows, curve_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--step-interval", type=int, default=1_000_000)
    parser.add_argument("--tag", action="append", dest="tags")
    args = parser.parse_args()
    if args.step_interval <= 0:
        raise SystemExit("--step-interval must be positive")

    tags = tuple(args.tags or DEFAULT_TAGS)
    streams = collect_policy_streams(args.batch_dir)
    if not streams:
        raise SystemExit(f"No policy event files found under {args.batch_dir}")
    latest, curves, summary = population_rows(streams, tags, args.step_interval)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "population_latest.csv", latest)
    write_rows(args.output_dir / "population_curves.csv", curves)
    (args.output_dir / "population_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "run_summary"}, indent=2))


if __name__ == "__main__":
    main()
