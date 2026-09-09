#!/usr/bin/env python3
"""Terminal-window analysis for the 2026-09-01 corrected-core batch."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CELLS = {
    1: "Flat control",
    2: "Direct delayed",
    3: "Direct immediate",
    4: "Immediate iterative",
    5: "Immediate G+R",
    6: "Immediate G+R+X",
    7: "Immediate O",
    8: "Immediate X+O",
    9: "Immediate HER16",
    10: "Immediate HER64",
    11: "Delayed HER64",
    12: "Delayed X+O long",
    13: "C12 + recovery",
    14: "Topology visit",
    15: "Topology UCB direct",
    16: "Topology UCB waypoint",
}


METRICS = {
    "coverage_auc": "policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc",
    "unique_cells": "policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_unique_cells",
    "coverage_entropy": "policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_entropy",
    "dg_density": "intrmotiv/dg/density",
    "dg_silent_fraction": "intrmotiv/dg/silent_unit_fraction",
    "dg_multi_fraction": "intrmotiv/dg/multi_activation_fraction",
    "dg_usage_entropy": "intrmotiv/dg/usage_entropy",
    "dg_ca3_conflict_fraction": "intrmotiv/dg/ca3_conflict_fraction",
    "intrinsic_reward_mean": "intrmotiv/reward/intrinsic_mean",
    "intrinsic_reward_nonzero_fraction": "intrmotiv/reward/intrinsic_nonzero_fraction",
    "active_target_fraction": "intrmotiv/hrl/active_target_fraction",
    "target_hit_rate": "intrmotiv/hrl/target_hit_rate",
    "target_hit_lift": "intrmotiv/hrl/target_hit_lift",
    "action_sensitivity": "intrmotiv/hrl/goal_condition/action_sensitivity",
    "target_valid_fraction": "intrmotiv/hrl/goal_condition/target_valid_fraction",
    "value_span": "intrmotiv/hrl/goal_condition/value_span",
    "option_success_fraction": "intrmotiv/hrl/option_success_fraction",
    "option_timeout_rate": "intrmotiv/hrl/option_timeout_rate",
    "known_edge_fraction": "intrmotiv/hrl/known_edge_fraction",
    "node_coverage_fraction": "intrmotiv/hrl/node_coverage_fraction",
    "learned_deadline_fraction": "intrmotiv/hrl/learned_deadline_fraction",
    "selected_deadline_mean": "intrmotiv/hrl/selected_deadline_mean",
    "exploration_selection_fraction": "intrmotiv/hrl/exploration/selection_fraction",
    "forced_exploration_fraction": "intrmotiv/hrl/exploration/forced_selection_fraction",
    "exploration_reward_nonzero_fraction": "intrmotiv/hrl/exploration/reward_nonzero_fraction",
    "her_loss": "intrmotiv/her/loss",
    "her_accepted_segments": "intrmotiv/her/accepted_segments_per_rollout",
    "her_segment_length": "intrmotiv/her/segment_length",
    "passive_updates": "intrmotiv/hrl/passive/updates_per_rollout",
    "passive_known_edge_fraction": "intrmotiv/hrl/passive/known_edge_fraction",
    "frontier_selection_rate": "intrmotiv/hrl/frontier/selection_rate",
    "frontier_yield": "intrmotiv/hrl/frontier/yield",
    "frontier_reached_fraction": "intrmotiv/hrl/frontier/reached_fraction",
    "route_available_rate": "intrmotiv/hrl/planning/route_available_rate",
    "hop_count_mean": "intrmotiv/hrl/planning/hop_count_mean",
    "waypoint_success_rate": "intrmotiv/hrl/planning/waypoint_success_rate",
    "final_frontier_reach_rate": "intrmotiv/hrl/planning/final_frontier_reach_rate",
    "validation_success_rate": "intrmotiv/hrl/validation/success_rate",
    "validation_timeout_rate": "intrmotiv/hrl/validation/timeout_rate",
}


CONTRASTS = (
    (2, 3, "immediate minus delayed"),
    (3, 4, "iterative minus simultaneous"),
    (3, 5, "G+R minus immediate control"),
    (3, 6, "G+R+X minus immediate control"),
    (3, 7, "O minus immediate control"),
    (3, 8, "X+O minus immediate control"),
    (3, 9, "HER16 minus immediate control"),
    (3, 10, "HER64 minus immediate control"),
    (2, 11, "delayed HER64 minus delayed control"),
    (12, 13, "recovery minus C12"),
    (14, 15, "UCB direct minus topology visit"),
    (15, 16, "waypoint minus UCB direct"),
)


def scalar_mean(acc: EventAccumulator, tag: str, low: int, high: int) -> tuple[float, int]:
    if tag not in acc.Tags().get("scalars", []):
        return float("nan"), 0
    values = [e.value for e in acc.Scalars(tag) if low <= e.step <= high]
    return (float(np.mean(values)), len(values)) if values else (float("nan"), 0)


def parse_run(event_path: Path, terminal_width: int) -> dict[str, object]:
    run_name = event_path.parents[2].name.removeprefix("00_")
    match = re.search(r"CCR_C(\d{2})_.*_S(8|99|123)$", run_name)
    if match is None:
        raise ValueError(f"Cannot parse corrected-core run name: {run_name}")
    cell, seed = int(match.group(1)), int(match.group(2))
    acc = EventAccumulator(str(event_path), size_guidance={"scalars": 30_000})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if "train/env_steps" in tags:
        max_step = max(e.step for e in acc.Scalars("train/env_steps"))
    else:
        max_step = max((e.step for tag in tags for e in acc.Scalars(tag)), default=0)
    low = max(0, max_step - terminal_width)
    row: dict[str, object] = {
        "cell": cell,
        "condition": f"C{cell:02d}",
        "label": CELLS[cell],
        "seed": seed,
        "run": run_name,
        "max_step": max_step,
        "terminal_low": low,
        "terminal_high": max_step,
    }
    for key, tag in METRICS.items():
        row[key], row[f"{key}__n"] = scalar_mean(acc, tag, low, max_step)
    return row


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = list(METRICS)
    grouped = frame.groupby(["cell", "condition", "label"], sort=True)[metrics]
    mean = grouped.mean().add_suffix("__mean")
    std = grouped.std(ddof=1).add_suffix("__sd")
    count = grouped.count().add_suffix("__n")
    return mean.join(std).join(count).reset_index()


def contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for base, treatment, label in CONTRASTS:
        left = frame[frame.cell == base].set_index("seed")
        right = frame[frame.cell == treatment].set_index("seed")
        shared = sorted(set(left.index) & set(right.index))
        for metric in METRICS:
            values = right.loc[shared, metric] - left.loc[shared, metric]
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "base": f"C{base:02d}",
                    "treatment": f"C{treatment:02d}",
                    "contrast": label,
                    "metric": metric,
                    "mean_difference": finite.mean() if len(finite) else np.nan,
                    "sd_difference": finite.std(ddof=1) if len(finite) > 1 else np.nan,
                    "n": len(finite),
                    **{f"seed_{seed}_difference": values.get(seed, np.nan) for seed in (8, 99, 123)},
                }
            )
    return pd.DataFrame(rows)


def metric_panel(summary: pd.DataFrame, metrics: list[tuple[str, str]], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3.2 * len(metrics)), sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(summary))
    labels = [f"C{int(c):02d}" for c in summary.cell]
    for ax, (metric, title) in zip(axes, metrics):
        mean = summary[f"{metric}__mean"].to_numpy(float)
        sd = summary[f"{metric}__sd"].to_numpy(float)
        ax.errorbar(x, mean, yerr=sd, fmt="o", capsize=3, color="#315b7d")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x, labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--terminal-width", type=int, default=10_000_000)
    args = parser.parse_args()

    event_paths = sorted(args.batch_root.glob("*/*/.summary/0/events.out.tfevents.*"))
    if len(event_paths) != 48:
        raise SystemExit(f"Expected 48 event files, found {len(event_paths)}")
    rows = [parse_run(path, args.terminal_width) for path in event_paths]
    frame = pd.DataFrame(rows).sort_values(["cell", "seed"])
    if frame.groupby("cell").size().to_dict() != {cell: 3 for cell in CELLS}:
        raise SystemExit("Corrected-core cell/seed matrix is incomplete")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = summarize(frame)
    paired = contrasts(frame)
    frame.to_csv(args.output_dir / "per_run_terminal_10m.csv", index=False)
    aggregate.to_csv(args.output_dir / "cell_terminal_10m.csv", index=False)
    paired.to_csv(args.output_dir / "paired_contrasts_terminal_10m.csv", index=False)

    metric_panel(
        aggregate,
        [("coverage_auc", "Coverage AUC"), ("unique_cells", "Unique cells")],
        args.output_dir / "coverage_terminal_10m.png",
    )
    metric_panel(
        aggregate,
        [
            ("target_hit_rate", "Target-hit rate"),
            ("target_hit_lift", "Target-hit lift"),
            ("action_sensitivity", "Action sensitivity"),
            ("option_success_fraction", "Option success fraction"),
        ],
        args.output_dir / "target_control_terminal_10m.png",
    )
    metric_panel(
        aggregate,
        [
            ("dg_density", "DG density"),
            ("dg_silent_fraction", "Silent-unit fraction"),
            ("dg_usage_entropy", "DG usage entropy"),
            ("dg_ca3_conflict_fraction", "CA3 conflict fraction"),
        ],
        args.output_dir / "dg_health_terminal_10m.png",
    )

    summary = {
        "runs": len(frame),
        "cells": len(aggregate),
        "seeds": sorted(frame.seed.unique().tolist()),
        "min_max_step": int(frame.max_step.min()),
        "max_max_step": int(frame.max_step.max()),
        "terminal_width": args.terminal_width,
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
