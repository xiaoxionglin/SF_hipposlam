#!/usr/bin/env python3
"""Analyze live or completed Graph-Stabilized Recruitment TensorBoard runs.

The batch is a balanced 3 (backbone) x 2 (redundancy threshold) x 2
(half-life) x 3 (seed) design.  Ordinary state/rate metrics are averaged over
each run's latest window.  Cumulative recruitment counters are sampled at the
latest available event instead of averaged, because their terminal value is
the scientifically meaningful quantity.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUN_RE = re.compile(r"GSR_(C05|C13|C15)_D(4|8)_H(5|10)K_S(8|99|123)$")
BACKBONES = ("C05", "C13", "C15")
SEEDS = (8, 99, 123)
PARAMETER_CELLS = ((4, 5), (4, 10), (8, 5), (8, 10))

WINDOW_METRICS = {
    "coverage_auc": "policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_auc",
    "unique_cells": "policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_unique_cells",
    "coverage_entropy": "policy_stats/avg_z_00_openfield_map2_fixed_loc3_fixedlength_noreward_coverage_entropy",
    "dg_density": "intrmotiv/dg/density",
    "dg_silent_fraction": "intrmotiv/dg/silent_unit_fraction",
    "dg_usage_entropy": "intrmotiv/dg/usage_entropy",
    "dg_behavior_event_fraction": "intrmotiv/dg/behavior_dominant_event_fraction",
    "target_hit_lift": "intrmotiv/hrl/target_hit_lift",
    "action_sensitivity": "intrmotiv/hrl/goal_condition/action_sensitivity",
    "option_success_fraction": "intrmotiv/hrl/option_success_fraction",
    "known_edge_fraction": "intrmotiv/hrl/known_edge_fraction",
    "node_coverage_fraction": "intrmotiv/hrl/node_coverage_fraction",
    "connected_fraction": "intrmotiv/dg/recruitment/connected_fraction",
    "isolated_fraction": "intrmotiv/dg/recruitment/isolated_fraction",
    "redundant_pair_count": "intrmotiv/dg/recruitment/redundant_pair_count",
    "eligible_vertex_count": "intrmotiv/dg/recruitment/eligible_vertex_count",
    "birth_protected_count": "intrmotiv/dg/recruitment/birth_protected_count",
    "isolated_assignments_per_rollout": "intrmotiv/dg/recruitment/isolated_assignments_per_rollout",
    "redundant_assignments_per_rollout": "intrmotiv/dg/recruitment/redundant_assignments_per_rollout",
    "repeat_assignments_per_rollout": "intrmotiv/dg/recruitment/repeat_assignments_per_rollout",
    "passive_graph_density": "intrmotiv/dg/recruitment/passive_graph_density",
    "passive_updates_per_rollout": "intrmotiv/dg/recruitment/passive_updates_per_rollout",
    "passive_stale_per_rollout": "intrmotiv/dg/recruitment/passive_stale_per_rollout",
    "passive_over_gap_per_rollout": "intrmotiv/dg/recruitment/passive_over_gap_per_rollout",
    "exploration_selection_fraction": "intrmotiv/hrl/exploration/selection_fraction",
    "forced_exploration_fraction": "intrmotiv/hrl/exploration/forced_selection_fraction",
    "frontier_selection_rate": "intrmotiv/hrl/frontier/selection_rate",
    "frontier_yield": "intrmotiv/hrl/frontier/yield",
    "frontier_reached_fraction": "intrmotiv/hrl/frontier/reached_fraction",
}

CUMULATIVE_METRICS = {
    "recruitment_total": "intrmotiv/dg/recruitment/total",
    "repeat_total": "intrmotiv/dg/recruitment/repeat_total",
    "tiny_residual_total": "intrmotiv/dg/recruitment/tiny_residual_total",
}


def mean_in_window(acc: EventAccumulator, tag: str, low: int, high: int) -> tuple[float, int]:
    if tag not in acc.Tags().get("scalars", []):
        return np.nan, 0
    values = [event.value for event in acc.Scalars(tag) if low <= event.step <= high]
    return (float(np.mean(values)), len(values)) if values else (np.nan, 0)


def latest_at_or_before(acc: EventAccumulator, tag: str, high: int) -> tuple[float, int]:
    if tag not in acc.Tags().get("scalars", []):
        return np.nan, 0
    values = [event for event in acc.Scalars(tag) if event.step <= high]
    if not values:
        return np.nan, 0
    latest = max(values, key=lambda event: (event.step, event.wall_time))
    return float(latest.value), int(latest.step)


def parse_run(
    event_path: Path,
    terminal_width: int,
    fixed_low: int | None = None,
    fixed_high: int | None = None,
) -> dict[str, object]:
    run_name = event_path.parents[2].name.removeprefix("00_")
    match = RUN_RE.search(run_name)
    if match is None:
        raise ValueError(f"Cannot parse graph-recruitment run name: {run_name}")
    backbone, d_value, h_value, seed = match.groups()
    acc = EventAccumulator(str(event_path), size_guidance={"scalars": 30_000})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if "train/env_steps" not in tags:
        raise ValueError(f"Missing train/env_steps in {event_path}")
    max_step = max(event.step for event in acc.Scalars("train/env_steps"))
    if (fixed_low is None) != (fixed_high is None):
        raise ValueError("fixed_low and fixed_high must be supplied together")
    if fixed_high is not None:
        if fixed_high > max_step:
            raise ValueError(f"Run {run_name} ends at {max_step}, before fixed window {fixed_high}")
        if fixed_low is None or fixed_low < 0 or fixed_low >= fixed_high:
            raise ValueError(f"Invalid fixed window: {fixed_low}--{fixed_high}")
        low, high = fixed_low, fixed_high
    else:
        low, high = max(0, max_step - terminal_width), max_step
    row: dict[str, object] = {
        "backbone": backbone,
        "redundancy_max_steps": int(d_value),
        "half_life_k": int(h_value),
        "seed": int(seed),
        "run": run_name,
        "max_step": int(max_step),
        "window_low": int(low),
        "window_high": int(high),
        "event_file": str(event_path),
    }
    for key, tag in WINDOW_METRICS.items():
        row[key], row[f"{key}__n"] = mean_in_window(acc, tag, low, high)
    for key, tag in CUMULATIVE_METRICS.items():
        row[key], row[f"{key}__step"] = latest_at_or_before(acc, tag, high)
    return row


def aggregate_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = list(WINDOW_METRICS) + list(CUMULATIVE_METRICS)
    keys = ["backbone", "redundancy_max_steps", "half_life_k"]
    grouped = frame.groupby(keys, sort=True)[metrics]
    return (
        grouped.mean().add_suffix("__mean")
        .join(grouped.std(ddof=1).add_suffix("__sd"))
        .join(grouped.count().add_suffix("__n"))
        .reset_index()
    )


def paired_factor_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Within-backbone, within-seed paired D and H effects plus interaction."""

    rows: list[dict[str, object]] = []
    for backbone in BACKBONES:
        subset = frame[frame.backbone == backbone].set_index(
            ["seed", "redundancy_max_steps", "half_life_k"]
        )
        for metric in list(WINDOW_METRICS) + list(CUMULATIVE_METRICS):
            by_seed: dict[int, dict[str, float]] = {}
            for seed in SEEDS:
                def value(d_value: int, h_value: int) -> float:
                    return float(subset.loc[(seed, d_value, h_value), metric])

                d_effect = 0.5 * ((value(8, 5) - value(4, 5)) + (value(8, 10) - value(4, 10)))
                h_effect = 0.5 * ((value(4, 10) - value(4, 5)) + (value(8, 10) - value(8, 5)))
                interaction = (value(8, 10) - value(4, 10)) - (value(8, 5) - value(4, 5))
                by_seed[seed] = {"D8_minus_D4": d_effect, "H10k_minus_H5k": h_effect, "interaction": interaction}
            for effect in ("D8_minus_D4", "H10k_minus_H5k", "interaction"):
                values = np.asarray([by_seed[seed][effect] for seed in SEEDS], dtype=float)
                finite = values[np.isfinite(values)]
                rows.append({
                    "backbone": backbone,
                    "metric": metric,
                    "effect": effect,
                    "mean": float(np.mean(finite)) if len(finite) else np.nan,
                    "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                    "n": int(len(finite)),
                    **{f"seed_{seed}": by_seed[seed][effect] for seed in SEEDS},
                })
    return pd.DataFrame(rows)


def _configure_matplotlib() -> None:
    import matplotlib as mpl
    from matplotlib import font_manager

    font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    if not font_path.lower().endswith((".ttf", ".otf")):
        raise RuntimeError(f"No scalable DejaVu Sans font resolved: {font_path}")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "savefig.dpi": 120,
    })


def plot_condition_grid(aggregate: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    _configure_matplotlib()
    colors = {"C05": "#0072B2", "C13": "#E69F00", "C15": "#009E73"}
    labels = [f"D{d}\nH{h}k" for d, h in PARAMETER_CELLS]
    panels = [
        ("coverage_auc", "Coverage AUC"),
        ("unique_cells", "Unique cells"),
        ("target_hit_lift", "Target-hit lift"),
        ("option_success_fraction", "Option success"),
        ("repeat_total", "Repeat assignments (cumulative)"),
        ("connected_fraction", "Connected DG fraction"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)
    x = np.arange(len(PARAMETER_CELLS), dtype=float)
    offsets = {"C05": -0.18, "C13": 0.0, "C15": 0.18}
    for panel_label, ax, (metric, ylabel) in zip("abcdef", axes.flat, panels):
        for backbone in BACKBONES:
            subset = aggregate[aggregate.backbone == backbone].set_index(
                ["redundancy_max_steps", "half_life_k"]
            )
            mean = np.asarray([subset.loc[cell, f"{metric}__mean"] for cell in PARAMETER_CELLS], dtype=float)
            sd = np.asarray([subset.loc[cell, f"{metric}__sd"] for cell in PARAMETER_CELLS], dtype=float)
            ax.errorbar(x + offsets[backbone], mean, yerr=sd, fmt="o", capsize=4,
                        markersize=7, color=colors[backbone], label=backbone)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#d0d0d0", linewidth=0.8)
        ax.text(
            0.015,
            0.985,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=22,
        )
    axes[0, 0].legend(frameon=False, ncol=3, loc="best")
    fig.savefig(output_dir / "graph_recruitment_interim_overview.png", dpi=120)
    fig.savefig(output_dir / "graph_recruitment_interim_overview.pdf")
    plt.close(fig)


def validate(frame: pd.DataFrame) -> None:
    if len(frame) != 36:
        raise SystemExit(f"Expected 36 event files, found {len(frame)}")
    observed = frame.groupby(["backbone", "redundancy_max_steps", "half_life_k"]).size().to_dict()
    expected = {(backbone, d_value, h_value): 3 for backbone in BACKBONES for d_value, h_value in PARAMETER_CELLS}
    if observed != expected:
        raise SystemExit(f"Incomplete or duplicated factorial matrix: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--terminal-width", type=int, default=10_000_000)
    parser.add_argument("--window-low", type=int)
    parser.add_argument("--window-high", type=int)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    event_paths = sorted(args.batch_root.glob("*/*/.summary/0/events.out.tfevents.*"))
    rows = [
        parse_run(path, args.terminal_width, args.window_low, args.window_high)
        for path in event_paths
    ]
    frame = pd.DataFrame(rows).sort_values(
        ["backbone", "redundancy_max_steps", "half_life_k", "seed"]
    )
    validate(frame)
    aggregate = aggregate_conditions(frame)
    effects = paired_factor_effects(frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "per_run_latest_10m.csv", index=False)
    aggregate.to_csv(args.output_dir / "condition_latest_10m.csv", index=False)
    effects.to_csv(args.output_dir / "paired_factor_effects_latest_10m.csv", index=False)
    if not args.no_plots:
        plot_condition_grid(aggregate, args.output_dir)

    window_description = (
        f"Fixed {args.window_low}--{args.window_high} frame window"
        if args.window_low is not None
        else f"Per-run latest {args.terminal_width}-frame window"
    )
    metadata = {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "runs": int(len(frame)),
        "min_max_step": int(frame.max_step.min()),
        "max_max_step": int(frame.max_step.max()),
        "terminal_width": int(args.terminal_width),
        "fixed_window": (
            [int(args.window_low), int(args.window_high)]
            if args.window_low is not None and args.window_high is not None
            else None
        ),
        "all_terminal": bool((frame.max_step >= 100_000_000).all()),
        "window_metrics": WINDOW_METRICS,
        "cumulative_metrics": CUMULATIVE_METRICS,
        "aggregation": f"{window_description}; condition mean and sample SD over three seeds.",
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
