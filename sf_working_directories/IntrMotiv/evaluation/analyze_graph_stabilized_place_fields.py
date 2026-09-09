#!/usr/bin/env python3
"""Summarize the aligned Graph-Stabilized Recruitment place-field sweep."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


BACKBONES = ("C05", "C13", "C15")
CELLS = ((4, 5), (4, 10), (8, 5), (8, 10))
SEEDS = (8, 99, 123)
RUN_RE = re.compile(r"^gsr_(c\d+)_d(\d+)_h(\d+)k$")
METRICS = (
    "visited_cells",
    "active_units",
    "silent_units",
    "mean_active_fraction",
    "active_unit_mean_si_bits",
    "active_map_cosine_mean",
    "active_unique_peak_bins",
    "active_peak_bin_entropy",
    "active_pairwise_peak_distance_bins",
    "prethreshold_map_cosine_mean",
    "prethreshold_unique_peak_bins",
    "prethreshold_peak_bin_entropy",
    "prethreshold_pairwise_peak_distance_bins",
)


def parse_design(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame["condition"].str.extract(RUN_RE)
    if parsed.isna().any().any():
        bad = frame.loc[parsed.isna().any(axis=1), "condition"].tolist()
        raise ValueError(f"Unrecognized conditions: {bad}")
    result = frame.copy()
    result["backbone"] = parsed[0].str.upper()
    result["redundancy_max_steps"] = parsed[1].astype(int)
    result["half_life_k"] = parsed[2].astype(int)
    return result


def validate(frame: pd.DataFrame) -> None:
    expected = {
        (backbone, distance, half_life, seed)
        for backbone in BACKBONES
        for distance, half_life in CELLS
        for seed in SEEDS
    }
    observed = set(
        frame[["backbone", "redundancy_max_steps", "half_life_k", "seed"]]
        .itertuples(index=False, name=None)
    )
    if observed != expected:
        raise ValueError(
            f"Incomplete factorial matrix; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    if len(frame) != 36:
        raise ValueError(f"Expected 36 rows, found {len(frame)}")
    if frame["target_frames"].nunique() != 1:
        raise ValueError("All rows must share one target_frames value")
    if not (frame["frames"] == 10_001).all():
        raise ValueError("Every rollout must contain 10,001 occupancy samples")


def condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["backbone", "redundancy_max_steps", "half_life_k"]
    grouped = frame.groupby(keys, sort=True)[list(METRICS)]
    return (
        grouped.mean()
        .add_suffix("__mean")
        .join(grouped.std(ddof=1).add_suffix("__sd"))
        .join(grouped.count().add_suffix("__n"))
        .reset_index()
    )


def paired_factor_effects(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for backbone in BACKBONES:
        subset = frame[frame["backbone"] == backbone].set_index(
            ["seed", "redundancy_max_steps", "half_life_k"]
        )
        for metric in METRICS:
            by_seed: dict[int, dict[str, float]] = {}
            for seed in SEEDS:
                def value(distance: int, half_life: int) -> float:
                    return float(subset.loc[(seed, distance, half_life), metric])

                by_seed[seed] = {
                    "D8_minus_D4": 0.5
                    * (
                        value(8, 5)
                        - value(4, 5)
                        + value(8, 10)
                        - value(4, 10)
                    ),
                    "H10k_minus_H5k": 0.5
                    * (
                        value(4, 10)
                        - value(4, 5)
                        + value(8, 10)
                        - value(8, 5)
                    ),
                    "interaction": (
                        value(8, 10)
                        - value(4, 10)
                        - value(8, 5)
                        + value(4, 5)
                    ),
                }
            for effect in ("D8_minus_D4", "H10k_minus_H5k", "interaction"):
                values = np.asarray([by_seed[seed][effect] for seed in SEEDS])
                rows.append(
                    {
                        "backbone": backbone,
                        "metric": metric,
                        "effect": effect,
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)),
                        "n": len(values),
                        **{
                            f"seed_{seed}": by_seed[seed][effect]
                            for seed in SEEDS
                        },
                    }
                )
    return pd.DataFrame(rows)


def _configure_matplotlib() -> None:
    import matplotlib as mpl
    from matplotlib import font_manager

    font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    if not font_path.lower().endswith((".ttf", ".otf")):
        raise RuntimeError(f"No scalable DejaVu Sans font resolved: {font_path}")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_overview(frame: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    _configure_matplotlib()
    colors = {"C05": "#0072B2", "C13": "#E69F00", "C15": "#009E73"}
    markers = {"C05": "o", "C13": "s", "C15": "^"}
    offsets = {"C05": -0.20, "C13": 0.0, "C15": 0.20}
    seed_jitter = {8: -0.045, 99: 0.0, 123: 0.045}
    labels = [f"D{distance}\nH{half_life}k" for distance, half_life in CELLS]
    panels = (
        ("active_map_cosine_mean", "Active-map cosine", None),
        ("active_unit_mean_si_bits", "Active-unit spatial information (bits)", None),
        ("active_unique_peak_bins", "Unique active peak bins (of 16)", 16.0),
        ("prethreshold_map_cosine_mean", "Pre-threshold map cosine", 0.0),
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    x = np.arange(len(CELLS), dtype=float)
    for panel_label, ax, (metric, ylabel, reference) in zip("abcd", axes.flat, panels):
        for backbone in BACKBONES:
            means: list[float] = []
            sds: list[float] = []
            for cell_index, (distance, half_life) in enumerate(CELLS):
                values = frame[
                    (frame["backbone"] == backbone)
                    & (frame["redundancy_max_steps"] == distance)
                    & (frame["half_life_k"] == half_life)
                ].sort_values("seed")
                means.append(float(values[metric].mean()))
                sds.append(float(values[metric].std(ddof=1)))
                for row in values.itertuples(index=False):
                    ax.scatter(
                        cell_index + offsets[backbone] + seed_jitter[int(row.seed)],
                        getattr(row, metric),
                        color=colors[backbone],
                        marker=markers[backbone],
                        s=34,
                        alpha=0.5,
                        linewidths=0,
                        zorder=2,
                    )
            ax.errorbar(
                x + offsets[backbone],
                means,
                yerr=sds,
                color=colors[backbone],
                marker=markers[backbone],
                markersize=8,
                capsize=4,
                linestyle="none",
                label=backbone,
                zorder=3,
            )
        if reference is not None:
            ax.axhline(reference, color="#777777", linewidth=1.0, linestyle="--", zorder=1)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#d0d0d0", linewidth=0.8, zorder=0)
        ax.text(
            0.015,
            0.985,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=20,
            fontweight="bold",
        )
    axes[0, 0].legend(
        title="Backbone",
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(1.08, 1.18),
    )
    fig.savefig(output_dir / "aligned_75m_place_field_overview.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "aligned_75m_place_field_overview.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    frame = parse_design(pd.read_csv(args.input_csv))
    validate(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    condition_summary(frame).to_csv(
        args.output_dir / "place_field_condition_summary.csv", index=False
    )
    paired_factor_effects(frame).to_csv(
        args.output_dir / "place_field_paired_factor_effects.csv", index=False
    )
    plot_overview(frame, args.output_dir)
    print(
        f"Analyzed {len(frame)} runs at target {int(frame.target_frames.iloc[0])}; "
        f"wrote summaries and overview to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
