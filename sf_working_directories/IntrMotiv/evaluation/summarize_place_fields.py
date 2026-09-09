"""Summarize occupancy-corrected DG place-field rollout artifacts.

The companion :mod:`place_fields` evaluator is intentionally expensive because
it runs DMLab. This utility operates only on its ``place_fields.npz`` outputs
and therefore makes field quality checks repeatable without another rollout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _map_cosines(rate_maps: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    """Return off-diagonal cosine similarities over cells actually visited."""
    samples = np.nan_to_num(rate_maps[occupancy > 0].T, nan=0.0)
    norms = np.linalg.norm(samples, axis=1)
    denom = norms[:, None] * norms[None, :]
    cosine = np.divide(samples @ samples.T, denom, out=np.zeros_like(denom), where=denom > 0)
    return cosine[np.triu_indices(cosine.shape[0], k=1)]


def summarize_artifact(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=False)
    occupancy = data["occupancy"]
    rate_maps = data["rate_maps"]
    information = data["spatial_information"]
    active = data["active_fraction"]
    cosines = _map_cosines(rate_maps, occupancy)
    peaks = np.nanargmax(rate_maps.reshape(-1, rate_maps.shape[-1]), axis=0)
    peak_bins = np.column_stack(np.unravel_index(peaks, rate_maps.shape[:2]))

    summary = {
        "label": path.parent.name,
        "frames": int(occupancy.sum()),
        "visited_cells": int((occupancy > 0).sum()),
        "visited_cell_fraction": float((occupancy > 0).mean()),
        "mean_spatial_information_bits": float(np.mean(information)),
        "median_spatial_information_bits": float(np.median(information)),
        "max_spatial_information_bits": float(np.max(information)),
        "units_si_over_0_5_bits": int((information > 0.5).sum()),
        "mean_active_fraction": float(np.mean(active)),
        "silent_units": int((active == 0).sum()),
        "map_cosine_mean": float(cosines.mean()) if cosines.size else np.nan,
        "map_cosine_min": float(cosines.min()) if cosines.size else np.nan,
        "map_cosine_max": float(cosines.max()) if cosines.size else np.nan,
        "unique_peak_bins": int(len({tuple(row) for row in peak_bins})),
    }
    if "pre_threshold_rate_maps" in data:
        summary.update(
            mean_pre_threshold_logit=float(np.mean(data["pre_threshold_mean"])),
            mean_pre_threshold_logit_std=float(np.mean(data["pre_threshold_std"])),
            mean_pre_threshold_above_fraction=float(np.mean(data["pre_threshold_above_fraction"])),
        )
    return summary


def display_label(label: str) -> str:
    """Turn the filesystem run label into a compact, unambiguous plot title."""
    suffix = label.rsplit("__", 1)[-1]
    checkpoint = f" | checkpoint {suffix}" if suffix.endswith("M") and suffix[:-1].isdigit() else ""
    if label.startswith("GHRL_"):
        half_life = label.split("_HL", 1)[1].split("_", 1)[0]
        schedule = "iterative" if "_iter_" in label else "simultaneous"
        seed = label.split("_S", 1)[1].split("_", 1)[0]
        return f"Global HRL | half-life {half_life} | {schedule} | seed {seed}{checkpoint}"
    if label.startswith("FB_"):
        schedule = "iterative" if "_iter_" in label else "simultaneous"
        seed = label.split("_S", 1)[1].split("_", 1)[0]
        return f"Flat encourage | {schedule} | seed {seed}{checkpoint}"
    return label


def compact_label(label: str) -> str:
    if label.startswith("GHRL_"):
        half_life = label.split("_HL", 1)[1].split("_", 1)[0]
        schedule = "iter" if "_iter_" in label else "sim"
        seed = label.split("_S", 1)[1].split("_", 1)[0]
        return f"GHRL {half_life} {schedule} S{seed}"
    if label.startswith("FB_"):
        schedule = "iter" if "_iter_" in label else "sim"
        seed = label.split("_S", 1)[1].split("_", 1)[0]
        return f"Flat {schedule} S{seed}"
    return label


def _font(size: int):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _viridis(value: float) -> tuple[int, int, int]:
    """Small dependency-free approximation to the viridis color map."""
    stops = ((68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37))
    value = min(max(value, 0.0), 1.0) * (len(stops) - 1)
    index = min(int(value), len(stops) - 2)
    fraction = value - index
    return tuple(round(stops[index][k] * (1.0 - fraction) + stops[index + 1][k] * fraction) for k in range(3))


def _rate_map_image(rate_map: np.ndarray, vmax: float, cell_pixels: int):
    from PIL import Image

    image = Image.new("RGB", (rate_map.shape[1], rate_map.shape[0]), (235, 235, 235))
    pixels = image.load()
    for y in range(rate_map.shape[0]):
        for x in range(rate_map.shape[1]):
            value = rate_map[y, x]
            if np.isfinite(value):
                pixels[x, y] = _viridis(float(value) / vmax)
    return image.resize((image.width * cell_pixels, image.height * cell_pixels), Image.Resampling.NEAREST)


def _logit_map_image(rate_map: np.ndarray, vmax: float, cell_pixels: int):
    """Render signed logits blue-white-red while retaining gray unvisited cells."""
    from PIL import Image

    image = Image.new("RGB", (rate_map.shape[1], rate_map.shape[0]), (235, 235, 235))
    pixels = image.load()
    for y in range(rate_map.shape[0]):
        for x in range(rate_map.shape[1]):
            value = rate_map[y, x]
            if not np.isfinite(value):
                continue
            normalized = min(max(float(value) / vmax, -1.0), 1.0)
            if normalized < 0:
                fraction = normalized + 1.0
                color = tuple(round((59, 76, 192)[i] * (1.0 - fraction) + 245 * fraction) for i in range(3))
            else:
                fraction = normalized
                color = tuple(round(245 * (1.0 - fraction) + (180, 4, 38)[i] * fraction) for i in range(3))
            pixels[x, y] = color
    return image.resize((image.width * cell_pixels, image.height * cell_pixels), Image.Resampling.NEAREST)


def plot_run(path: Path, out_path: Path) -> None:
    """Render all DG rate maps for one run on a shared within-run color scale."""
    from PIL import Image, ImageDraw

    data = np.load(path, allow_pickle=False)
    occupancy = data["occupancy"]
    rate_maps = data["rate_maps"]
    information = data["spatial_information"]
    active = data["active_fraction"]
    finite = rate_maps[np.isfinite(rate_maps)]
    vmax = max(float(np.percentile(finite, 98)) if finite.size else 0.0, 1e-8)
    n_units = rate_maps.shape[-1]
    n_cols = 4
    n_rows = int(np.ceil(n_units / n_cols))
    cell_pixels = 10
    map_pixels = rate_maps.shape[0] * cell_pixels
    title_height = 96
    panel_width = 300
    panel_height = 270
    image = Image.new("RGB", (n_cols * panel_width, title_height + n_rows * panel_height + 30), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(22)
    unit_font = _font(14)
    footer_font = _font(13)
    draw.text((18, 12), "DG occupancy-corrected rate maps", font=title_font, fill="black")
    draw.text((18, 42), display_label(path.parent.name), font=unit_font, fill="black")
    draw.text(
        (18, 64),
        f"shared within-run scale; {int(occupancy.sum())} policy decisions; gray = unvisited cell",
        font=footer_font,
        fill="black",
    )
    occupancy_mask = occupancy.T <= 0
    for unit in range(n_units):
        row, col = divmod(unit, n_cols)
        x = col * panel_width + 18
        y = title_height + row * panel_height + 30
        rate_map = rate_maps[:, :, unit].T.copy()
        rate_map[occupancy_mask] = np.nan
        map_image = _rate_map_image(rate_map, vmax, cell_pixels)
        image.paste(map_image, (x, y))
        draw.rectangle((x, y, x + map_pixels, y + map_pixels), outline=(80, 80, 80), width=1)
        draw.text(
            (x, y + map_pixels + 8),
            f"DG {unit:02d} | SI {information[unit]:.2f} bits | active {active[unit]:.3f}",
            font=unit_font,
            fill="black",
        )
    image.save(out_path)


def plot_pre_threshold_logits(path: Path, out_path: Path) -> None:
    """Render continuous pre-threshold logits when the evaluator stored them."""
    from PIL import Image, ImageDraw

    data = np.load(path, allow_pickle=False)
    if "pre_threshold_rate_maps" not in data:
        return
    occupancy = data["occupancy"]
    rate_maps = data["pre_threshold_rate_maps"]
    mean_logits = data["pre_threshold_mean"]
    std_logits = data["pre_threshold_std"]
    finite = rate_maps[np.isfinite(rate_maps)]
    vmax = max(float(np.percentile(np.abs(finite), 98)) if finite.size else 0.0, 1e-8)
    n_units = rate_maps.shape[-1]
    n_cols = 4
    n_rows = int(np.ceil(n_units / n_cols))
    cell_pixels = 10
    map_pixels = rate_maps.shape[0] * cell_pixels
    title_height = 96
    panel_width = 300
    panel_height = 270
    image = Image.new("RGB", (n_cols * panel_width, title_height + n_rows * panel_height + 30), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(22)
    unit_font = _font(14)
    footer_font = _font(13)
    draw.text((18, 12), "DG occupancy-corrected pre-threshold logit maps", font=title_font, fill="black")
    draw.text((18, 42), display_label(path.parent.name), font=unit_font, fill="black")
    draw.text(
        (18, 64),
        f"symmetric shared scale; {int(occupancy.sum())} policy decisions; gray = unvisited cell",
        font=footer_font,
        fill="black",
    )
    occupancy_mask = occupancy.T <= 0
    for unit in range(n_units):
        row, col = divmod(unit, n_cols)
        x = col * panel_width + 18
        y = title_height + row * panel_height + 30
        rate_map = rate_maps[:, :, unit].T.copy()
        rate_map[occupancy_mask] = np.nan
        map_image = _logit_map_image(rate_map, vmax, cell_pixels)
        image.paste(map_image, (x, y))
        draw.rectangle((x, y, x + map_pixels, y + map_pixels), outline=(80, 80, 80), width=1)
        draw.text(
            (x, y + map_pixels + 8),
            f"DG {unit:02d} | mean {mean_logits[unit]:.2f} | std {std_logits[unit]:.2f}",
            font=unit_font,
            fill="black",
        )
    image.save(out_path)


def plot_comparison(rows: list[dict[str, object]], out_path: Path) -> None:
    """Render high-level selectivity and redundancy comparison across runs."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1600, 720), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(25)
    axis_font = _font(16)
    label_font = _font(13)
    draw.text((24, 20), "Representative final-checkpoint DG place-field comparison", font=title_font, fill="black")

    panels = (
        (70, "Mean DG spatial information (bits)", "mean_spatial_information_bits", 0.65, (55, 126, 184)),
        (840, "Mean pairwise DG-map cosine (1 = identical)", "map_cosine_mean", 1.0, (230, 85, 13)),
    )
    chart_top, chart_height, chart_width = 105, 430, 650
    for left, title, key, maximum, color in panels:
        draw.text((left, 72), title, font=axis_font, fill="black")
        draw.line((left, chart_top, left, chart_top + chart_height), fill=(80, 80, 80), width=2)
        draw.line(
            (left, chart_top + chart_height, left + chart_width, chart_top + chart_height), fill=(80, 80, 80), width=2
        )
        for tick in range(5):
            y = chart_top + chart_height - int(chart_height * tick / 4)
            value = maximum * tick / 4
            draw.line((left, y, left + chart_width, y), fill=(225, 225, 225), width=1)
            draw.text((left - 50, y - 7), f"{value:.2f}", font=label_font, fill=(80, 80, 80))
        bar_width = 78
        spacing = 42
        for index, row in enumerate(rows):
            x = left + 32 + index * (bar_width + spacing)
            value = float(row[key])
            height = int(chart_height * min(value / maximum, 1.0))
            draw.rectangle((x, chart_top + chart_height - height, x + bar_width, chart_top + chart_height), fill=color)
            draw.text((x, chart_top + chart_height - height - 19), f"{value:.3f}", font=label_font, fill="black")
            draw.text(
                (x - 8, chart_top + chart_height + 12), compact_label(str(row["label"])), font=label_font, fill="black"
            )
    image.save(out_path)


def write_report(rows: list[dict[str, object]], out_path: Path) -> None:
    lines = [
        "# DG Place-Field Rollout Summary",
        "",
        "Each field is the occupancy-corrected mean current DG activation in a "
        "19x19 spatial bin. Spatial information is in bits: ",
        "",
        "```text",
        "I = sum_c p(c) * r(c) * log2(r(c) / r_bar)",
        "```",
        "",
        "where `p(c)` is rollout occupancy, `r(c)` is that unit's mean activity "
        "in cell `c`, and `r_bar` is its occupancy-weighted mean activity. "
        "Map cosine measures redundancy between two DG rate maps over visited "
        "cells; one means identical nonnegative maps. These are descriptive "
        "rollout metrics, not fixed-probe field-stability measurements.",
        "",
        "| Run | Samples | Visited cells | Mean SI (bits) | Active fraction | Map cosine | Unique peak bins |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {frames} | {visited_cells} | {mean_spatial_information_bits:.3f} | "
            "{mean_active_fraction:.3f} | {map_cosine_mean:.3f} | {unique_peak_bins} |".format(**row)
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifacts = sorted(args.input_dir.glob("*/place_fields.npz"))
    if not artifacts:
        raise FileNotFoundError(f"No place_fields.npz files under {args.input_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_artifact(path) for path in artifacts]
    pd.DataFrame(rows).to_csv(args.out_dir / "place_field_summary.csv", index=False)
    write_report(rows, args.out_dir / "place_field_summary.md")
    if not args.no_plots:
        for path in artifacts:
            plot_run(path, args.out_dir / f"place_fields_{path.parent.name}.png")
            plot_pre_threshold_logits(path, args.out_dir / f"pre_threshold_logits_{path.parent.name}.png")
        plot_comparison(rows, args.out_dir / "place_field_comparison.png")
    print(f"Wrote {len(rows)} summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
