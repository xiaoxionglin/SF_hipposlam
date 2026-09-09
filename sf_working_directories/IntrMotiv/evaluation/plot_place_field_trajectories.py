"""Create time-resolved DG field plots from a sweep manifest and NPZ artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from summarize_place_fields import _font, display_label, plot_run, summarize_artifact

COLORS = ((55, 126, 184), (228, 26, 28), (77, 175, 74), (152, 78, 163), (255, 127, 0), (166, 86, 40))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw/<run>/place_fields.npz")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def artifact_for_suffix(raw_dir: Path, suffix: str) -> Path:
    matches = list(raw_dir.glob(f"*__{suffix}/place_fields.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one field artifact for {suffix}, found {len(matches)}")
    return matches[0]


def collect_rows(manifest: list[dict[str, str]], raw_dir: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    rows: list[dict[str, object]] = []
    paths: dict[str, Path] = {}
    for item in manifest:
        path = artifact_for_suffix(raw_dir, item["label_suffix"])
        row = summarize_artifact(path)
        row.update(item)
        row["checkpoint_frames"] = int(item["checkpoint_frames"])
        row["target_frames"] = int(item["target_frames"])
        rows.append(row)
        paths[item["label_suffix"]] = path
    return pd.DataFrame(rows), paths


def chart_label(row: pd.Series) -> str:
    condition = str(row.get("condition", ""))
    if condition.startswith(("flat_t", "global_t")):
        return condition.replace("_", " ")
    family = row["family"]
    if family == "fixed_flat":
        return f"{row['feedback']} {row['schedule']}"
    if family == "global_hrl":
        return f"HL {row['half_life']} {row['schedule']}"
    if family == "stream_long_hrl":
        return f"stream HRL {row['schedule']}"
    return f"long flat {row['schedule']}"


def draw_family_trajectory(frame: pd.DataFrame, families: tuple[str, ...], title: str, out_path: Path) -> None:
    subset = frame[frame["family"].isin(families)].sort_values(["condition", "checkpoint_frames"])
    conditions = list(dict.fromkeys(subset["condition"]))
    image = Image.new("RGB", (1760, 900), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, small_font = _font(26), _font(17), _font(13)
    draw.text((30, 20), title, font=title_font, fill="black")
    plots = (
        (80, "Mean spatial information (bits)", "mean_spatial_information_bits", 0.8),
        (920, "Mean DG map cosine (1 = identical)", "map_cosine_mean", 1.0),
    )
    top, height, width, baseline = 135, 470, 690, 0.0
    for left, metric_title, metric, maximum in plots:
        draw.text((left, 96), metric_title, font=label_font, fill="black")
        draw.line((left, top, left, top + height), fill=(70, 70, 70), width=2)
        draw.line((left, top + height, left + width, top + height), fill=(70, 70, 70), width=2)
        for tick in range(5):
            value = maximum * tick / 4
            y = top + height - round(height * (value - baseline) / (maximum - baseline))
            draw.line((left, y, left + width, y), fill=(225, 225, 225), width=1)
            draw.text((left - 58, y - 7), f"{value:.2f}", font=small_font, fill=(80, 80, 80))
        for tick in (0, 25, 50, 75, 100):
            x = left + round(width * tick / 100)
            draw.line((x, top + height, x, top + height + 6), fill=(70, 70, 70), width=1)
            draw.text((x - 11, top + height + 12), f"{tick}M", font=small_font, fill=(80, 80, 80))
        for index, condition in enumerate(conditions):
            run = subset[subset["condition"] == condition]
            color = COLORS[index % len(COLORS)]
            points = []
            for _, row in run.iterrows():
                x = left + round(width * float(row["checkpoint_frames"]) / 100_000_000)
                value = float(row[metric])
                y = top + height - round(height * (value - baseline) / (maximum - baseline))
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            for point in points:
                draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=color)
        draw.text(
            (left + width // 2 - 100, top + height + 45), "checkpoint environment frames", font=small_font, fill="black"
        )
    legend_top = 665
    for index, condition in enumerate(conditions):
        example = subset[subset["condition"] == condition].iloc[0]
        x = 100 + (index % 2) * 760
        y = legend_top + (index // 2) * 42
        color = COLORS[index % len(COLORS)]
        draw.rectangle((x, y + 4, x + 24, y + 24), fill=color)
        draw.text((x + 34, y), chart_label(example), font=label_font, fill="black")
    image.save(out_path)


def make_contact_sheets(frame: pd.DataFrame, paths: dict[str, Path], out_dir: Path) -> None:
    grid_dir = out_dir / "per_checkpoint_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    for condition, group in frame.groupby("condition", sort=False):
        # New manifests include final checkpoints for replica seeds. A contact
        # sheet is a time trajectory, so use the full seed-99 checkpoint path
        # when it is available. Historical manifests already have seed 99.
        if "seed" in group and (group["seed"].astype(str) == "99").any():
            group = group[group["seed"].astype(str) == "99"]
        tiles: list[Image.Image] = []
        for _, row in group.sort_values("checkpoint_frames").iterrows():
            suffix = str(row["label_suffix"])
            tile_path = grid_dir / f"{suffix}.png"
            plot_run(paths[suffix], tile_path)
            tile = Image.open(tile_path).convert("RGB")
            tile.thumbnail((570, 610), Image.Resampling.LANCZOS)
            tiles.append(tile)
        canvas = Image.new("RGB", (1770, 1370), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (24, 18), f"DG field evolution | {chart_label(group.iloc[0])} | seed 99", font=_font(25), fill="black"
        )
        for index, tile in enumerate(tiles):
            row, col = divmod(index, 3)
            x, y = 12 + col * 585, 75 + row * 640
            canvas.paste(tile, (x, y))
        canvas.save(out_dir / f"field_evolution_{condition}.png")


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame, paths = collect_rows(manifest, args.input_dir / "raw")
    frame.to_csv(args.out_dir / "place_field_trajectory_summary.csv", index=False)
    make_contact_sheets(frame, paths, args.out_dir)
    draw_family_trajectory(
        frame,
        ("fixed_flat", "flat"),
        "DG field trajectories: fixed flat encoder feedback",
        args.out_dir / "trajectory_fixed_flat.png",
    )
    draw_family_trajectory(
        frame,
        ("global_hrl", "global_fixed_hrl"),
        "DG field trajectories: fixed/global HRL",
        args.out_dir / "trajectory_global_hrl.png",
    )
    long_frame = frame[frame["family"].isin(("stream_long_hrl", "long_flat"))]
    draw_family_trajectory(
        long_frame,
        ("stream_long_hrl",),
        "DG field trajectories: long per-stream HRL",
        args.out_dir / "trajectory_stream_long_hrl.png",
    )
    draw_family_trajectory(
        long_frame,
        ("long_flat",),
        "DG field trajectories: long flat controls",
        args.out_dir / "trajectory_long_flat.png",
    )
    print(f"Wrote trajectories for {len(frame)} checkpoint artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
