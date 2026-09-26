"""Render the reviewed map and one native first-person approach per cue."""

from __future__ import annotations

import argparse
from pathlib import Path

import deepmind_lab
import matplotlib.pyplot as plt
import numpy as np

from hpc_runs.intrmotiv_study.geometry import load_landmark_geometry


def render(runfiles: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    archive = Path(__file__).with_name("studies") / "assets/easy_landmark_maze/maps.json"
    record = load_landmark_geometry(str(archive), 1001, 0.85, 11, 11, 20260923, "rich")
    deepmind_lab.set_runfiles_path(str(runfiles))

    images = []
    for index, cue in enumerate(record["cue_sites"], start=1):
        lab = deepmind_lab.Lab(
            "easy_landmark_maze_preview",
            ["RGB_INTERLEAVED"],
            config={"previewCue": str(index), "width": "320", "height": "240"},
            renderer="software",
        )
        try:
            lab.reset(seed=1)
            lab.step(np.zeros(7, np.intc), num_steps=1)
            image = np.asarray(lab.observations()["RGB_INTERLEAVED"]).copy()
        finally:
            lab.close()
        images.append((cue, image))

    plt.rcParams.update({"font.size": 12})
    figure, axes = plt.subplots(4, 5, figsize=(16, 10), constrained_layout=True)
    for axis, (cue, image) in zip(axes.flat, images):
        axis.imshow(image)
        axis.set_title(f"{cue['cue_id']} · {cue['cue_type']} · {cue['orientation']}")
        axis.axis("off")
    figure.suptitle("Easy landmark maze: native first-person cue approaches", fontsize=16)
    figure.savefig(output / "cue_approaches.png", dpi=160)
    plt.close(figure)

    grid = np.asarray([list(row) for row in record["entity_layer"].splitlines()])
    figure, axis = plt.subplots(figsize=(10, 10), constrained_layout=True)
    axis.imshow(grid == "*", cmap="Greys", origin="upper", interpolation="nearest")
    for cue in record["cue_sites"]:
        row, column = cue["wall_rc"]
        color = "#1464F4" if cue["cue_type"] == "decal" else cue["asset"]
        axis.scatter(column, row, s=180, c=color, edgecolors="white", linewidths=1.5)
        axis.text(
            column,
            row,
            cue["cue_id"],
            ha="center",
            va="center",
            fontsize=8,
            color="white" if cue["cue_type"] == "decal" else "black",
            weight="bold",
        )
    height, width = record["entity_shape"]
    axis.set_title("Easy landmark maze: native 11×11 entity layer and 20 reserved wall faces")
    axis.set_xticks(range(width))
    axis.set_yticks(range(height))
    axis.set_xlabel("Entity column")
    axis.set_ylabel("Entity row")
    axis.grid(color="#BBBBBB", linewidth=0.4)
    figure.savefig(output / "map_preview.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runfiles", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.runfiles, args.output)
