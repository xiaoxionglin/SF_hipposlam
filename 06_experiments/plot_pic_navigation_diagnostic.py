"""Readable selected-map panels from the diagnostic's lightweight CSV export."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    assert Path(font).suffix.lower() in (".ttf", ".otf") and Path(font).is_file()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 28,
            "axes.titlesize": 28,
            "axes.labelsize": 28,
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 1.2,
        }
    )
    data = pd.read_csv(args.data)
    outputs = []
    pre = data[data.layer == "pre_threshold"].rate
    pre_min, pre_max = (float(pre.min()), float(pre.max())) if len(pre) else (0.0, 1.0)
    for (condition, seed, layer), frame in data[data.layer.isin(["raw", "pre_threshold"])].groupby(
        ["condition", "seed", "layer"], sort=False
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
        for ax, unit in zip(axes.flat, (0, 4, 8, 12)):
            rows = frame[frame.unit == unit]
            values = np.full((19, 19), np.nan)
            for row in rows.itertuples():
                if row.occupancy > 0:
                    values[row.y_bin, row.x_bin] = row.rate
            peak = float(rows.unit_peak.iloc[0])
            normalized = values / peak if layer == "raw" and peak > 0 else values
            im = ax.imshow(
                normalized,
                origin="lower",
                extent=(1, 20, 1, 20),
                vmin=0 if layer == "raw" else pre_min,
                vmax=1 if layer == "raw" else pre_max,
                cmap="viridis",
                interpolation="nearest",
            )
            ax.set_title(f"Unit {unit}\nmax {peak:.3g}", pad=10)
            ax.set_xticks([1, 10, 20])
            ax.set_yticks([1, 10, 20])
            ax.set_xlabel("x / 100")
            ax.set_ylabel("y / 100")
        name = condition.replace("PIC_", "")
        title = "Raw DG" if layer == "raw" else "Pre-threshold logits"
        fig.suptitle(f"{name} · seed {seed}\n{title} · {frame.checkpoint_frames.iloc[0]/1e6:.2f}M frames", fontsize=30)
        ticks = [0, 0.5, 1] if layer == "raw" else np.linspace(pre_min, pre_max, 3)
        bar = fig.colorbar(im, ax=axes, location="bottom", shrink=0.85, pad=0.04, fraction=0.045, ticks=ticks)
        if layer != "raw":
            bar.ax.set_xticklabels([f"{t:.1f}" for t in ticks])
        bar.set_label("Fraction of each unit’s peak" if layer == "raw" else "Logit (common scale)", fontsize=28)
        stem = args.output / f"{condition}_s{seed}_{layer}"
        fig.savefig(stem.with_suffix(".png"), dpi=150)
        fig.savefig(stem.with_suffix(".svg"))
        outputs.append(str(stem.with_suffix(".png")))
        plt.close(fig)
    (args.output / "figure_metadata.json").write_text(
        json.dumps(
            dict(
                input=str(args.data.resolve()),
                font=font,
                matplotlib=matplotlib.__version__,
                outputs=outputs,
                selected_units=[0, 4, 8, 12],
                selection="Fixed before inspecting maps",
                mapping="Occupancy-corrected maps. Raw maps divided by each unit peak to compare shape; absolute peak printed. Pre-threshold logits use one common scale across all displayed conditions and units. No smoothing; unvisited bins blank. Axes divide DMLab x/y coordinates by 100.",
                viewing="PNG intended at about 1000 pixels wide; 28pt source ticks at 150dpi retain approximately 39px at that width.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
