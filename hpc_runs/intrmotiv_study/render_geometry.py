"""Render archived geometry, one readable three-condition page per map seed."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap


def render(archive, output):
    font = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
    if font.suffix.lower() not in (".ttf", ".otf"):
        raise ValueError("Scalable font required")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "pdf.fonttype": 42,
        }
    )
    records = json.loads(Path(archive).read_text())["maps"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for seed in sorted({r["map_seed"] for r in records}):
        row = sorted([r for r in records if r["map_seed"] == seed], key=lambda r: r["wall_removal_probability"])
        fig, axes = plt.subplots(1, len(row), figsize=(15, 5.8), layout="constrained")
        for ax, r in zip(axes, row):
            grid = np.array([list(line) for line in r["entity_layer"].splitlines()])
            ax.imshow(
                grid != "*",
                cmap=ListedColormap(["#252525", "#fafafa"]),
                interpolation="nearest",
                extent=(0, 2100, 0, 2100),
            )
            ax.set_title(
                f"Opening {r['wall_removal_probability']:g}\n{r['accessible_cells']} floor · degree two {r['corridor_fraction']:.0%}"
            )
            ax.set_xlabel("x (DMLab units)")
            ax.set_xticks([0, 1000, 2000])
            ax.set_yticks([0, 1000, 2000])
        axes[0].set_ylabel("y (DMLab units)")
        fig.suptitle(f"Map seed {seed} — shared maze backbone and spawn cells", fontsize=22)
        fig.supxlabel("Black: wall · White: accessible floor · No smoothing or aggregation", fontsize=18)
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"map_seed_{seed}.{suffix}", dpi=100)
        plt.close(fig)
    (output / "figure_metadata.json").write_text(
        json.dumps(
            {
                "source": str(archive),
                "font": str(font),
                "units": "DMLab world coordinates",
                "transformation": "entity walls vs accessible floor; no aggregation",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive")
    p.add_argument("output")
    a = p.parse_args()
    render(a.archive, a.output)
