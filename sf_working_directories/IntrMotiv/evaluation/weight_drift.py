"""Checkpoint-based DG projection drift analysis.

The frozen visual trunk makes the DG projection rows the changing visual
landmark parameters. This tool measures those rows directly from saved
checkpoints; it does not claim that parameter stability proves spatial fields.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


WEIGHT_KEY = "encoder.DG_projection.linear.weight"
BN_MEAN_KEY = "encoder.DG_projection.batchnorm1d.running_mean"
BN_VAR_KEY = "encoder.DG_projection.batchnorm1d.running_var"


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    env_steps: int


def checkpoint_env_steps(path: Path) -> int:
    match = re.search(r"_(\d+)\.pth$", path.name)
    if not match:
        raise ValueError(f"Cannot parse environment steps from {path}")
    return int(match.group(1))


def discover_checkpoints(run_dir: Path) -> list[Checkpoint]:
    candidates = [Checkpoint(path, checkpoint_env_steps(path)) for path in run_dir.glob("checkpoint_p0/**/*.pth")]
    by_step: dict[int, Checkpoint] = {}
    for checkpoint in candidates:
        existing = by_step.get(checkpoint.env_steps)
        # Prefer milestones for historical targets and top-level checkpoints for a final-only step.
        if existing is None or ("milestones" in checkpoint.path.parts and "milestones" not in existing.path.parts):
            by_step[checkpoint.env_steps] = checkpoint
    return sorted(by_step.values(), key=lambda item: item.env_steps)


def select_checkpoints(checkpoints: list[Checkpoint], targets: list[int]) -> list[Checkpoint]:
    if not checkpoints:
        raise ValueError("No checkpoints found")
    selected: dict[int, Checkpoint] = {}
    final_step = checkpoints[-1].env_steps
    for target in targets:
        # The actual final checkpoint is always retained below. Do not add a
        # near-final milestone as a separate point, or it would replace the
        # intended last pre-final (normally 75M) stability comparison.
        if target >= 0.99 * final_step:
            continue
        checkpoint = min(checkpoints, key=lambda item: abs(item.env_steps - target))
        selected[checkpoint.env_steps] = checkpoint
    selected[checkpoints[0].env_steps] = checkpoints[0]
    selected[checkpoints[-1].env_steps] = checkpoints[-1]
    return sorted(selected.values(), key=lambda item: item.env_steps)


def load_state(checkpoint: Checkpoint) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.load(checkpoint.path, map_location="cpu", weights_only=True)["model"]
    missing = [key for key in (WEIGHT_KEY, BN_MEAN_KEY, BN_VAR_KEY) if key not in state]
    if missing:
        raise KeyError(f"{checkpoint.path} lacks {missing}")
    return tuple(state[key].detach().float() for key in (WEIGHT_KEY, BN_MEAN_KEY, BN_VAR_KEY))


def row_cosine(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Cosine of corresponding DG projection rows."""

    return F.cosine_similarity(current, reference, dim=1, eps=1e-8)


def analyze_run(run_dir: Path, targets: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    checkpoints = select_checkpoints(discover_checkpoints(run_dir), targets)
    loaded = [(checkpoint, *load_state(checkpoint)) for checkpoint in checkpoints]
    first_checkpoint, first_weight, first_mean, first_var = loaded[0]
    previous_weight = None
    per_checkpoint: list[dict[str, object]] = []
    for checkpoint, weight, running_mean, running_var in loaded:
        cosine_initial = row_cosine(weight, first_weight)
        cosine_previous = row_cosine(weight, previous_weight) if previous_weight is not None else torch.ones_like(cosine_initial)
        per_checkpoint.append(
            {
                "run": run_dir.name,
                "env_steps": checkpoint.env_steps,
                "checkpoint": str(checkpoint.path),
                "mean_row_cosine_to_initial": float(cosine_initial.mean()),
                "min_row_cosine_to_initial": float(cosine_initial.min()),
                "mean_row_angle_deg_to_initial": float(torch.rad2deg(torch.acos(cosine_initial.clamp(-1, 1))).mean()),
                "mean_row_cosine_to_previous": float(cosine_previous.mean()),
                "weight_norm_mean": float(weight.norm(dim=1).mean()),
                "weight_norm_std": float(weight.norm(dim=1).std(unbiased=False)),
                "bn_running_mean_mae_to_initial": float((running_mean - first_mean).abs().mean()),
                "bn_running_logvar_mae_to_initial": float((running_var.clamp_min(1e-8).log() - first_var.clamp_min(1e-8).log()).abs().mean()),
            }
        )
        previous_weight = weight

    final_checkpoint, final_weight, final_mean, final_var = loaded[-1]
    late_checkpoint, late_weight, _, _ = loaded[-2] if len(loaded) > 1 else loaded[-1]
    initial_cosine = row_cosine(final_weight, first_weight)
    late_cosine = row_cosine(final_weight, late_weight)
    per_unit = pd.DataFrame(
        {
            "run": run_dir.name,
            "dg_id": np.arange(initial_cosine.numel()),
            "final_cosine_to_initial": initial_cosine.numpy(),
            "final_angle_deg_to_initial": torch.rad2deg(torch.acos(initial_cosine.clamp(-1, 1))).numpy(),
            "late_cosine_to_final": late_cosine.numpy(),
            "late_angle_deg_to_final": torch.rad2deg(torch.acos(late_cosine.clamp(-1, 1))).numpy(),
        }
    )
    summary = {
        "run": run_dir.name,
        "initial_env_steps": first_checkpoint.env_steps,
        "late_env_steps": late_checkpoint.env_steps,
        "final_env_steps": final_checkpoint.env_steps,
        "initial_to_final_mean_cosine": float(initial_cosine.mean()),
        "initial_to_final_min_cosine": float(initial_cosine.min()),
        "initial_to_final_mean_angle_deg": float(torch.rad2deg(torch.acos(initial_cosine.clamp(-1, 1))).mean()),
        "late_to_final_mean_cosine": float(late_cosine.mean()),
        "late_to_final_min_cosine": float(late_cosine.min()),
        "late_to_final_stable_rows_cosine_ge_0_99": float((late_cosine >= 0.99).float().mean()),
        "final_weight_norm_mean": float(final_weight.norm(dim=1).mean()),
        "final_weight_norm_std": float(final_weight.norm(dim=1).std(unbiased=False)),
        "final_bn_running_mean_mae_to_initial": float((final_mean - first_mean).abs().mean()),
        "final_bn_running_logvar_mae_to_initial": float((final_var.clamp_min(1e-8).log() - first_var.clamp_min(1e-8).log()).abs().mean()),
    }
    return pd.DataFrame(per_checkpoint), per_unit, summary


def write_report(output_dir: Path, per_checkpoint: pd.DataFrame, per_unit: pd.DataFrame, summary: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_checkpoint.to_csv(output_dir / "dg_projection_drift_by_checkpoint.csv", index=False)
    per_unit.to_csv(output_dir / "dg_projection_drift_by_unit.csv", index=False)
    summary.to_csv(output_dir / "dg_projection_drift_summary.csv", index=False)
    shown = summary[
        [
            "run", "initial_env_steps", "late_env_steps", "final_env_steps",
            "initial_to_final_mean_cosine", "initial_to_final_mean_angle_deg",
            "late_to_final_mean_cosine", "late_to_final_stable_rows_cosine_ge_0_99",
        ]
    ]
    columns = list(shown.columns)
    table = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in shown.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.5f}" if isinstance(value, (float, np.floating)) else str(value))
        table.append("| " + " | ".join(values) + " |")
    lines = [
        "# DG Projection Weight Drift", "",
        "The layer-2 ResNet-18 trunk is ImageNet-pretrained and fixed. This report measures corresponding rows of `encoder.DG_projection.linear.weight` with cosine similarity. Row norm stability is not receptive-field stability; the probe does not measure DG activity on fixed observations or spatial place fields.",
        "",
        "`late_to_final` compares the last two sampled checkpoints, normally near 75M and 100M frames. `stable_rows_cosine_ge_0_99` is a descriptive threshold, not a learned consolidation criterion.",
        "",
        "## Summary", "",
        *table, "",
        "## Interpretation", "",
        "Repeated high reward or target hits are not causally evaluated here. In the present HRL implementation, target-hit reward gates the worker reward, while DG projection updates use the separate encoder-reward stream. A causal stabilization claim requires future fixed-observation and spatial-field probes conditioned on visits and reward.",
    ]
    (output_dir / "dg_projection_drift_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure DG projection row drift from IntrMotiv checkpoints.")
    parser.add_argument("run_dir", nargs="+", type=Path, help="Sample Factory policy directories containing checkpoint_p0.")
    parser.add_argument("--output", type=Path, required=True, help="Workspace directory for analysis artifacts.")
    parser.add_argument("--checkpoint-targets-m", type=float, nargs="+", default=[0, 25, 50, 75])
    args = parser.parse_args()
    targets = [int(value * 1_000_000) for value in args.checkpoint_targets_m]
    results = [analyze_run(run_dir, targets) for run_dir in args.run_dir]
    write_report(
        args.output,
        pd.concat([result[0] for result in results], ignore_index=True),
        pd.concat([result[1] for result in results], ignore_index=True),
        pd.DataFrame([result[2] for result in results]),
    )
    print(f"Wrote DG projection drift analysis for {len(results)} runs to {args.output}")


if __name__ == "__main__":
    main()
