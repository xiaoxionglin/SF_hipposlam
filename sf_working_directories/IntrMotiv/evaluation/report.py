"""Machine-readable tables and a concise diagnostic Markdown report."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .schema import EVALUATION_SCHEMA_VERSION, METRICS

OUTCOME_COLUMNS = [
    "max_step",
    "training_progress",
    "coverage_auc",
    "coverage_unique_cells",
    "coverage_entropy",
    "dg_density",
    "dg_silent_unit_fraction",
    "dg_multi_activation_fraction",
    "intrinsic_reward_mean",
    "intrinsic_reward_nonzero_fraction",
    "target_hit_rate",
    "option_timeout_rate",
    "option_success_fraction",
    "known_edge_fraction",
    "tctrl_update_rate",
]


def _summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    numeric = [column for column in OUTCOME_COLUMNS if column in frame]
    result = frame.groupby(group_columns, dropna=False)[numeric].agg(["mean", "std", "count"])
    result.columns = ["__".join(column) for column in result.columns]
    return result.reset_index()


def _display(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if math.isnan(number) else f"{number:.{digits}g}"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "No matching runs.\n"
    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(_display(value) if isinstance(value, (int, float)) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows) + "\n"


def write_outputs(frame: pd.DataFrame, output_dir: Path, batch_roots: list[Path], target_env_steps: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "per_run_terminal.csv", index=False)
    family = _summary(frame, ["batch", "family", "graph_scope"])
    family.to_csv(output_dir / "family_terminal_summary.csv", index=False)
    conditions = _summary(
        frame,
        [
            "batch",
            "family",
            "graph_scope",
            "update_schedule",
            "half_life_options",
            "encoder_reward_method",
            "dg_anti_collapse_arm",
            "dg_threshold",
            "dg_global_punishment_coeff",
            "dg_row_repulsion_coeff",
        ],
    )
    conditions.to_csv(output_dir / "condition_terminal_summary.csv", index=False)
    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "batch_roots": [str(root) for root in batch_roots],
        "target_env_steps": target_env_steps,
        "runs": int(len(frame)),
        "families": frame.groupby("family").size().to_dict(),
        "observed_metric_runs": {metric.key: int(frame[f"{metric.key}__samples"].gt(0).sum()) for metric in METRICS},
        "terminal_window": "last min(10M frames, 20% of observed frames), with a 1M minimum",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_markdown(frame, family, output_dir / "diagnostic_report.md", manifest)


def _write_markdown(frame: pd.DataFrame, family: pd.DataFrame, path: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# IntrMotiv Retrospective Diagnostic Report",
        "",
        f"Schema: `{manifest['schema_version']}`.",
        "",
        "## Scope",
        "",
        "This report uses existing TensorBoard scalars only. It can establish training progress, external coverage, DG minibatch health, and the observable HRL option funnel. It cannot retrospectively establish DG place fields, chance-corrected target control, target-conditioned policy sensitivity, or graph calibration because the required trajectory-level records were not collected.",
        "",
        "## Run Validity",
        "",
        _markdown_table(
            frame.groupby(["batch", "family", "run_status"], dropna=False).size().reset_index(name="runs"),
            ["batch", "family", "run_status", "runs"],
        ),
        "## Family Terminal Summary",
        "",
        _markdown_table(
            family,
            [
                "batch",
                "family",
                "graph_scope",
                "max_step__mean",
                "training_progress__mean",
                "coverage_auc__mean",
                "coverage_unique_cells__mean",
                "dg_density__mean",
                "dg_silent_unit_fraction__mean",
                "target_hit_rate__mean",
                "option_success_fraction__mean",
                "known_edge_fraction__mean",
            ],
        ),
        "## Interpretation Guards",
        "",
        "- Compare coverage only within the same measurement scope. `physical_episode` and `telemetry_window` are not interchangeable.",
        "- A nonzero target-hit rate includes incidental DG matches; it is not evidence of target following without a chance baseline and trajectory probe.",
        "- A nonzero known-edge fraction only says that graph entries passed the logged criterion. It does not demonstrate calibrated or causal reachability.",
        "- DG density and minibatch silent-unit fraction are health checks, not spatial place-field measurements.",
        "",
        "## Metrics Still Required For Causal Diagnosis",
        "",
        "Checkpoint evaluation should add position/DG traces, option-event records, target marginal activation frequencies, selected-target versus shuffled-target policy probes, and predicted-versus-realized arrival times. These are intentionally listed as missing rather than estimated from current scalars.",
    ]
    path.write_text("\n".join(lines) + "\n")
