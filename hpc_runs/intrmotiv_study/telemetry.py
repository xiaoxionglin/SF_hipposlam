"""Generate the established place-field manifests from a StudySpec."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .discovery import discover_run_directories
from .spec import RunSpec, SpecError, StudySpec


MANIFEST_COLUMNS = (
    "condition",
    "family",
    "schedule",
    "feedback",
    "half_life",
    "seed",
    "target_frames",
    "checkpoint_frames",
    "checkpoint",
    "run_dir",
    "label_suffix",
)


@dataclass(frozen=True)
class CheckpointRecord:
    run_name: str
    target_frames: int
    checkpoint_frames: int
    checkpoint: Path
    run_dir: Path


def _under_workspace(path: Path, workspace_root: str) -> bool:
    try:
        PurePosixPath(str(path)).relative_to(PurePosixPath(workspace_root))
        return path.is_absolute()
    except ValueError:
        return False


def _render_fields(run: RunSpec, templates: Mapping[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for name in ("family", "schedule", "feedback", "half_life"):
        template = templates.get(name, "")
        if not isinstance(template, str):
            raise SpecError(f"telemetry.manifest_fields.{name} must be a string")
        try:
            fields[name] = template.format_map(dict(run.context))
        except KeyError as error:
            raise SpecError(
                f"telemetry field {name!r} references unknown field {error.args[0]!r}"
            ) from error
    return fields


def build_place_field_manifests(
    study: StudySpec,
    inventory: Iterable[CheckpointRecord],
    require_checkpoint_files: bool = True,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    telemetry = study.telemetry
    if telemetry.get("protocol") != "dg-place-fields-v1":
        raise SpecError("telemetry.protocol must be 'dg-place-fields-v1'")
    target_frames = telemetry.get("target_frames")
    if not isinstance(target_frames, list) or not target_frames or not all(
        isinstance(value, int) and value > 0 for value in target_frames
    ):
        raise SpecError("telemetry.target_frames must be a nonempty integer array")
    if sorted(set(target_frames)) != target_frames:
        raise SpecError("telemetry.target_frames must be sorted and unique")
    trajectory_seed = telemetry.get("trajectory_seed", 99)
    terminal_seeds = telemetry.get("terminal_seeds", [8, 123])
    if not isinstance(trajectory_seed, int) or not isinstance(terminal_seeds, list):
        raise SpecError("telemetry trajectory_seed and terminal_seeds are invalid")
    selected_seeds = {trajectory_seed, *terminal_seeds}
    if not selected_seeds.issubset(set(study.seeds)):
        raise SpecError("telemetry seeds must be present in study.seeds")

    by_run_target: dict[tuple[str, int], CheckpointRecord] = {}
    for record in inventory:
        key = (record.run_name, record.target_frames)
        if key in by_run_target:
            raise SpecError(f"duplicate checkpoint inventory row for {key!r}")
        by_run_target[key] = record

    templates = telemetry.get("manifest_fields", {})
    if not isinstance(templates, Mapping):
        raise SpecError("telemetry.manifest_fields must be an object")
    label_template = telemetry.get(
        "label_template", "{condition}__s{seed}__f{checkpoint_frames}"
    )
    if not isinstance(label_template, str):
        raise SpecError("telemetry.label_template must be a string")

    rows: list[dict[str, str]] = []
    trajectory_rows: list[dict[str, str]] = []
    intervention_runs = {run.name for run in selected_intervention_runs(study)}
    intervention_targets = (telemetry.get("intervention") or {}).get("target_frames", [])
    for run in study.expand_runs():
        if run.seed not in selected_seeds and run.name not in intervention_runs:
            continue
        selected_targets = target_frames if run.seed == trajectory_seed else ([target_frames[-1]] if run.seed in selected_seeds else [])
        if run.name in intervention_runs:
            selected_targets = sorted(set(selected_targets) | set(intervention_targets))
        for target in selected_targets:
            key = (run.name, target)
            if key not in by_run_target:
                raise SpecError(f"checkpoint inventory is missing {key!r}")
            checkpoint = by_run_target[key]
            for path_name, path in (("checkpoint", checkpoint.checkpoint), ("run_dir", checkpoint.run_dir)):
                if not _under_workspace(path, study.workspace_root):
                    raise SpecError(f"telemetry {path_name} is outside the workspace: {path}")
            if require_checkpoint_files and not checkpoint.checkpoint.is_file():
                raise SpecError(f"telemetry checkpoint does not exist: {checkpoint.checkpoint}")
            context = {
                **run.context,
                "condition": run.condition,
                "target_frames": target,
                "checkpoint_frames": checkpoint.checkpoint_frames,
            }
            try:
                label = label_template.format_map(context)
            except KeyError as error:
                raise SpecError(
                    f"telemetry.label_template references unknown field {error.args[0]!r}"
                ) from error
            row = {
                "condition": run.condition,
                **_render_fields(run, templates),
                "seed": str(run.seed),
                "target_frames": str(target),
                "checkpoint_frames": str(checkpoint.checkpoint_frames),
                "checkpoint": str(checkpoint.checkpoint),
                "run_dir": str(checkpoint.run_dir),
                "label_suffix": label,
            }
            rows.append(row)
            if run.seed == trajectory_seed:
                trajectory_rows.append(row)
    labels = [row["label_suffix"] for row in rows]
    if len(set(labels)) != len(labels):
        raise SpecError("telemetry labels are not unique")
    return rows, trajectory_rows


def selected_intervention_runs(study: StudySpec) -> list[RunSpec]:
    intervention = study.telemetry.get("intervention")
    if intervention is None:
        return []
    if not isinstance(intervention, Mapping):
        raise SpecError("telemetry.intervention must be an object")
    where = intervention.get("where", {})
    if not isinstance(where, Mapping):
        raise SpecError("telemetry.intervention.where must be an object")
    runs = study.expand_runs()
    for key in where:
        if any(key not in run.context for run in runs):
            raise SpecError(f"unknown intervention selection field {key!r}")
    selected = [run for run in runs if all(run.context[k] == value for k, value in where.items())]
    if not selected:
        raise SpecError("intervention selection is empty")
    return selected


def build_intervention_manifest(
    study: StudySpec,
    rows: Iterable[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Select the declared intervention checkpoint once for every study run.

    The intervention manifest deliberately reuses the checkpoint inventory and
    row contract of the standard place-field manifest.  This prevents an
    evaluator from silently choosing a different run directory or checkpoint.
    """

    intervention = study.telemetry.get("intervention")
    if intervention is None:
        return []
    if not isinstance(intervention, Mapping):
        raise SpecError("telemetry.intervention must be an object")
    if intervention.get("protocol") != "target-control-intervention-v1":
        raise SpecError(
            "telemetry.intervention.protocol must be "
            "'target-control-intervention-v1'"
        )
    target_frames = intervention.get("target_frames")
    if (
        not isinstance(target_frames, list)
        or len(target_frames) != 1
        or not isinstance(target_frames[0], int)
        or target_frames[0] <= 0
    ):
        raise SpecError(
            "telemetry.intervention.target_frames must contain one positive integer"
        )
    target = str(target_frames[0])
    expected = {
        (run.condition, str(run.seed))
        for run in selected_intervention_runs(study)
    }
    selected = [dict(row) for row in rows if row.get("target_frames") == target
                and (row.get("condition"), row.get("seed")) in expected]
    observed = [(row.get("condition"), row.get("seed")) for row in selected]
    if len(observed) != len(set(observed)):
        raise SpecError("intervention manifest contains duplicate condition/seed rows")
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        unexpected = sorted(set(observed) - expected)
        raise SpecError(
            "intervention rows differ from the declared study; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return selected


def discover_nemo_checkpoints(study: StudySpec, batch_root: Path) -> list[CheckpointRecord]:
    """Use the authoritative NEMO2 checkpoint selector for every expected run."""

    try:
        from sf_working_directories.IntrMotiv.evaluation.build_place_field_sweep import (
            checkpoint_frames,
            select_checkpoints,
        )
    except ImportError as error:
        raise RuntimeError(
            "render-telemetry must run from the NEMO2 SF_hipposlam checkout"
        ) from error

    inventory: list[CheckpointRecord] = []
    run_directories = discover_run_directories(study, batch_root)
    for run in study.expand_runs():
        run_dir = run_directories[run.name]
        for target, checkpoint in select_checkpoints(run_dir):
            inventory.append(CheckpointRecord(
                run_name=run.name,
                target_frames=int(target),
                checkpoint_frames=int(checkpoint_frames(checkpoint)),
                checkpoint=Path(checkpoint),
                run_dir=run_dir,
            ))
    return inventory


def write_manifest(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
