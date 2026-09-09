"""Audit Sample Factory launcher manifests against a StudySpec."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path, PurePosixPath
import shlex
from typing import Any

from .spec import SpecError, StudySpec


REQUIRED_COLUMNS = {
    "job_id",
    "status",
    "experiment",
    "train_root",
    "sbatch_file",
    "stdout",
    "stderr",
    "command",
}


def _normalized_experiment(name: str) -> str:
    return name.removeprefix("00_")


def _under_workspace(value: str, workspace_root: str) -> bool:
    path = PurePosixPath(value)
    try:
        path.relative_to(PurePosixPath(workspace_root))
        return path.is_absolute()
    except ValueError:
        return False


def _flag_values(command: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in shlex.split(command):
        if token.startswith("--") and "=" in token:
            flag, value = token.split("=", 1)
            if flag in values:
                raise SpecError(f"submitted command contains duplicate flag {flag!r}")
            values[flag] = value
    return values


def audit_submission(
    study: StudySpec,
    jobs_tsv: Path,
    require_submitted: bool = False,
) -> dict[str, Any]:
    with jobs_tsv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - fields
        if missing_columns:
            raise SpecError(f"jobs.tsv is missing columns {sorted(missing_columns)!r}")
        rows = list(reader)

    expected = {run.name: run for run in study.expand_runs()}
    observed_names = [_normalized_experiment(row["experiment"]) for row in rows]
    if len(observed_names) != len(set(observed_names)):
        raise SpecError("jobs.tsv contains duplicate experiment rows")
    if set(observed_names) != set(expected):
        raise SpecError(
            "jobs.tsv run matrix differs from the study; "
            f"missing={sorted(set(expected) - set(observed_names))}, "
            f"unexpected={sorted(set(observed_names) - set(expected))}"
        )

    job_ids: list[str] = []
    for row, run_name in zip(rows, observed_names):
        run = expected[run_name]
        for field in ("sbatch_file", "stdout", "stderr"):
            if not _under_workspace(row[field], study.workspace_root):
                raise SpecError(f"{run_name} {field} is outside the workspace: {row[field]}")
        flags = _flag_values(row["command"])
        if flags.get("--experiment") not in {run.name, f"00_{run.name}"}:
            raise SpecError(f"{run_name} command has a mismatched --experiment")
        train_dir = flags.get("--train_dir")
        if train_dir is None or not _under_workspace(train_dir, study.workspace_root):
            raise SpecError(f"{run_name} command has no workspace --train_dir")
        submitted_tokens = set(shlex.split(row["command"]))
        missing_args = [arg for arg in run.args if arg not in submitted_tokens]
        if missing_args:
            raise SpecError(f"{run_name} command is missing study arguments {missing_args!r}")
        if require_submitted:
            if row["status"] != "submitted" or not row["job_id"].isdigit():
                raise SpecError(f"{run_name} is not recorded as submitted with a numeric job ID")
        if row["job_id"]:
            job_ids.append(row["job_id"])
    if len(job_ids) != len(set(job_ids)):
        raise SpecError("jobs.tsv contains duplicate nonempty job IDs")

    return {
        **study.provenance(),
        "jobs_tsv": str(jobs_tsv),
        "rows": len(rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "job_ids": job_ids,
        "submitted_complete": len(job_ids) == study.expected_runs
        and all(row["status"] == "submitted" for row in rows),
        "commands_match_study": True,
        "workspace_paths_valid": True,
    }

