"""Exact, one-pass discovery of directories declared by a StudySpec."""

from __future__ import annotations

from pathlib import Path

from .spec import SpecError, StudySpec


def discover_run_directories(study: StudySpec, batch_root: Path) -> dict[str, Path]:
    expected = {run.name for run in study.expand_runs()}
    accepted_names = {
        candidate: run_name
        for run_name in expected
        for candidate in (run_name, f"00_{run_name}")
    }
    found: dict[str, list[Path]] = {run_name: [] for run_name in expected}
    for path in batch_root.rglob("*"):
        if path.is_dir() and path.name in accepted_names:
            found[accepted_names[path.name]].append(path)
    errors = {
        run_name: paths for run_name, paths in found.items() if len(paths) != 1
    }
    if errors:
        details = ", ".join(
            f"{run_name}={len(paths)}" for run_name, paths in sorted(errors.items())
        )
        raise SpecError(f"expected exactly one directory per declared run; {details}")
    return {run_name: paths[0] for run_name, paths in found.items()}

