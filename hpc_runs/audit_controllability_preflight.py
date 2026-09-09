"""Audit the 16-cell seed-99 controllability preflight launcher plan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import shlex

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.spec import SpecError


def _preflight_args(study, run) -> list[str]:
    result = []
    for arg in run.args:
        if arg == "--train_for_env_steps=75000000":
            arg = "--train_for_env_steps=1000000"
        elif arg == f"--wandb_group={study.batch_name}":
            arg = f"--wandb_group={study.batch_name}_preflight"
        result.append(arg)
    return result


def audit_preflight(study, jobs_tsv: Path, require_submitted: bool = False) -> dict:
    with jobs_tsv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = {run.name: run for run in study.expand_runs() if run.seed == 99}
    observed: dict[str, dict[str, str]] = {}
    workspace = PurePosixPath(study.workspace_root)
    for row in rows:
        experiment = row.get("experiment", "")
        prefix = "00_PF_"
        if not experiment.startswith(prefix):
            raise SpecError(f"unexpected preflight experiment {experiment!r}")
        run_name = experiment[len(prefix):]
        if run_name not in expected or run_name in observed:
            raise SpecError(f"unexpected or duplicate preflight run {run_name!r}")
        tokens = shlex.split(row.get("command", ""))
        flags = [token.split("=", 1)[0] for token in tokens if token.startswith("--")]
        if len(flags) != len(set(flags)):
            raise SpecError(f"preflight run {run_name!r} contains duplicate CLI flags")
        missing = [arg for arg in _preflight_args(study, expected[run_name]) if arg not in tokens]
        if missing:
            raise SpecError(f"preflight run {run_name!r} is missing arguments {missing!r}")
        train_dirs = [token.split("=", 1)[1] for token in tokens if token.startswith("--train_dir=")]
        if len(train_dirs) != 1:
            raise SpecError(f"preflight run {run_name!r} must declare one train_dir")
        try:
            PurePosixPath(train_dirs[0]).relative_to(workspace)
        except ValueError as error:
            raise SpecError(f"preflight train_dir is outside workspace: {train_dirs[0]}") from error
        status = row.get("status", "")
        if require_submitted and (status != "submitted" or not row.get("job_id", "").isdigit()):
            raise SpecError(f"preflight run {run_name!r} is not submitted with a numeric job ID")
        observed[run_name] = row
    if set(observed) != set(expected):
        raise SpecError(
            f"preflight matrix mismatch; missing={sorted(set(expected) - set(observed))}, "
            f"unexpected={sorted(set(observed) - set(expected))}"
        )
    factor_balance = {
        factor.name: {str(level.value): 0 for level in factor.levels}
        for factor in study.factors
    }
    for run in expected.values():
        for factor, level in run.factors.items():
            factor_balance[factor][str(level)] += 1
    return {
        **study.provenance(),
        "jobs_tsv": str(jobs_tsv),
        "preflight_rows": len(rows),
        "commands_match_study": True,
        "workspace_paths_valid": True,
        "factor_balance": factor_balance,
        "status_counts": {
            status: sum(row.get("status") == status for row in rows)
            for status in sorted({row.get("status", "") for row in rows})
        },
        "job_ids": [row["job_id"] for row in rows if row.get("job_id", "").isdigit()],
        "submitted_complete": all(row.get("status") == "submitted" for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("jobs_tsv", type=Path)
    parser.add_argument("--submitted", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit_preflight(load_study(args.study), args.jobs_tsv, args.submitted)
    except (OSError, SpecError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
