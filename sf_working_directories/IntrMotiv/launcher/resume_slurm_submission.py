#!/usr/bin/env python3
"""Resume an interrupted IntrMotiv Sample Factory Slurm submission safely."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

JOB_FIELDS = ["job_id", "status", "experiment", "train_root", "sbatch_file", "stdout", "stderr", "command"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path, help="Existing Sample Factory Slurm work directory")
    parser.add_argument("--submit", action="store_true", help="Actually submit pending jobs")
    parser.add_argument("--pause-between", type=float, default=1.0, help="Seconds to wait between sbatch calls")
    return parser.parse_args()


def atomic_write(path: Path, content: str, mode: int | None = None) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary_path = Path(stream.name)
    if mode is not None:
        temporary_path.chmod(mode)
    os.replace(temporary_path, path)


def write_manifest(workdir: Path, jobs: list[dict[str, str]], submission: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=workdir, newline="", delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=JOB_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(jobs)
        temporary_jobs = Path(stream.name)
    os.replace(temporary_jobs, workdir / "jobs.tsv")

    submission["jobs"] = [
        {"job_id": job["job_id"], "status": job["status"], "experiment": job["experiment"]} for job in jobs
    ]
    atomic_write(workdir / "submission.json", json.dumps(submission, indent=2) + "\n")

    job_ids = [job["job_id"] for job in jobs if job["status"] == "submitted" and job["job_id"]]
    scancel = "#!/bin/bash\n"
    scancel += f"scancel {' '.join(job_ids)}\n" if job_ids else "# No jobs were submitted.\n"
    atomic_write(workdir / "scancel.sh", scancel, mode=0o750)


def sbatch_command(job: dict[str, str], launcher_args: dict) -> list[str]:
    cpus = launcher_args.get("slurm_cpus_per_job")
    if cpus is None:
        cpus = launcher_args["slurm_cpus_per_gpu"] * launcher_args["slurm_gpus_per_job"]
        if cpus == 0:
            cpus = launcher_args["slurm_cpus_per_gpu"]

    command = ["sbatch"]
    partition = launcher_args.get("slurm_partition")
    if partition:
        command.extend(["-p", partition])
    command.extend(
        [
            f"--gres=gpu:{launcher_args['slurm_gpus_per_job']}",
            "-c",
            str(cpus),
            "--parsable",
            "--output",
            job["stdout"],
        ]
    )
    if launcher_args.get("slurm_separate_stderr"):
        command.extend(["--error", job["stderr"]])
    command.append(job["sbatch_file"])
    return command


def main() -> int:
    args = parse_args()
    workdir = args.workdir.resolve()
    jobs_path = workdir / "jobs.tsv"
    submission_path = workdir / "submission.json"
    if not jobs_path.is_file() or not submission_path.is_file():
        raise FileNotFoundError("workdir must contain jobs.tsv and submission.json")

    with jobs_path.open(encoding="utf-8", newline="") as stream:
        jobs = list(csv.DictReader(stream, delimiter="\t"))
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    pending = [job for job in jobs if job["status"] == "pending_submission"]
    print(f"{len(pending)} pending, {len(jobs) - len(pending)} already recorded in {workdir}")
    if not args.submit:
        for job in pending:
            print(job["experiment"])
        return 0

    for job in pending:
        command = sbatch_command(job, submission["launcher_args"])
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            job["status"] = "submission_failed"
            job["command"] += f"\n# sbatch stderr: {result.stderr.strip()}"
            write_manifest(workdir, jobs, submission)
            print(f"Submission failed for {job['experiment']}: {result.stderr.strip()}")
            return 1

        job["job_id"] = result.stdout.strip().split(";", 1)[0]
        job["status"] = "submitted"
        write_manifest(workdir, jobs, submission)
        print(f"Submitted {job['experiment']} as {job['job_id']}")
        time.sleep(args.pause_between)

    print("All pending jobs submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
