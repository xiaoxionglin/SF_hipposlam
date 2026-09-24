"""Submit missing GPU shard jobs when the account's Slurm quota permits.

The launcher has already rendered and audited the full shard. This helper only
submits rows that lack a job ID, preserving their reviewed sbatch scripts.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def read_rows(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return reader.fieldnames, list(reader)


def write_rows(path, fields, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def submit_missing(workdir):
    manifest = workdir / "jobs.tsv"
    fields, rows = read_rows(manifest)
    for row in rows:
        if row["status"] == "submitted" and row["job_id"]:
            continue
        name = row["experiment"]
        # A previous process may have been interrupted after sbatch succeeded.
        found = subprocess.run(
            ["squeue", "-u", subprocess.check_output(["whoami"], text=True).strip(),
             "-h", "-o", "%i %j"], text=True, capture_output=True, check=True
        ).stdout
        existing = [line.split()[0] for line in found.splitlines()
                    if len(line.split()) >= 2 and line.split()[1] == name]
        if len(existing) > 1:
            raise RuntimeError(f"Duplicate queued jobs for {name}: {existing}")
        if existing:
            job_id = existing[0]
        else:
            command = [
                "sbatch", "-p", "rtx,l40s", "--gres=gpu:1", "-c", "40", "--parsable",
                "--output", str(workdir / "logs" / f"{name}-slurm-%j.out"),
                "--error", str(workdir / "logs" / f"{name}-slurm-%j.err"),
                row["sbatch_file"],
            ]
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode:
                print(f"Waiting for Slurm quota: {result.stderr.strip()}", flush=True)
                break
            job_id = result.stdout.strip().split(";")[0]
        row["job_id"] = job_id
        row["status"] = "submitted"
        write_rows(manifest, fields, rows)
        print(f"Submitted {name}: {job_id}", flush=True)
    return all(row["status"] == "submitted" and row["job_id"] for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    while not submit_missing(args.workdir):
        time.sleep(args.interval)
    submission = args.workdir / "submission.json"
    if submission.exists():
        data = json.loads(submission.read_text())
        _, rows = read_rows(args.workdir / "jobs.tsv")
        data["jobs"] = [dict(job_id=row["job_id"], status=row["status"],
                             experiment=row["experiment"]) for row in rows]
        submission.write_text(json.dumps(data, indent=2) + "\n")
    root = args.workdir.parent
    subprocess.run(
        [sys.executable, "-m", "hpc_runs.combine_fixed_reward_manifests",
         "hpc_runs/studies/fixed_reward_dg_peak_transfer_20260924.study.json",
         str(root / "combined_submitted" / "jobs.tsv"),
         str(root / "cpu_submitted" / "jobs.tsv"),
         str(args.workdir / "jobs.tsv"), "--submitted"],
        check=True,
    )
    print("GPU shard fully submitted", flush=True)


if __name__ == "__main__":
    main()
