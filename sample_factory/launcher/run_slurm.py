"""Run experiment grids through Slurm."""

import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from os.path import join
from string import Template

from sample_factory.utils.utils import log, str2bool

SBATCH_TEMPLATE_DEFAULT = "#!/bin/bash\nconda activate sf2\ncd ~/sample-factory\n"


def add_slurm_args(parser):
    parser.add_argument("--slurm_gpus_per_job", default=1, type=int, help="GPUs in a single Slurm job")
    parser.add_argument(
        "--slurm_cpus_per_gpu",
        default=14,
        type=int,
        help="Legacy CPU resource setting. Also acts as CPUs per job when no GPUs are requested.",
    )
    parser.add_argument(
        "--slurm_cpus_per_job",
        default=None,
        type=int,
        help="CPU cores per job. Overrides --slurm_cpus_per_gpu when provided.",
    )
    parser.add_argument(
        "--slurm_memory",
        default="80G",
        type=str,
        help="Memory request exposed to sbatch templates as $MEMORY.",
    )
    parser.add_argument(
        "--slurm_print_only", default=False, type=str2bool, help="Generate scripts without submitting jobs"
    )
    parser.add_argument(
        "--slurm_workdir",
        default=None,
        type=str,
        help="Directory for generated scripts, submission metadata, and cancellation commands.",
    )
    parser.add_argument(
        "--slurm_log_dir",
        default=None,
        type=str,
        help="Directory for Slurm stdout/stderr. Defaults to --slurm_workdir.",
    )
    parser.add_argument(
        "--slurm_separate_stderr",
        default=False,
        type=str2bool,
        help="Write stderr to a separate per-job file instead of combining it with stdout.",
    )
    parser.add_argument(
        "--slurm_partition",
        default=None,
        type=str,
        help='Adds a Slurm partition, e.g. "gpu".',
    )
    parser.add_argument(
        "--slurm_sbatch_template",
        default=None,
        type=str,
        help="Template containing environment setup and the experiment command.",
    )
    parser.add_argument(
        "--slurm_timeout",
        default="0",
        type=str,
        help="Slurm walltime. Defaults to 0, which does not time out the job.",
    )
    return parser


def _git_metadata():
    result = {}
    for key, command in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("status", ["git", "status", "--short"]),
    ):
        proc = subprocess.run(command, text=True, capture_output=True, check=False)
        if proc.returncode == 0:
            result[key] = proc.stdout.strip()
    return result


def _write_jobs(path, jobs):
    fields = ["job_id", "status", "experiment", "train_root", "sbatch_file", "stdout", "stderr", "command"]
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(jobs)


def _write_scancel(path, job_ids):
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("#!/bin/bash\n")
        if job_ids:
            stream.write(f"scancel {' '.join(job_ids)}\n")
        else:
            stream.write("# No jobs were submitted.\n")
    os.chmod(path, 0o750)


def _write_submission(path, run_description, args, template_text, jobs, created_at, git_metadata):
    payload = {
        "run_name": run_description.run_name,
        "created_at": created_at,
        "launcher_args": vars(args),
        "template_sha256": hashlib.sha256(template_text.encode("utf-8")).hexdigest(),
        "git": git_metadata,
        "jobs": [{"job_id": job["job_id"], "status": job["status"], "experiment": job["experiment"]} for job in jobs],
    }
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)
        stream.write("\n")


def run_slurm(run_description, args):
    workdir = os.path.abspath(args.slurm_workdir)
    configured_log_dir = args.slurm_log_dir
    log_dir = os.path.abspath(configured_log_dir or workdir)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.slurm_sbatch_template is not None:
        with open(args.slurm_sbatch_template, "r", encoding="utf-8") as template_file:
            sbatch_template = template_file.read()
    else:
        sbatch_template = SBATCH_TEMPLATE_DEFAULT

    partition_directive = f"-p {args.slurm_partition} " if args.slurm_partition else ""
    legacy_num_cpus = args.slurm_cpus_per_gpu * args.slurm_gpus_per_job
    if legacy_num_cpus == 0:
        legacy_num_cpus = args.slurm_cpus_per_gpu
    num_cpus = args.slurm_cpus_per_job if args.slurm_cpus_per_job is not None else legacy_num_cpus

    generated = list(run_description.generate_experiments(args.train_dir, makedirs=not args.slurm_print_only))
    created_at = datetime.now(timezone.utc).isoformat()
    git_metadata = _git_metadata()
    jobs = []
    for command, name, train_root, _env_vars in generated:
        sbatch_file = os.path.abspath(join(workdir, f"sbatch_{name}.sh"))
        if configured_log_dir:
            stdout_path = join(log_dir, f"{name}-slurm-%j.out")
        else:
            stdout_path = join(log_dir, f"{os.path.basename(sbatch_file)}-slurm-%j.out")
        stderr_path = join(log_dir, f"{name}-slurm-%j.err") if args.slurm_separate_stderr else ""
        file_content = Template(sbatch_template).substitute(
            CMD=command,
            FILENAME=sbatch_file,
            PARTITION=partition_directive,
            GPU=args.slurm_gpus_per_job,
            CPU=num_cpus,
            MEMORY=args.slurm_memory,
            TIMEOUT=args.slurm_timeout,
            NAME=name,
            LOG_DIR=log_dir,
        )
        with open(sbatch_file, "w", encoding="utf-8") as stream:
            stream.write(file_content)
        jobs.append(
            {
                "job_id": "",
                "status": "generated" if args.slurm_print_only else "pending_submission",
                "experiment": name,
                "train_root": train_root,
                "sbatch_file": sbatch_file,
                "stdout": stdout_path,
                "stderr": stderr_path,
                "command": command,
            }
        )

    jobs_path = join(workdir, "jobs.tsv")
    submission_path = join(workdir, "submission.json")
    scancel_path = join(workdir, "scancel.sh")
    _write_jobs(jobs_path, jobs)
    _write_scancel(scancel_path, [])
    _write_submission(submission_path, run_description, args, sbatch_template, jobs, created_at, git_metadata)

    if args.slurm_print_only:
        log.info("Generated %d Slurm scripts in %s (print-only)", len(jobs), workdir)
        return 0

    job_ids = []
    for job in jobs:
        command = ["sbatch"]
        if args.slurm_partition:
            command.extend(["-p", args.slurm_partition])
        command.append(f"--gres=gpu:{args.slurm_gpus_per_job}")
        command.extend(["-c", str(num_cpus), "--parsable", "--output", job["stdout"]])
        if args.slurm_separate_stderr:
            command.extend(["--error", job["stderr"]])
        command.append(job["sbatch_file"])
        log.info("Executing %s", " ".join(command))

        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            job["status"] = "submission_failed"
            job["command"] += f"\n# sbatch stderr: {result.stderr.strip()}"
            _write_jobs(jobs_path, jobs)
            _write_submission(submission_path, run_description, args, sbatch_template, jobs, created_at, git_metadata)
            log.error("sbatch failed for %s: %s", job["experiment"], result.stderr.strip())
            return 1

        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id:
            job["status"] = "submission_failed"
            _write_jobs(jobs_path, jobs)
            _write_submission(submission_path, run_description, args, sbatch_template, jobs, created_at, git_metadata)
            log.error("sbatch returned no job ID for %s", job["experiment"])
            return 1
        job["job_id"] = job_id
        job["status"] = "submitted"
        job_ids.append(job_id)
        _write_jobs(jobs_path, jobs)
        _write_scancel(scancel_path, job_ids)
        _write_submission(submission_path, run_description, args, sbatch_template, jobs, created_at, git_metadata)
        time.sleep(args.pause_between)

    log.info("Submitted %d jobs. Logs: %s", len(job_ids), log_dir)
    log.info("Cancel this submission with %s", scancel_path)
    return 0
