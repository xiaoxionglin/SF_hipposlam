# IntrMotiv Batch Submission on NEMO2

This guide describes the supported way to define, inspect, submit, monitor, and
cancel IntrMotiv training batches. It is intended for students, collaborators,
and coding agents working in the shared Sample Factory repository.

## 1. Design of the launcher

The workflow keeps three concerns separate:

1. A Python run-description module defines the scientific experiment matrix.
2. The IntrMotiv wrapper defines the normal NEMO2 resource profile.
3. Sample Factory's existing Slurm backend generates and submits one independent
   Slurm job per run.

The relevant files are:

```text
sf_working_directories/IntrMotiv/
|-- BATCH_SUBMISSION.md
|-- launcher/
|   |-- launch_nemo2.sh
|   `-- LAUNCHER.md
`-- dmlab/experiments/
    |-- hrl_intrinsic_arch_search.py
    `-- nemo2_sfgit_intrmotiv.sh
```

For a new batch, normally create or edit only a run-description module under
`sf_working_directories/IntrMotiv/dmlab/experiments/`. Do not copy or edit the
launcher and Slurm template for every batch.

The wrapper currently requests:

| Setting | Default | Reason |
| --- | ---: | --- |
| Partition | `cpu` | Same partition used by the established DMLab runs |
| CPUs per job | `40` | Supports 32 Sample Factory workers |
| Memory per job | `80G` | Previous comparable jobs peaked at about 50 GB |
| Wall time | `30:00:00` | Expected 20-40 hour runtime with better queue concurrency |
| GPUs | `0` | DMLab software rendering and CPU training |
| Environment | `SFgit` | Shared environment containing all dependencies |

These are submission defaults, not model hyperparameters.

The shared IntrMotiv Slurm template routes runtime-generated data to
`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime`: per-job
`TMPDIR`, XDG/Torch cache, Matplotlib config, and W&B cache/staging. Set
`INTRMOTIV_RUNTIME_ROOT` before launching only when a different allocated
workspace is required. DMLab's level cache is a model CLI setting rather than a
template setting; every experiment using `--dmlab_use_level_cache=True` must
also set `--dmlab_level_cache_path` to a workspace path. Confirm the resolved
value in a preflight `config.json`.

## 2. Prerequisites

Log in to NEMO2 and enter the repository:

```bash
ssh nemo2
cd ~/SF_git_XXL/SF_hipposlam
```

The launcher uses this interpreter by default:

```text
/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python
```

You do not need to activate the environment before using the wrapper. Each
generated Slurm script activates `SFgit` before starting training.

Before preparing a batch, check the worktree and current queue:

```bash
git status --short
squeue -u "$USER"
```

A dirty worktree does not block submission, but `submission.json` records its
state. Know which code the jobs will run. Do not alter unrelated user changes.

## 3. Define a batch

Use an existing IntrMotiv experiment module as the starting pattern. Give every
batch a unique, descriptive `BATCH_NAME`:

```python
from sample_factory.launcher.run_description import Experiment, RunDescription


BATCH_NAME = "intrmotiv_hrl_batch2_20260819"
SEEDS = [8, 99, 123, 456]
SEQUENCE_LENGTHS = [32, 64, 128]
REWARD_METHODS = ["punish", "encourage", "mean"]

BASE_CLI = (
    "--env=openfield_map2_fixed_loc3_noreward "
    "--train_for_env_steps=25000000 "
    "--with_wandb=True "
    "--wandb_project=SF_HRL_Intrinsic_ArchSearch "
    f"--wandb_group={BATCH_NAME} "
    # Add the remaining shared training arguments here.
)


def experiment(seed: int, seq_len: int, reward_method: str) -> Experiment:
    name = f"B2_F16_L{seq_len}_ER{reward_method}_S{seed}"
    cli = (
        BASE_CLI
        + f"--seed={seed} "
        + f"--Hippo_L={seq_len} "
        + f"--encoder_reward_method={reward_method} "
    )
    return Experiment(name, cli, [{}])


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[
        experiment(seed, seq_len, reward_method)
        for seed in SEEDS
        for seq_len in SEQUENCE_LENGTHS
        for reward_method in REWARD_METHODS
    ],
)
```

Before submission, verify that `train_dir`, checkpoints, W&B local data, Slurm
logs, caches, and temporary environment data all resolve into an allocated
workspace. Do not run training against paths under the home filesystem.
The IntrMotiv launcher defaults to
`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir`; override it
only when necessary with `INTRMOTIV_TRAIN_ROOT`.

Use the following naming levels consistently:

| Level | Purpose | Example |
| --- | --- | --- |
| W&B project | Long-lived research line | `SF_HRL_Intrinsic_ArchSearch` |
| W&B group | One coherent batch | `intrmotiv_hrl_batch2_20260819` |
| Run name | Exact condition and seed | `B2_F16_L64_ERmean_S99` |

Every run name must be unique within the batch. Include all swept values and the
seed in the name. Avoid relying only on the generated list index.

The number of jobs is the Cartesian product of the sweep dimensions. In the
example above:

```text
4 seeds * 3 sequence lengths * 3 reward methods = 36 jobs
```

Check this count before submission. Large accidental products are a common and
expensive error.

## 4. Preflight without submitting

Pass the importable Python module name, without `.py`, to the wrapper:

```bash
sf_working_directories/IntrMotiv/launcher/launch_nemo2.sh \
  sf_working_directories.IntrMotiv.dmlab.experiments.hrl_intrinsic_arch_search \
  --print-only
```

`--print-only` is the default if no mode is supplied, but spelling it out is
recommended. It generates scripts and metadata without calling `sbatch` and
without creating Sample Factory training directories.

The launcher prints a work directory of this form:

```text
train_dir/_slurm/<BATCH_NAME>/<UTC timestamp>/
```

Save it in a shell variable for inspection:

```bash
WORKDIR=$(find train_dir/_slurm/<BATCH_NAME> \
  -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
echo "$WORKDIR"
```

The directory contains:

| File | Meaning |
| --- | --- |
| `sbatch_<run>.sh` | Exact Slurm script for one run |
| `jobs.tsv` | One row per run with status, paths, and command |
| `submission.json` | Launcher arguments, template hash, git state, and jobs |
| `logs/` | Destination for separate stdout and stderr files |
| `scancel.sh` | Exact cancellation commands after a real submission |

Perform at least these checks:

```bash
# jobs.tsv has one header row, so this should equal expected jobs + 1.
wc -l "$WORKDIR/jobs.tsv"

# Count generated scripts.
find "$WORKDIR" -maxdepth 1 -name 'sbatch_*.sh' | wc -l

# Inspect the manifest without losing long command lines.
column -ts $'\t' "$WORKDIR/jobs.tsv" | less -S

# Inspect one complete generated job.
SCRIPT=$(find "$WORKDIR" -maxdepth 1 -name 'sbatch_*.sh' | sort | head -n 1)
sed -n '1,160p' "$SCRIPT" | less

# Inspect recorded launcher and git metadata.
less "$WORKDIR/submission.json"
```

Verify all of the following before submission:

- Job count equals the intended Cartesian product.
- Every run name is unique and identifies its condition and seed.
- The environment, map, encoder, model flags, and training duration are correct.
- `--with_wandb=True`, project, group, and account are correct.
- Slurm partition, CPU count, memory, and timeout are appropriate.
- The generated training roots point to the intended batch.
- There are no commands writing into another researcher's working directory.

Do not edit the run-description module between preflight and submission unless
you repeat the preflight.

Sample Factory applies `--train_for_env_steps` per policy: its runner stops
when every policy has crossed the threshold. For a four-policy PBT population,
use `--train_for_env_steps=25000000` to target approximately 100M aggregate
environment steps. Report the population as one replicate and average the four
policy series; do not interpret the four policy thresholds as four independent
replicates. Request roughly 30 hours per allocation to improve scheduling
concurrency. If a job reaches the
wall-time limit first, submit the same run description again with the default
`restart_behavior=resume`; Sample Factory will continue from the existing
checkpoint.

## 5. Submit the batch

Run the same command with `--submit`:

```bash
sf_working_directories/IntrMotiv/launcher/launch_nemo2.sh \
  sf_working_directories.IntrMotiv.dmlab.experiments.hrl_intrinsic_arch_search \
  --submit
```

The launcher submits one independent job per experiment. There is no local
concurrency throttle. Slurm decides how many jobs run simultaneously based on
cluster-wide priorities and available resources.

Submission takes roughly one second per job because the launcher deliberately
pauses between `sbatch` calls. Do not interrupt it. A scheduler rejection causes
the launcher to stop with a nonzero exit status rather than silently continuing.

### Resume an interrupted submission

If the launcher itself is interrupted after some `sbatch` calls, do **not** run
the original command again: it creates a new manifest and duplicates the jobs
already accepted by Slurm. First make sure the original launcher process is no
longer running, then inspect the existing work directory:

```bash
WORKDIR=train_dir/_slurm/<BATCH_NAME>/<submission-time>
python sf_working_directories/IntrMotiv/launcher/resume_slurm_submission.py \
  "$WORKDIR"
```

The dry run prints only rows whose status is `pending_submission`. To submit
exactly those rows and update `jobs.tsv`, `submission.json`, and `scancel.sh`
after every accepted Slurm job, use:

```bash
python sf_working_directories/IntrMotiv/launcher/resume_slurm_submission.py \
  "$WORKDIR" --submit
```

This IntrMotiv helper reuses the generated sbatch scripts and original resource
arguments. It never regenerates a run description, so submitted rows are not
duplicated. It stops on the first failed submission and leaves an updated
manifest for a subsequent retry.

After submission, find the new work directory and verify the manifest:

```bash
WORKDIR=$(find train_dir/_slurm/<BATCH_NAME> \
  -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)

awk -F '\t' '
  NR > 1 { count[$2]++; first = first ? first : $1; last = $1 }
  END {
    for (status in count) print status, count[status]
    print "first job", first
    print "last job", last
  }
' "$WORKDIR/jobs.tsv"
```

For a complete submission, every row should have status `submitted` and a Slurm
job ID. `submission.json` is updated with the same IDs.

## 6. Monitor jobs and logs

Show the current queue with useful fields:

```bash
squeue -u "$USER" \
  -o '%.18i %.40j %.10P %.10T %.10M %.10l %R'
```

Common pending reasons include:

- `Priority`: valid job waiting behind higher-priority work.
- `Resources`: valid job waiting for a suitable node.
- `Dependency`, `QOS...`, or `Partition...`: inspect the request more closely.

Build a comma-separated list from the submission manifest when inspecting only
one batch:

```bash
JOB_IDS=$(awk -F '\t' 'NR > 1 && $1 != "" {print $1}' \
  "$WORKDIR/jobs.tsv" | paste -sd, -)
squeue -j "$JOB_IDS" -o '%.18i %.40j %.10P %.10T %.10M %.10l %R'
```

Stdout and stderr are separate and include the Slurm job ID:

```text
<WORKDIR>/logs/<run-name>-slurm-<job-id>.out
<WORKDIR>/logs/<run-name>-slurm-<job-id>.err
```

Useful log commands:

```bash
ls -ltr "$WORKDIR/logs" | tail
tail -f "$WORKDIR/logs/<run-name>-slurm-<job-id>.out"
tail -f "$WORKDIR/logs/<run-name>-slurm-<job-id>.err"
grep -RniE 'traceback|out of memory|killed|error' "$WORKDIR/logs"
```

Warnings are not automatically failures. Confirm the job state and read enough
surrounding log context before acting.

For completed or failed jobs, use accounting rather than `squeue`:

```bash
sacct -j "$JOB_IDS" \
  --format=JobID,JobName%40,Partition,State,Elapsed,ExitCode,ReqMem,MaxRSS
```

Peak process memory is usually reported on the `.batch` step. Use these values
to justify future memory changes instead of guessing.

W&B runs appear only after jobs start and initialize logging. The project and
group do not indicate whether a still-pending Slurm job is healthy.

For PBT reporting, average the four policy streams as one population. The
IntrMotiv helper aligns each run at the latest step shared by all policies and
writes both a terminal summary and one-million-step curves:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate SFgit
python sf_working_directories/IntrMotiv/analysis/aggregate_pbt_population.py \
  train_dir/<batch-name> \
  train_dir/<batch-name>/population_analysis
```

The output files are `population_latest.csv`, `population_curves.csv`, and
`population_summary.json`. A population is only considered complete when all
four policy streams are present; do not sum their curves as four independent
replicates.

## 7. Cancel exactly one submission

Every successful submission has a generated cancellation script containing only
that submission's job IDs:

```bash
bash "$WORKDIR/scancel.sh"
```

Review it first when working on a shared account:

```bash
sed -n '1,200p' "$WORKDIR/scancel.sh"
```

Prefer this script over broad commands such as cancelling all jobs owned by the
account or matching a short job-name pattern.

After cancellation, verify with both tools:

```bash
squeue -j "$JOB_IDS"
sacct -j "$JOB_IDS" --format=JobID,JobName%40,State,Elapsed,ExitCode
```

## 8. Retry failed runs

Do not resubmit the full sweep blindly. First identify the failure cause and
determine whether the training directory contains a usable checkpoint.

For a small retry set, create a separate run-description module that imports the
original experiment constructor, preserves the original batch and experiment
names, and lists only the failed conditions:

```python
from sample_factory.launcher.run_description import RunDescription
from sf_working_directories.IntrMotiv.dmlab.experiments.hrl_intrinsic_arch_search import (
    BATCH_NAME,
    experiment,
)


RUN_DESCRIPTION = RunDescription(
    BATCH_NAME,
    experiments=[
        experiment(99, 64, 2.43, "mean"),
        experiment(456, 128, 2.0, "punish"),
    ],
)
```

The argument order must match the original constructor. Preflight the retry
module exactly as for a new batch. Preserving the names points Sample Factory at
the existing training roots; whether it resumes or restarts is then controlled
by the experiment's Sample Factory restart behavior and available checkpoints.
Verify that behavior before submission.

If code or scientific settings changed, use a new batch and run name. Do not
resume an old directory with behaviorally different code while presenting it as
the same run.

## 9. Override resources only with a reason

Additional launcher arguments placed after the mode override wrapper defaults:

```bash
sf_working_directories/IntrMotiv/launcher/launch_nemo2.sh \
  sf_working_directories.IntrMotiv.dmlab.experiments.my_batch \
  --print-only \
  --slurm_partition=cpu \
  --slurm_cpus_per_job=20 \
  --slurm_memory=60G \
  --slurm_timeout=08:30:00
```

Always preflight overridden settings. Change them based on worker count,
measured `MaxRSS`, measured run time, or an explicit cluster requirement. Do not
request more resources merely because a previous script did.

Two environment variables are available for exceptional cases:

```bash
# Use an explicit metadata/log directory.
SLURM_WORKDIR=/absolute/path/to/submission_dir \
  sf_working_directories/IntrMotiv/launcher/launch_nemo2.sh \
  sf_working_directories.IntrMotiv.dmlab.experiments.my_batch --print-only

# Use a different launcher interpreter.
SFGIT_PYTHON=/absolute/path/to/python \
  sf_working_directories/IntrMotiv/launcher/launch_nemo2.sh \
  sf_working_directories.IntrMotiv.dmlab.experiments.my_batch --print-only
```

The generated training jobs still activate `SFgit` unless the IntrMotiv Slurm
template is intentionally changed.

## 10. Compatibility and ownership rules

- Keep new experiment code, run descriptions, and documentation in
  `sf_working_directories/IntrMotiv/` whenever possible.
- Do not edit `sf_working_directories/jannek/`.
- Do not replace Sample Factory's launcher with a private submission loop for
  training or ordinary experiment batches. The documented exception is the
  manifest-driven post-training place-field evaluator: use
  `evaluation/submit_place_field_sweep.py`, which submits one ordinary Slurm
  job per explicit checkpoint row. See the IntrMotiv vault's
  `04_implementation/reusable_place_field_telemetry.md` for its required
  print-only review and workspace policy.
- Changes to `sample_factory/launcher/` must remain backward compatible with
  existing run descriptions and templates.
- Do not edit generated `sbatch_*.sh` files and then assume the run description
  documents what was submitted.
- Do not reuse a batch name for a scientifically different experiment.
- Do not delete another user's logs, checkpoints, queued jobs, or dirty files.
- Failure email currently goes to the address in
  `nemo2_sfgit_intrmotiv.sh`; do not silently redirect it.

## 11. Short checklist

Before submission:

- [ ] Work is contained in `IntrMotiv` unless a compatible core change is needed.
- [ ] `BATCH_NAME` is unique and descriptive.
- [ ] W&B project, group, and run names follow the intended hierarchy.
- [ ] Sweep product and generated script count match the intended job count.
- [ ] Every run has a seed and an unambiguous name.
- [ ] A `--print-only` preflight completed successfully.
- [ ] One generated Slurm script and command were read completely.
- [ ] Resource requests have evidence or match the established profile.
- [ ] No one edited the run description after preflight.

After submission:

- [ ] Every `jobs.tsv` row says `submitted` and contains a job ID.
- [ ] `squeue` shows the expected partition and job count.
- [ ] One job's stdout and stderr paths point into the submission work directory.
- [ ] W&B runs are checked after jobs begin running.
- [ ] The location of `scancel.sh` is recorded for the batch owner.

## 12. Minimal command reference

```bash
# Preflight
sf_working_directories/IntrMotiv/launcher/launch_nemo2.sh MODULE --print-only

# Submit
sf_working_directories/IntrMotiv/launcher/launch_nemo2.sh MODULE --submit

# Monitor
squeue -u "$USER" -o '%.18i %.40j %.10P %.10T %.10M %.10l %R'

# Inspect completed jobs
sacct -j "$JOB_IDS" --format=JobID,JobName%40,State,Elapsed,ExitCode,ReqMem,MaxRSS

# Cancel only the recorded submission
bash "$WORKDIR/scancel.sh"
```
