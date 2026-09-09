#!/bin/bash
#SBATCH --job-name=SCR_preflight_gate
#SBATCH --time=00:30:00
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=xiaoxiong.lin@bcf.uni-freiburg.de

set -euo pipefail

REPO_ROOT=/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam
PYTHON=/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python
TRAIN_ROOT=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir
STUDY=hpc_runs/studies/source_credit_retirement.study.json
PREFLIGHT_JOBS=$TRAIN_ROOT/_slurm/intrmotiv_source_credit_retirement_20260904_preflight_v3/20260904_submitted/jobs.tsv
GATE_ROOT=$TRAIN_ROOT/analysis/source_credit_retirement_20260904/preflight_gate_v3
PRINT_ROOT=$TRAIN_ROOT/_slurm/intrmotiv_source_credit_retirement_20260904/20260904_gate_print_only
SUBMIT_ROOT=$TRAIN_ROOT/_slurm/intrmotiv_source_credit_retirement_20260904/20260904_gate_submitted
TEMPLATE=$REPO_ROOT/sf_working_directories/IntrMotiv/dmlab/experiments/nemo2_sfgit_intrmotiv.sh

cd "$REPO_ROOT"
mkdir -p "$GATE_ROOT"

PYTHONPATH=. "$PYTHON" hpc_runs/analyze_source_credit_retirement_preflight.py \
  "$STUDY" "$PREFLIGHT_JOBS" "$TRAIN_ROOT" "$GATE_ROOT/preflight_report.json"
PYTHONPATH=. "$PYTHON" -m hpc_runs.intrmotiv_study validate "$STUDY" \
  > "$GATE_ROOT/validated_study.json"
PYTHONPATH=. "$PYTHON" -m hpc_runs.intrmotiv_study render-runs "$STUDY" \
  --output "$GATE_ROOT/production_runs.json"

PYTHONPATH=. "$PYTHON" -m sample_factory.launcher.run \
  --train_dir="$TRAIN_ROOT" \
  --run=hpc_runs.source_credit_retirement \
  --backend=slurm \
  --pause_between=1 \
  --slurm_gpus_per_job=0 \
  --slurm_cpus_per_job=40 \
  --slurm_memory=80G \
  --slurm_print_only=True \
  --slurm_workdir="$PRINT_ROOT" \
  --slurm_log_dir="$PRINT_ROOT/logs" \
  --slurm_separate_stderr=True \
  --slurm_partition=cpu \
  --slurm_sbatch_template="$TEMPLATE" \
  --slurm_timeout=30:00:00
PYTHONPATH=. "$PYTHON" -m hpc_runs.intrmotiv_study audit-submission \
  "$STUDY" "$PRINT_ROOT/jobs.tsv" --output "$GATE_ROOT/print_only_audit.json"

PYTHONPATH=. "$PYTHON" -m sample_factory.launcher.run \
  --train_dir="$TRAIN_ROOT" \
  --run=hpc_runs.source_credit_retirement \
  --backend=slurm \
  --pause_between=1 \
  --slurm_gpus_per_job=0 \
  --slurm_cpus_per_job=40 \
  --slurm_memory=80G \
  --slurm_print_only=False \
  --slurm_workdir="$SUBMIT_ROOT" \
  --slurm_log_dir="$SUBMIT_ROOT/logs" \
  --slurm_separate_stderr=True \
  --slurm_partition=cpu \
  --slurm_sbatch_template="$TEMPLATE" \
  --slurm_timeout=30:00:00
PYTHONPATH=. "$PYTHON" -m hpc_runs.intrmotiv_study audit-submission \
  "$STUDY" "$SUBMIT_ROOT/jobs.tsv" --submitted \
  --output "$GATE_ROOT/submitted_audit.json"
