#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_MODULE [--print-only|--submit] [launcher arguments...]" >&2
  exit 2
fi

RUN_MODULE=$1
shift
MODE=--print-only
if [[ $# -gt 0 && ( $1 == --print-only || $1 == --submit ) ]]; then
  MODE=$1
  shift
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON=${SFGIT_PYTHON:-/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python}
TRAIN_ROOT=${INTRMOTIV_TRAIN_ROOT:-/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir}
TIMEOUT=${INTRMOTIV_OFFPOLICY_SLURM_TIMEOUT:-02:00:00}
TRAIN_MODULE=${INTRMOTIV_OFFPOLICY_TRAIN_MODULE:-hpc_runs.offpolicy_goal_baselines.train}
PARTITION=${INTRMOTIV_OFFPOLICY_SLURM_PARTITION:-cpu}
GPUS=${INTRMOTIV_OFFPOLICY_SLURM_GPUS:-0}
CPUS=${INTRMOTIV_OFFPOLICY_SLURM_CPUS:-16}
MEMORY=${INTRMOTIV_OFFPOLICY_SLURM_MEMORY:-40G}
RUN_NAME=$(
  "$PYTHON" -c 'import importlib, sys; print(importlib.import_module(sys.argv[1]).RUN_DESCRIPTION.run_name)' "$RUN_MODULE"
)
SUBMISSION_TIME=$(date -u +%Y%m%dT%H%M%SZ)
WORKDIR=${SLURM_WORKDIR:-$TRAIN_ROOT/_slurm/$RUN_NAME/$SUBMISSION_TIME}
PRINT_ONLY=True
if [[ $MODE == --submit ]]; then
  PRINT_ONLY=False
fi

echo "Batch:      $RUN_NAME"
echo "Mode:       $MODE"
echo "Workdir:    $WORKDIR"
echo "Train root: $TRAIN_ROOT"

exec "$PYTHON" -m sample_factory.launcher.run \
  --backend=slurm \
  --run="$RUN_MODULE" \
  --train_dir="$TRAIN_ROOT" \
  --slurm_workdir="$WORKDIR" \
  --slurm_log_dir="$WORKDIR/logs" \
  --slurm_sbatch_template="$REPO_ROOT/hpc_runs/offpolicy_goal_baselines/nemo2_sfgit_offpolicy.sh" \
  --slurm_partition="$PARTITION" \
  --slurm_gpus_per_job="$GPUS" \
  --slurm_cpus_per_job="$CPUS" \
  --slurm_memory="$MEMORY" \
  --slurm_timeout="$TIMEOUT" \
  --slurm_separate_stderr=True \
  --slurm_print_only="$PRINT_ONLY" \
  --pause_between=1 \
  "$@"
