#!/bin/bash
#SBATCH --job-name=$NAME
#SBATCH --time=$TIMEOUT
#SBATCH $PARTITION
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPU
#SBATCH --mem=$MEMORY
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=xiaoxiong.lin@bcf.uni-freiburg.de

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate SFgit
cd "$${SLURM_SUBMIT_DIR:?Submit from the intended source checkout}" || exit 1

WORKSPACE_ROOT=$${INTRMOTIV_WORKSPACE_ROOT:-/work/classic/fr_xl1014-corridor-geometry}
RUNTIME_ROOT=$${INTRMOTIV_RUNTIME_ROOT:-$$WORKSPACE_ROOT/IntrMotiv/SF_hipposlam/runtime}
export PYTHONPATH="$$RUNTIME_ROOT/torch_cuda_2_9_1:$$RUNTIME_ROOT/controller_terminal_binding_v1:$${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export XDG_CACHE_HOME="$$RUNTIME_ROOT/cache"
export MPLCONFIGDIR="$$RUNTIME_ROOT/matplotlib"
export WANDB_CACHE_DIR="$$RUNTIME_ROOT/wandb_cache"
export WANDB_DATA_DIR="$$RUNTIME_ROOT/wandb_data"
export WANDB_DIR="$$RUNTIME_ROOT/wandb"
export TMPDIR="$$WORKSPACE_ROOT/tmp/intrmotiv_$${SLURM_JOB_ID:-manual}"
mkdir -p "$$XDG_CACHE_HOME" "$$MPLCONFIGDIR" "$$WANDB_CACHE_DIR" "$$WANDB_DATA_DIR" "$$WANDB_DIR" "$$TMPDIR"

exec python -m sf_working_directories.IntrMotiv.dmlab.train_hipposlam $CMD \
  --heartbeat_interval=40 \
  --heartbeat_reporting_interval=600
