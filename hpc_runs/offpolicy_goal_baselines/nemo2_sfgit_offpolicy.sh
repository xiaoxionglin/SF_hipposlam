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
CONDA_ENV=$${INTRMOTIV_OFFPOLICY_CONDA_ENV:-SFgit}
set +u
conda activate "$$CONDA_ENV"
set -u
if [[ $${INTRMOTIV_OFFPOLICY_PRELOAD_CONDA_LIBSTDCXX:-0} == 1 ]]; then
  export LD_PRELOAD="$$CONDA_PREFIX/lib/libstdc++.so.6$${LD_PRELOAD:+:$$LD_PRELOAD}"
fi
cd ~/SF_git_XXL/SF_hipposlam || exit 1

RUNTIME_ROOT=$${INTRMOTIV_RUNTIME_ROOT:-/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime}
export XDG_CACHE_HOME="$$RUNTIME_ROOT/cache"
export MPLCONFIGDIR="$$RUNTIME_ROOT/matplotlib"
export WANDB_CACHE_DIR="$$RUNTIME_ROOT/wandb_cache"
export WANDB_DATA_DIR="$$RUNTIME_ROOT/wandb_data"
export WANDB_DIR="$$RUNTIME_ROOT/wandb"
export TMPDIR="$$RUNTIME_ROOT/tmp/$${SLURM_JOB_ID:-manual}"
mkdir -p "$$XDG_CACHE_HOME" "$$MPLCONFIGDIR" "$$WANDB_CACHE_DIR" "$$WANDB_DATA_DIR" "$$WANDB_DIR" "$$TMPDIR"

TRAIN_MODULE=$${INTRMOTIV_OFFPOLICY_TRAIN_MODULE:-hpc_runs.offpolicy_goal_baselines.train}
exec python -m "$$TRAIN_MODULE" $CMD
