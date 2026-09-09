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
cd ~/SF_git_XXL/SF_hipposlam || exit 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUNTIME_ROOT=$${INTRMOTIV_RUNTIME_ROOT:-/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime}
export XDG_CACHE_HOME="$$RUNTIME_ROOT/cache"
export MPLCONFIGDIR="$$RUNTIME_ROOT/matplotlib"
export WANDB_CACHE_DIR="$$RUNTIME_ROOT/wandb_cache"
export WANDB_DATA_DIR="$$RUNTIME_ROOT/wandb_data"
export WANDB_DIR="$$RUNTIME_ROOT/wandb"
export TMPDIR="$$RUNTIME_ROOT/tmp/$${SLURM_JOB_ID:-manual}"
mkdir -p "$$XDG_CACHE_HOME" "$$MPLCONFIGDIR" "$$WANDB_CACHE_DIR" "$$WANDB_DATA_DIR" "$$WANDB_DIR" "$$TMPDIR"

exec python -m sf_working_directories.IntrMotiv.dmlab.train_hipposlam $CMD \
  --heartbeat_interval=40 \
  --heartbeat_reporting_interval=600
