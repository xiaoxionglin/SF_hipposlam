#!/bin/bash
#SBATCH --job-name=$NAME
#SBATCH --time=$TIMEOUT
#SBATCH $PARTITION
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPU
#SBATCH --mem=$MEMORY
set -euo pipefail
source /home/fr/fr_xl1014/miniforge3/etc/profile.d/conda.sh
conda activate SFgit
cd /home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/cache
export MPLCONFIGDIR=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/matplotlib
export WANDB_CACHE_DIR=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/wandb_cache
export WANDB_DATA_DIR=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/wandb_data
export WANDB_DIR=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/wandb
export TMPDIR=/work/classic/fr_xl1014-train/tmp/ddqn_$${SLURM_JOB_ID}
mkdir -p "$$TMPDIR" "$$XDG_CACHE_HOME" "$$MPLCONFIGDIR" "$$WANDB_DIR" "$$WANDB_CACHE_DIR" "$$WANDB_DATA_DIR"
exec python -m hpc_runs.intrmotiv_offpolicy.sf_native $CMD
