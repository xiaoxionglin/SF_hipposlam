#!/bin/bash
#SBATCH --job-name=hrl_intrinsic_arch
#SBATCH --time=$TIMEOUT
#SBATCH $PARTITION
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPU
#SBATCH --mem=200G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiaoxiong.lin@bcf.uni-freiburg.de

source ~/miniforge3/etc/profile.d/conda.sh
conda activate SFgit
cd ~/SF_git_XXL/SF_hipposlam || exit 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m sf_working_directories.jannek.dmlab.train_hipposlam $CMD \
  --heartbeat_interval=40 \
  --heartbeat_reporting_interval=600
