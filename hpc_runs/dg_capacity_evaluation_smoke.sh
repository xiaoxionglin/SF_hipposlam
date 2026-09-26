#!/bin/bash
#SBATCH --job-name=DGC_eval_smoke
#SBATCH --time=00:20:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
set -euo pipefail
cd /home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DGC_SMOKE_ROOT=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/dg_capacity_preflight/evaluation_smoke_${SLURM_JOB_ID}
export XDG_CACHE_HOME=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/cache
export MPLCONFIGDIR=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/matplotlib
export TMPDIR=$DGC_SMOKE_ROOT/tmp
export WANDB_MODE=disabled WANDB_DIR=$DGC_SMOKE_ROOT/wandb
mkdir -p "$TMPDIR" "$WANDB_DIR"
exec /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.dg_capacity_evaluation_smoke \
  hpc_runs/studies/dg_capacity_goal_conditioning_preflight.study.json \
  /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/intrmotiv_dg_capacity_goal_conditioning_preflight2_20260910 \
  "$DGC_SMOKE_ROOT"
