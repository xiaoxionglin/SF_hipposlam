#!/bin/bash
#SBATCH --job-name=ddqn-throughput
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR=/work/classic/fr_xl1014-train/tmp/dqperf_${SLURM_JOB_ID}
export XDG_CACHE_HOME=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/cache
export MPLCONFIGDIR=/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime/matplotlib
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
exec /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.intrmotiv_offpolicy.benchmark "$@"
