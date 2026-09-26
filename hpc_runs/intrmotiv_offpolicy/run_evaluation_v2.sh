#!/bin/bash
set -euo pipefail
export TMPDIR=/work/classic/fr_xl1014-train/tmp/ddqn_eval_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd /home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam
exec /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m hpc_runs.intrmotiv_offpolicy.evaluate_manifest "$@"
