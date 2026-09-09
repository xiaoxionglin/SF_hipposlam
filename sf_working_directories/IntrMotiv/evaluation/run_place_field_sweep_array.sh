#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for historical array submissions. New NEMO2 sweeps
# must use submit_place_field_sweep.py, which creates ordinary Slurm jobs.
manifest=${1:?usage: run_place_field_sweep_array.sh MANIFEST OUTPUT_DIR}
output_dir=${2:?usage: run_place_field_sweep_array.sh MANIFEST OUTPUT_DIR}
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/run_place_field_sweep_single.sh" "$manifest" "$task_id" "$output_dir"
