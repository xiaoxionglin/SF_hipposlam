#!/usr/bin/env bash
set -euo pipefail

# Same three-argument contract as the canonical place-field runner. The existing
# submit_place_field_sweep.py supplies one ordinary Slurm job per terminal row.
manifest=${1:?manifest TSV}
row_index=${2:?zero-based manifest row}
output_dir=${3:?workspace output directory}
workspace_root=${INTRMOTIV_WORKSPACE_ROOT:?active workspace}
runtime_source=${INTRMOTIV_RUNTIME_SOURCE:?isolated runtime release}
python=${INTRMOTIV_EVAL_PYTHON:?evaluation Python}
[[ $row_index =~ ^[0-9]+$ ]] || { echo "Invalid manifest row" >&2; exit 2; }

require_workspace_path() {
    case "$1" in "$workspace_root"/*) ;; *) echo "Outside active workspace: $1" >&2; exit 2 ;; esac
}
require_workspace_path "$manifest"
require_workspace_path "$output_dir"

# Reassert short workspace temp paths inside the job: node prologs may replace
# submitted TMPDIR. Source and Python are inherited from the audited dispatcher.
export TMPDIR="$workspace_root/tmp/cth_${SLURM_JOB_ID:-local}_${row_index}"
export DMLAB_CACHE_DIR="$output_dir/dmlab_cache"
export XDG_CACHE_HOME="$output_dir/cache"
export WANDB_DIR="$output_dir/wandb"
export WANDB_MODE=disabled
export TORCH_HOME="$workspace_root/IntrMotiv/SF_hipposlam/runtime/cache/torch"
export PYTHONPATH="$runtime_source${PYTHONPATH:+:$PYTHONPATH}"
terminal_binding=${INTRMOTIV_TERMINAL_BINDING:-$workspace_root/IntrMotiv/SF_hipposlam/runtime/controller_terminal_binding_v1}
if [[ -d $terminal_binding ]]; then
    export PYTHONPATH="$runtime_source:$terminal_binding${PYTHONPATH:+:$PYTHONPATH}"
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
mkdir -p "$TMPDIR" "$DMLAB_CACHE_DIR" "$XDG_CACHE_HOME" "$WANDB_DIR" "$output_dir/results"

mapfile -t fields < <("$python" - "$manifest" "$row_index" "$workspace_root" <<'PY'
import csv
import json
from pathlib import Path
import sys

with open(sys.argv[1], newline="") as handle:
    row = list(csv.DictReader(handle, delimiter="\t"))[int(sys.argv[2])]
workspace = Path(sys.argv[3]).resolve(strict=True)
run_dir = Path(row["run_dir"]).resolve(strict=True)
checkpoint = Path(row["checkpoint"]).resolve(strict=True)
config = json.loads((run_dir / "config.json").read_text())
runfiles = Path(config["dmlab_runfiles_path"]).resolve(strict=True)
for path in (run_dir, checkpoint, runfiles):
    if not path.is_relative_to(workspace):
        raise ValueError(f"Evaluation input outside active workspace: {path}")
label = row["label_suffix"]
if not label or Path(label).name != label:
    raise ValueError("Invalid output label")
for value in (run_dir, checkpoint, runfiles, label):
    print(value)
PY
)
[[ ${#fields[@]} -eq 4 ]] || { echo "Cannot resolve heldout manifest row" >&2; exit 2; }
cd "$runtime_source"
"$python" sf_working_directories/IntrMotiv/evaluation/fixed_reward_heldout.py \
    --run-dir "${fields[0]}" --checkpoint "${fields[1]}" --runfiles "${fields[2]}" \
    --site cued5 --out "$output_dir/results/${fields[3]}.csv" \
    --trials 100 --horizon 1800 --seed-base 61000 --stochastic
