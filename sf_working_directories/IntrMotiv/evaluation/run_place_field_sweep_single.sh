#!/usr/bin/env bash
set -euo pipefail

# Run one zero-based row from the standard place-field manifest. This is the
# canonical worker for ordinary (non-array) Slurm submissions on NEMO2.
manifest=${1:?usage: run_place_field_sweep_single.sh MANIFEST ROW_INDEX OUTPUT_DIR}
row_index=${2:?usage: run_place_field_sweep_single.sh MANIFEST ROW_INDEX OUTPUT_DIR}
output_dir=${3:?usage: run_place_field_sweep_single.sh MANIFEST ROW_INDEX OUTPUT_DIR}

if [[ ! $row_index =~ ^[0-9]+$ ]]; then
  echo "ROW_INDEX must be a non-negative integer, got: $row_index" >&2
  exit 2
fi

workspace_root=${INTRMOTIV_WORKSPACE_ROOT:-/work/classic/fr_xl1014-train}
require_workspace_path() {
  local path=$1
  local label=$2
  case "$path" in
    "$workspace_root"/*) ;;
    *)
      echo "$label must resolve under $workspace_root, got: $path" >&2
      exit 2
      ;;
  esac
}

require_workspace_path "$manifest" "Manifest"
require_workspace_path "$output_dir" "Output directory"
if [[ ! -f $manifest ]]; then
  echo "Manifest does not exist: $manifest" >&2
  exit 2
fi

line=$((row_index + 2))
# Bash treats tabs as IFS whitespace and collapses consecutive delimiters, so
# `read` cannot preserve intentionally empty TSV fields. Parse with csv.reader
# and emit one physical line per field; `mapfile` retains blank lines as empty
# array entries.
mapfile -t manifest_fields < <(
  /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python - "$manifest" "$line" <<'PY'
import csv
import sys

manifest, line_number = sys.argv[1], int(sys.argv[2])
with open(manifest, newline="", encoding="utf-8") as handle:
    rows = csv.reader(handle, delimiter="\t")
    row = next((value for index, value in enumerate(rows, start=1) if index == line_number), None)
if row is None:
    raise SystemExit(f"No manifest row at physical line {line_number}")
if len(row) != 11:
    raise SystemExit(f"Expected 11 manifest fields, found {len(row)} at physical line {line_number}")
for value in row:
    print(value)
PY
)
if [[ ${#manifest_fields[@]} -ne 11 ]]; then
  echo "Could not parse 11 manifest fields for zero-based index ${row_index}" >&2
  exit 2
fi
condition=${manifest_fields[0]}
family=${manifest_fields[1]}
schedule=${manifest_fields[2]}
feedback=${manifest_fields[3]}
half_life=${manifest_fields[4]}
seed=${manifest_fields[5]}
target_frames=${manifest_fields[6]}
checkpoint_frames=${manifest_fields[7]}
checkpoint=${manifest_fields[8]}
run_dir=${manifest_fields[9]}
label_suffix=${manifest_fields[10]}
label_suffix=${label_suffix%$'\r'}

if [[ -z ${condition:-} ]]; then
  echo "No manifest row for zero-based index ${row_index}" >&2
  exit 2
fi
require_workspace_path "$checkpoint" "Checkpoint"
require_workspace_path "$run_dir" "Run directory"
if [[ ! -f $checkpoint ]]; then
  echo "Checkpoint does not exist: $checkpoint" >&2
  exit 2
fi
if [[ ! -d $run_dir ]]; then
  echo "Run directory does not exist: $run_dir" >&2
  exit 2
fi

# NEMO2's node prolog may replace a submitted TMPDIR with /tmp. Set every
# runtime path again inside the job so the storage policy is deterministic.
export TMPDIR=$output_dir/tmp
export DMLAB_CACHE_DIR=$output_dir/dmlab_cache
export XDG_CACHE_HOME=$output_dir/cache
export WANDB_DIR=$output_dir/wandb
export WANDB_MODE=disabled
require_workspace_path "$TMPDIR" "TMPDIR"
require_workspace_path "$DMLAB_CACHE_DIR" "DMLab cache"
require_workspace_path "$XDG_CACHE_HOME" "XDG cache"
require_workspace_path "$WANDB_DIR" "W&B directory"
mkdir -p "$output_dir/raw" "$output_dir/slurm" "$TMPDIR" "$DMLAB_CACHE_DIR" "$XDG_CACHE_HOME" "$WANDB_DIR"

cd /home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

deterministic_args=()
if [[ ${PLACE_FIELD_DETERMINISTIC:-0} == 1 ]]; then
  deterministic_args+=(--deterministic)
fi
panel_args=()
if [[ -n ${PLACE_FIELD_RECORD_PANEL:-} ]]; then
  require_workspace_path "$PLACE_FIELD_RECORD_PANEL" "Observation panel"
  panel_args+=(--record-observation-panel "$PLACE_FIELD_RECORD_PANEL")
fi
if [[ -n ${PLACE_FIELD_REPLAY_PANEL:-} ]]; then
  require_workspace_path "$PLACE_FIELD_REPLAY_PANEL" "Observation panel"
  panel_args+=(--replay-observation-panel "$PLACE_FIELD_REPLAY_PANEL")
fi
place_field_max_frames=${PLACE_FIELD_MAX_FRAMES:-10000}

/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python \
  sf_working_directories/IntrMotiv/evaluation/place_fields.py \
  --run-dir "$run_dir" \
  --out-dir "$output_dir/raw" \
  --checkpoint "$checkpoint" \
  --label-suffix "$label_suffix" \
  --max-num-frames "$place_field_max_frames" \
  --no-plots \
  "${deterministic_args[@]}" "${panel_args[@]}"
