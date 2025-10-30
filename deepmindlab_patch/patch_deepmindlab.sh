#!/usr/bin/env bash
# === copy_dmlab_asset.sh ===
# Copy game_scripts/assets.pk3 into the active conda env's DMLab baselab folder.

set -euo pipefail

# --- locate source file ---
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_FILE="${SRC_DIR}/assets.pk3"

SRC_FILE_DIR="${SRC_DIR}/game_scripts/"

if [[ ! -f "$SRC_FILE" ]]; then
  echo "❌ Error: ${SRC_FILE} not found." >&2
  exit 1
fi

# --- detect conda env base path ---
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "❌ Error: No conda environment detected (CONDA_PREFIX not set)." >&2
  exit 1
fi

# --- find Python site-packages path for this env ---
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"

TARGET_DIR="${SITE_PACKAGES}/deepmind_lab/baselab"
mkdir -p "$TARGET_DIR"

echo "Copying ${SRC_FILE}"
echo "→ ${TARGET_DIR}/assets.pk3"
cp -v "$SRC_FILE" "$TARGET_DIR/"

cp -r -v "$SRC_FILE_DIR" "$TARGET_DIR/"

echo "✅ Done: asset copied into ${CONDA_PREFIX}"
