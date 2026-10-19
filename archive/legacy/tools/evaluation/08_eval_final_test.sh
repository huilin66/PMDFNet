#!/usr/bin/env sh
set -eu

# Evaluate exactly one frozen checkpoint. This script performs no training and
# cannot alter model selection. Set CHECKPOINT explicitly.

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${CHECKPOINT:?Set CHECKPOINT to a selected best.pth file}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -n "${DATA_ROOT:-}" ]; then
  export MMLSV2_DATA_ROOT="${DATA_ROOT}"
fi

"${PYTHON_BIN}" tools/evaluation/eval_skg.py \
  --checkpoint "${CHECKPOINT}" \
  --split "${SPLIT:-test}" \
  --device "${DEVICE:-auto}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --output-json "${OUTPUT_JSON:-}"
