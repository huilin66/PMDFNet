#!/usr/bin/env bash
set -euo pipefail

# Evaluate a frozen validation-selected checkpoint on the public MMLSv2 test
# masks. This script never trains and requires CHECKPOINT explicitly.

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

. "${REPO_ROOT}/scripts/_env.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
# Same resolution order as the trainer (and .env); no baked-in host path.
DATA_ROOT="${DATA_ROOT:-${MARS_DATA_ROOT:-${MAPPED_DATA_ROOT:-${MMLSV2_DATA_ROOT:-}}}}"
if [ -z "${DATA_ROOT}" ]; then
  echo "DATA_ROOT is not set. Export DATA_ROOT (or MARS_DATA_ROOT/MAPPED_DATA_ROOT/MMLSV2_DATA_ROOT) to the mapped MMLSv2 root." >&2
  exit 1
fi
DEVICE="${DEVICE:-${TRAIN_DEVICE:-cuda:0}}"
MODEL_NAME="${MODEL_NAME:-dual_segformer_convnextsmall_chv1_add}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the selected best.pth file}"
OUTPUT_DIR="${OUTPUT_DIR:-$(cd "$(dirname "${CHECKPOINT}")/.." && pwd)}"
STAGING_OUTPUT_DIR="${STAGING_OUTPUT_DIR:-}"

if [ ! -d "${DATA_ROOT}/test/images" ] || [ ! -d "${DATA_ROOT}/test/masks" ]; then
  echo "Public labeled test split is incomplete: ${DATA_ROOT}" >&2
  exit 1
fi

echo "[test] data=${DATA_ROOT}"
echo "[test] model=${MODEL_NAME}"
echo "[test] checkpoint=${CHECKPOINT}"
echo "[test] device=${DEVICE}"

args=(
  tools/evaluation/evaluate_test_metrics.py
  --data-root "${DATA_ROOT}"
  --checkpoint "${CHECKPOINT}"
  --model-name "${MODEL_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
)
if [ -n "${STAGING_OUTPUT_DIR}" ]; then
  args+=(--staging-output-dir "${STAGING_OUTPUT_DIR}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" "${args[@]}"
