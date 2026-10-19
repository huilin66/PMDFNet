#!/usr/bin/env sh
set -eu

# Probe one synthetic training step per paper configuration. The default
# batch/allocator settings leave headroom on a shared 80 GB A100.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

. "${REPO_ROOT}/scripts/_env.sh"

PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG=${CONFIG:-configs/mmlsv2_paper_protocol.json}
GROUP=${GROUP:-all}
DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-8}
SEED=${SEED:-41}
MAX_MEMORY_GB=${MAX_MEMORY_GB:-40}
ALLOCATOR_LIMIT_GB=${ALLOCATOR_LIMIT_GB:-38}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-300}
OUTPUT_DIR=${OUTPUT_DIR:-outputs_experiments/smoke/vram}

export PYTHONUNBUFFERED=1

echo "[vram-smoke] config=${CONFIG} group=${GROUP} device=${DEVICE} batch=${BATCH_SIZE}"
echo "[vram-smoke] safety limit=${MAX_MEMORY_GB}GB allocator limit=${ALLOCATOR_LIMIT_GB}GB"

PYTHONHASHSEED="${SEED}" \
CUBLAS_WORKSPACE_CONFIG=":4096:8" \
TRAIN_STRICT_DETERMINISM=1 \
"${PYTHON_BIN}" tools/validation/smoke_test_vram.py \
  --config "${CONFIG}" \
  --group "${GROUP}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --max-memory-gb "${MAX_MEMORY_GB}" \
  --allocator-limit-gb "${ALLOCATOR_LIMIT_GB}" \
  --timeout-seconds "${TIMEOUT_SECONDS}" \
  --output-dir "${OUTPUT_DIR}"
