#!/usr/bin/env sh
set -eu

# Unified inference measurement for the main comparison baselines.
# The default IDs are M1-M4, M6-M11 and L1 (the stored run displayed as M5).
# This reads frozen best.pth files and never trains or changes a run directory.
#
# Overrides:
#   RUN_ROOT=outputs_experiments/pmdfnet_mmlsv2
#   COMPLEXITY_IDS=M1,M2,...,L1
#   COMPLEXITY_SEEDS=41
#   DEVICE=cuda:0
#   COMPLEXITY_BATCH_SIZE=1
#   COMPLEXITY_WARMUP=20
#   COMPLEXITY_ITERATIONS=100
#   COMPLEXITY_OUTPUT_DIR=outputs_experiments/complexity

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

RUN_ROOT="${RUN_ROOT:-outputs_experiments/pmdfnet_mmlsv2}"
COMPLEXITY_IDS="${COMPLEXITY_IDS:-M1,M2,M3,M4,M6,M7,M8,M9,M10,M11,L1}"
COMPLEXITY_SEEDS="${COMPLEXITY_SEEDS:-41}"
COMPLEXITY_BATCH_SIZE="${COMPLEXITY_BATCH_SIZE:-1}"
COMPLEXITY_WARMUP="${COMPLEXITY_WARMUP:-20}"
COMPLEXITY_ITERATIONS="${COMPLEXITY_ITERATIONS:-100}"
COMPLEXITY_OUTPUT_DIR="${COMPLEXITY_OUTPUT_DIR:-outputs_experiments/complexity}"

echo "######## complexity: ids=${COMPLEXITY_IDS} seeds=${COMPLEXITY_SEEDS} device=${DEVICE} ########"
"${PYTHON_BIN}" tools/evaluation/measure_model_complexity.py \
  --config "${CONFIG}" \
  --run-root "${RUN_ROOT}" \
  --ids "${COMPLEXITY_IDS}" \
  --seeds "${COMPLEXITY_SEEDS}" \
  --device "${DEVICE}" \
  --batch-size "${COMPLEXITY_BATCH_SIZE}" \
  --warmup "${COMPLEXITY_WARMUP}" \
  --iterations "${COMPLEXITY_ITERATIONS}" \
  --output-dir "${COMPLEXITY_OUTPUT_DIR}"
