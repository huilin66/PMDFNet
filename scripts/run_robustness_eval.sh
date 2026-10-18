#!/usr/bin/env sh
set -eu

# Paired clean/robustness Test evaluation. The default compares M4 with the
# stored L1 main model (paper-facing M5) over seeds 41/42/43 and five frozen
# conditions. Per-image TP/FP/FN/TN are written to a JSONL sidecar.
#
# Overrides:
#   DATA_ROOT=...                         mapped MMLSv2 root with public Test masks
#   RUN_ROOT=outputs_experiments/pmdfnet_mmlsv2
#   ROBUSTNESS_IDS=M4,L1
#   ROBUSTNESS_SEEDS="41 42 43"
#   DEVICE=cuda:0  BATCH_SIZE=16  NUM_WORKERS=4
#   ROBUSTNESS_OUTPUT_DIR=outputs_experiments/robustness

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

RUN_ROOT="${RUN_ROOT:-outputs_experiments/pmdfnet_mmlsv2}"
ROBUSTNESS_IDS="${ROBUSTNESS_IDS:-M4,L1}"
ROBUSTNESS_SEEDS="${ROBUSTNESS_SEEDS:-41 42 43}"
ROBUSTNESS_OUTPUT_DIR="${ROBUSTNESS_OUTPUT_DIR:-outputs_experiments/robustness}"
ROBUSTNESS_CONFIG="${ROBUSTNESS_CONFIG:-configs/mmlsv2_robustness_protocol.json}"
EVAL_BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_NUM_WORKERS="${NUM_WORKERS:-4}"

if [ -z "${DATA_ROOT:-}" ]; then
  echo "DATA_ROOT is empty; set DATA_ROOT or MMLSV2_DATA_ROOT in .env" >&2
  exit 2
fi

echo "######## robustness: ids=${ROBUSTNESS_IDS} seeds=${ROBUSTNESS_SEEDS} device=${DEVICE} ########"
"${PYTHON_BIN}" tools/evaluation/evaluate_robustness.py \
  --config "${CONFIG}" \
  --robustness-config "${ROBUSTNESS_CONFIG}" \
  --run-root "${RUN_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --ids "${ROBUSTNESS_IDS}" \
  --seeds "${ROBUSTNESS_SEEDS}" \
  --device "${DEVICE}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${EVAL_NUM_WORKERS}" \
  --output-dir "${ROBUSTNESS_OUTPUT_DIR}"
