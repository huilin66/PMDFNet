#!/usr/bin/env sh
set -eu

# Frozen-checkpoint Test diagnostics.  This script does not train and does not
# rewrite per-run test_metrics.json; it writes auxiliary diagnostics only.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

if [ -z "${DATA_ROOT:-}" ]; then
  echo "DATA_ROOT is empty; set DATA_ROOT or MMLSV2_DATA_ROOT" >&2
  exit 2
fi

DIAGNOSTIC_CONFIG="${DIAGNOSTIC_CONFIG:-configs/mmlsv2_paper_protocol.json}"
DIAGNOSTIC_IDS="${DIAGNOSTIC_IDS:-M4,L1}"
DIAGNOSTIC_SEEDS="${DIAGNOSTIC_SEEDS:-${SEEDS}}"
DIAGNOSTIC_RUN_ROOT="${DIAGNOSTIC_RUN_ROOT:-${RUN_ROOT:-outputs_experiments/pmdfnet_mmlsv2}}"
DIAGNOSTIC_OUTPUT_DIR="${DIAGNOSTIC_OUTPUT_DIR:-outputs_experiments/test_diagnostics}"
DIAGNOSTIC_BATCH_SIZE="${DIAGNOSTIC_BATCH_SIZE:-8}"
DIAGNOSTIC_BOUNDARY_TOLERANCE="${DIAGNOSTIC_BOUNDARY_TOLERANCE:-2}"
DIAGNOSTIC_NO_PLOTS="${DIAGNOSTIC_NO_PLOTS:-0}"

set -- \
  --data-root "${DATA_ROOT}" \
  --run-root "${DIAGNOSTIC_RUN_ROOT}" \
  --config "${DIAGNOSTIC_CONFIG}" \
  --ids "${DIAGNOSTIC_IDS}" \
  --seeds "${DIAGNOSTIC_SEEDS}" \
  --output-dir "${DIAGNOSTIC_OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --batch-size "${DIAGNOSTIC_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --boundary-tolerance "${DIAGNOSTIC_BOUNDARY_TOLERANCE}"

if [ "${DIAGNOSTIC_NO_PLOTS}" != "0" ]; then
  set -- "$@" --no-plots
fi

echo "######## frozen Test diagnostics ########"
echo "config=${DIAGNOSTIC_CONFIG} run_root=${DIAGNOSTIC_RUN_ROOT} ids=${DIAGNOSTIC_IDS} seeds=${DIAGNOSTIC_SEEDS}"
echo "output=${DIAGNOSTIC_OUTPUT_DIR} boundary_tolerance=${DIAGNOSTIC_BOUNDARY_TOLERANCE}"
"${PYTHON_BIN}" tools/evaluation/evaluate_test_diagnostics.py "$@"
