#!/usr/bin/env sh
set -eu

# Evaluate frozen G0-G3 modality-grouping checkpoints.  G0/G1 are resolved
# from the formal protocol (L1/B2); G2/G3 are resolved from the supplementary
# grouping protocol.  The script does not train and does not rewrite the
# individual run test_metrics.json files.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

if [ -z "${DATA_ROOT:-}" ]; then
  echo "DATA_ROOT is empty; set DATA_ROOT or MMLSV2_DATA_ROOT" >&2
  exit 2
fi

GROUPING_MAIN_CONFIG="${GROUPING_MAIN_CONFIG:-configs/mmlsv2_paper_protocol.json}"
GROUPING_CONFIG="${GROUPING_CONFIG:-configs/mmlsv2_modality_grouping_protocol.json}"
GROUPING_MAIN_RUN_ROOT="${GROUPING_MAIN_RUN_ROOT:-outputs_experiments/pmdfnet_mmlsv2}"
GROUPING_RUN_ROOT="${GROUPING_RUN_ROOT:-outputs_experiments/pmdfnet_mmlsv2_modality_grouping}"
GROUPING_OUTPUT_DIR="${GROUPING_OUTPUT_DIR:-outputs_experiments/modality_grouping}"
GROUPING_IDS="${GROUPING_IDS:-G0,G1,G2,G3}"
GROUPING_SEEDS="${GROUPING_SEEDS:-${SEEDS}}"
GROUPING_BATCH_SIZE="${GROUPING_BATCH_SIZE:-16}"
GROUPING_BOUNDARY_TOLERANCE="${GROUPING_BOUNDARY_TOLERANCE:-2}"
GROUPING_BOOTSTRAP_RESAMPLES="${GROUPING_BOOTSTRAP_RESAMPLES:-10000}"
GROUPING_PERMUTATION_RESAMPLES="${GROUPING_PERMUTATION_RESAMPLES:-10000}"
GROUPING_COMPARISONS="${GROUPING_COMPARISONS:-G0:G1,G0:G2,G0:G3}"
GROUPING_PRIMARY_COMPARISON="${GROUPING_PRIMARY_COMPARISON:-G0:G1}"

echo "######## modality grouping evaluation ########"
echo "groups=${GROUPING_IDS} seeds=${GROUPING_SEEDS} output=${GROUPING_OUTPUT_DIR}"

"${PYTHON_BIN}" tools/evaluation/evaluate_modality_grouping.py \
  --data-root "${DATA_ROOT}" \
  --main-run-root "${GROUPING_MAIN_RUN_ROOT}" \
  --main-config "${GROUPING_MAIN_CONFIG}" \
  --grouping-run-root "${GROUPING_RUN_ROOT}" \
  --grouping-config "${GROUPING_CONFIG}" \
  --groups "${GROUPING_IDS}" \
  --seeds "${GROUPING_SEEDS}" \
  --output-dir "${GROUPING_OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --batch-size "${GROUPING_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --boundary-tolerance "${GROUPING_BOUNDARY_TOLERANCE}" \
  --bootstrap-resamples "${GROUPING_BOOTSTRAP_RESAMPLES}" \
  --permutation-resamples "${GROUPING_PERMUTATION_RESAMPLES}" \
  --comparisons "${GROUPING_COMPARISONS}" \
  --primary-comparison "${GROUPING_PRIMARY_COMPARISON}"
