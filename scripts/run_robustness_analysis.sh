#!/usr/bin/env sh
set -eu

# Analyze the already completed paired robustness output and render the
# deterministic clean-Test qualitative panels selected by failure_cases.json.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

ROBUSTNESS_OUTPUT_DIR="${ROBUSTNESS_OUTPUT_DIR:-outputs_experiments/robustness}"
ROBUSTNESS_ANALYSIS_DIR="${ROBUSTNESS_ANALYSIS_DIR:-${ROBUSTNESS_OUTPUT_DIR}/analysis}"
REFERENCE_ID="${REFERENCE_ID:-M4}"
CANDIDATE_ID="${CANDIDATE_ID:-L1}"
BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-10000}"
PERMUTATION_RESAMPLES="${PERMUTATION_RESAMPLES:-10000}"
FAILURE_SEED="${FAILURE_SEED:-41}"
FAILURE_CONDITION="${FAILURE_CONDITION:-clean}"
FAILURE_TOP_K="${FAILURE_TOP_K:-5}"
QUALITATIVE_OUTPUT_DIR="${QUALITATIVE_OUTPUT_DIR:-${ROBUSTNESS_OUTPUT_DIR}/qualitative}"

"${PYTHON_BIN}" tools/evaluation/analyze_robustness.py \
  --input "${ROBUSTNESS_OUTPUT_DIR}/robustness_per_image.jsonl" \
  --output-dir "${ROBUSTNESS_ANALYSIS_DIR}" \
  --reference-id "${REFERENCE_ID}" \
  --candidate-id "${CANDIDATE_ID}" \
  --bootstrap-resamples "${BOOTSTRAP_RESAMPLES}" \
  --permutation-resamples "${PERMUTATION_RESAMPLES}" \
  --failure-seed "${FAILURE_SEED}" \
  --failure-condition "${FAILURE_CONDITION}" \
  --top-k "${FAILURE_TOP_K}"

if [ -z "${DATA_ROOT:-}" ]; then
  echo "DATA_ROOT is empty; set DATA_ROOT or MMLSV2_DATA_ROOT for qualitative rendering" >&2
  exit 2
fi

"${PYTHON_BIN}" tools/evaluation/render_qualitative_cases.py \
  --cases-json "${ROBUSTNESS_ANALYSIS_DIR}/failure_cases.json" \
  --data-root "${DATA_ROOT}" \
  --run-root "${RUN_ROOT:-outputs_experiments/pmdfnet_mmlsv2}" \
  --config "${CONFIG:-configs/mmlsv2_paper_protocol.json}" \
  --reference-id "${REFERENCE_ID}" \
  --candidate-id "${CANDIDATE_ID}" \
  --device "${DEVICE:-cuda:0}" \
  --output-dir "${QUALITATIVE_OUTPUT_DIR}"
