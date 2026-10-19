#!/usr/bin/env sh
set -eu

# Smoke test 1: verify CLI parsing, .env data root, dataset import, and one sample read.
# This does not initialize a model or run training.

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

. "${REPO_ROOT}/scripts/_env.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_experiments/smoke/data}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_DIR}/smoke_data_summary.csv}"
SMOKE_STATUS_CSV="${SMOKE_STATUS_CSV:-${OUTPUT_DIR}/smoke_status.csv}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-test}"
INPUT_SIZE="${INPUT_SIZE:-64}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CURRENT_STEP="init"

mkdir -p "${OUTPUT_DIR}"

append_smoke_status() {
  exit_code="$1"
  if [ "${exit_code}" = "0" ]; then
    status="passed"
  else
    status="failed"
  fi
  if [ ! -f "${SMOKE_STATUS_CSV}" ]; then
    printf '%s\n' "completed_at,script,status,exit_code,failed_or_last_step,output_dir,summary_csv,train_split,val_split,input_size,max_train_samples,max_val_samples,num_workers" >>"${SMOKE_STATUS_CSV}"
  fi
  completed_at="$(date '+%Y-%m-%dT%H:%M:%S')"
  printf '%s\n' "${completed_at},smoke_test_data.sh,${status},${exit_code},${CURRENT_STEP},${OUTPUT_DIR},${SUMMARY_CSV},${TRAIN_SPLIT},${VAL_SPLIT},${INPUT_SIZE},${MAX_TRAIN_SAMPLES},${MAX_VAL_SAMPLES},${NUM_WORKERS}" >>"${SMOKE_STATUS_CSV}"
}

on_exit() {
  exit_code="$1"
  append_smoke_status "${exit_code}"
}

trap 'on_exit "$?"' EXIT

echo "=== Smoke test: Python syntax ==="
CURRENT_STEP="python_compile"
"${PYTHON_BIN}" -m py_compile train_ablation.py env_utils.py

echo
echo "=== Smoke test: train_ablation.py --help ==="
CURRENT_STEP="train_ablation_help"
"${PYTHON_BIN}" train_ablation.py --help >/dev/null

echo
echo "=== Smoke test: dataset dry-run ==="
CURRENT_STEP="dataset_dry_run"
if [ -n "${DATA_ROOT:-}" ]; then
  "${PYTHON_BIN}" train_ablation.py \
    --data-root "${DATA_ROOT}" \
    --experiment-name smoke_data \
    --output-dir "${OUTPUT_DIR}" \
    --summary-csv "${SUMMARY_CSV}" \
    --train-split "${TRAIN_SPLIT}" \
    --val-split "${VAL_SPLIT}" \
    --input-size "${INPUT_SIZE}" \
    --batch-size 1 \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --num-workers "${NUM_WORKERS}" \
    --dry-run \
    --skip-model-init
else
  "${PYTHON_BIN}" train_ablation.py \
    --experiment-name smoke_data \
    --output-dir "${OUTPUT_DIR}" \
    --summary-csv "${SUMMARY_CSV}" \
    --train-split "${TRAIN_SPLIT}" \
    --val-split "${VAL_SPLIT}" \
    --input-size "${INPUT_SIZE}" \
    --batch-size 1 \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --num-workers "${NUM_WORKERS}" \
    --dry-run \
    --skip-model-init
fi

echo
echo "Smoke data test passed."
echo "Output dir: ${OUTPUT_DIR}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Smoke status CSV: ${SMOKE_STATUS_CSV}"
