#!/usr/bin/env sh
set -eu

# Smoke test 2: run a tiny one-epoch training/evaluation loop.
# It uses the generic arch/encoder path with pretrain disabled to avoid network downloads.

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_experiments/smoke/train}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_DIR}/smoke_train_summary.csv}"
SMOKE_STATUS_CSV="${SMOKE_STATUS_CSV:-${OUTPUT_DIR}/smoke_status.csv}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-test}"
SMOKE_ARCH="${SMOKE_ARCH:-${MODEL_ARCH:-segformer}}"
ENCODER="${ENCODER:-tu-convnext_tiny}"
LOSS="${LOSS:-ce}"
INPUT_SIZE="${INPUT_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-4}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-4}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SCHEDULER="${SCHEDULER:-none}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
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
    printf '%s\n' "completed_at,script,status,exit_code,failed_or_last_step,output_dir,summary_csv,train_split,val_split,arch,encoder,loss,input_size,batch_size,max_train_samples,max_val_samples,num_workers,device,seed" >>"${SMOKE_STATUS_CSV}"
  fi
  completed_at="$(date '+%Y-%m-%dT%H:%M:%S')"
  printf '%s\n' "${completed_at},smoke_test_train.sh,${status},${exit_code},${CURRENT_STEP},${OUTPUT_DIR},${SUMMARY_CSV},${TRAIN_SPLIT},${VAL_SPLIT},${SMOKE_ARCH},${ENCODER},${LOSS},${INPUT_SIZE},${BATCH_SIZE},${MAX_TRAIN_SAMPLES},${MAX_VAL_SAMPLES},${NUM_WORKERS},${DEVICE},${SEED}" >>"${SMOKE_STATUS_CSV}"
}

on_exit() {
  exit_code="$1"
  append_smoke_status "${exit_code}"
}

trap 'on_exit "$?"' EXIT

echo "=== Smoke test: tiny train/eval loop ==="
CURRENT_STEP="tiny_train_eval"
if [ -n "${DATA_ROOT:-}" ]; then
  "${PYTHON_BIN}" train_ablation.py \
    --data-root "${DATA_ROOT}" \
    --experiment-name smoke_train \
    --output-dir "${OUTPUT_DIR}" \
    --summary-csv "${SUMMARY_CSV}" \
    --train-split "${TRAIN_SPLIT}" \
    --val-split "${VAL_SPLIT}" \
    --model-name "" \
    --arch "${SMOKE_ARCH}" \
    --encoder "${ENCODER}" \
    --pretrain false \
    --loss "${LOSS}" \
    --epochs 1 \
    --input-size "${INPUT_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --optimizer "${OPTIMIZER}" \
    --scheduler "${SCHEDULER}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --mixed-precision false \
    --deterministic true
else
  "${PYTHON_BIN}" train_ablation.py \
    --experiment-name smoke_train \
    --output-dir "${OUTPUT_DIR}" \
    --summary-csv "${SUMMARY_CSV}" \
    --train-split "${TRAIN_SPLIT}" \
    --val-split "${VAL_SPLIT}" \
    --model-name "" \
    --arch "${SMOKE_ARCH}" \
    --encoder "${ENCODER}" \
    --pretrain false \
    --loss "${LOSS}" \
    --epochs 1 \
    --input-size "${INPUT_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-val-samples "${MAX_VAL_SAMPLES}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --optimizer "${OPTIMIZER}" \
    --scheduler "${SCHEDULER}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --mixed-precision false \
    --deterministic true
fi

echo
echo "Smoke train test passed."
echo "Output dir: ${OUTPUT_DIR}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Smoke status CSV: ${SMOKE_STATUS_CSV}"
