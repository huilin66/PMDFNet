#!/usr/bin/env bash
set -euo pipefail

# Purpose: reproduce the historical competition configuration on the mapped
# MMLSv2 release. This group uses train -> val only and defaults to the
# historical Small chv1+add model.

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${MARS_DATA_ROOT:-${DATA_ROOT:-/scrinvme/huilin/bdd/cp_data/mmlsv2_mapped_mars_ls}}"
DEVICE="${DEVICE:-${TRAIN_DEVICE:-cuda:0}}"
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-${TRAIN_BATCH_SIZE:-0}}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
NUM_WORKERS="${NUM_WORKERS:-${TRAIN_NUM_WORKERS:-4}}"
STRICT_DETERMINISM="${STRICT_DETERMINISM:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_experiments/reproduction}"
FINAL_OUTPUT_ROOT="${FINAL_OUTPUT_ROOT:-${DATA_ROOT}/outputs}"
GROUP_NAME="${GROUP_NAME:-competition_reproduction}"
MODEL_NAMES="${MODEL_NAMES:-dual_segformer_convnextsmall_chv1_add}"

case "${STRICT_DETERMINISM}" in
  1|true|TRUE|yes|YES|on|ON) STRICT_FLAG="--strict-determinism" ;;
  *) STRICT_FLAG="--no-strict-determinism" ;;
esac

if [ ! -d "${DATA_ROOT}/train/images" ] || [ ! -d "${DATA_ROOT}/val/masks" ]; then
  echo "Dataset is incomplete: ${DATA_ROOT}" >&2
  echo "Set MARS_DATA_ROOT or DATA_ROOT to the mapped dataset root." >&2
  exit 1
fi

export MARS_DATA_ROOT="${DATA_ROOT}"
export DATA_ROOT="${DATA_ROOT}"
export PYTHONUNBUFFERED=1

echo "[${GROUP_NAME}] data=${DATA_ROOT}"
echo "[${GROUP_NAME}] models=${MODEL_NAMES}"
echo "[${GROUP_NAME}] seeds=${SEEDS} device=${DEVICE} epochs=${EPOCHS}"

for model_name in ${MODEL_NAMES}; do
  for seed in ${SEEDS}; do
    run_output="${OUTPUT_ROOT}/${model_name}/seed${seed}"
    echo "=== ${GROUP_NAME}: ${model_name}, seed ${seed} ==="
    PYTHONHASHSEED="${seed}" \
    CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    TRAIN_STRICT_DETERMINISM="${STRICT_DETERMINISM}" \
    TRAIN_NUM_WORKERS="${NUM_WORKERS}" \
    "${PYTHON_BIN}" scripts/train_competition_reproduction.py \
      --model-name "${model_name}" \
      --data-root "${DATA_ROOT}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --output-root "${run_output}" \
      --final-output-root "${FINAL_OUTPUT_ROOT}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --val-interval "${VAL_INTERVAL}" \
      --num-workers "${NUM_WORKERS}" \
      "${STRICT_FLAG}"
  done
done

echo "[${GROUP_NAME}] completed"
