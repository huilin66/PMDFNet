#!/usr/bin/env sh
set -eu

# Train comparison models for post-release MMLSv2 experiments.
# This script only runs experiments; it does not generate paper text.
#
# Typical use:
#   sh scripts/train_comparison_models.sh
#
# Useful overrides:
#   EPOCHS=50 DEVICE=cuda:0 RUN_HEAVY=0 sh scripts/train_comparison_models.sh

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_experiments/comparison}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_DIR}/comparison_summary.csv}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-test}"
EPOCHS="${EPOCHS:-100}"
INPUT_SIZE="${INPUT_SIZE:-128}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SCHEDULER="${SCHEDULER:-cosine}"
LOSS="${LOSS:-unetformer}"
AUGMENTATION="${AUGMENTATION:-mars}"
AUG_PROB="${AUG_PROB:-0.5}"
MOSAIC_PROB="${MOSAIC_PROB:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
NORMALIZATION="${NORMALIZATION:-auto}"
MIXED_PRECISION="${MIXED_PRECISION:-false}"
DETERMINISTIC="${DETERMINISTIC:-true}"
RUN_HEAVY="${RUN_HEAVY:-1}"

run_exp() {
  exp_name="$1"
  model_name="$2"
  batch_size="$3"
  echo
  echo "=== ${exp_name}: ${model_name} ==="

  if [ -n "${DATA_ROOT:-}" ]; then
    "${PYTHON_BIN}" train_ablation.py \
      --data-root "${DATA_ROOT}" \
      --train-split "${TRAIN_SPLIT}" \
      --val-split "${VAL_SPLIT}" \
      --output-dir "${OUTPUT_DIR}" \
      --summary-csv "${SUMMARY_CSV}" \
      --epochs "${EPOCHS}" \
      --input-size "${INPUT_SIZE}" \
      --lr "${LR}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --optimizer "${OPTIMIZER}" \
      --scheduler "${SCHEDULER}" \
      --loss "${LOSS}" \
      --augmentation "${AUGMENTATION}" \
      --aug-prob "${AUG_PROB}" \
      --mosaic-prob "${MOSAIC_PROB}" \
      --num-workers "${NUM_WORKERS}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --val-interval "${VAL_INTERVAL}" \
      --save-interval "${SAVE_INTERVAL}" \
      --normalization "${NORMALIZATION}" \
      --mixed-precision "${MIXED_PRECISION}" \
      --deterministic "${DETERMINISTIC}" \
      --experiment-name "${exp_name}" \
      --model-name "${model_name}" \
      --batch-size "${batch_size}"
  else
    "${PYTHON_BIN}" train_ablation.py \
      --train-split "${TRAIN_SPLIT}" \
      --val-split "${VAL_SPLIT}" \
      --output-dir "${OUTPUT_DIR}" \
      --summary-csv "${SUMMARY_CSV}" \
      --epochs "${EPOCHS}" \
      --input-size "${INPUT_SIZE}" \
      --lr "${LR}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --optimizer "${OPTIMIZER}" \
      --scheduler "${SCHEDULER}" \
      --loss "${LOSS}" \
      --augmentation "${AUGMENTATION}" \
      --aug-prob "${AUG_PROB}" \
      --mosaic-prob "${MOSAIC_PROB}" \
      --num-workers "${NUM_WORKERS}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --val-interval "${VAL_INTERVAL}" \
      --save-interval "${SAVE_INTERVAL}" \
      --normalization "${NORMALIZATION}" \
      --mixed-precision "${MIXED_PRECISION}" \
      --deterministic "${DETERMINISTIC}" \
      --experiment-name "${exp_name}" \
      --model-name "${model_name}" \
      --batch-size "${batch_size}"
  fi
}

run_exp "cmp_01_m3lsnet_vssd" "m3lsnet" 32
run_exp "cmp_02_ocrnet_hrnet_w48" "ocrnet_hrnet_w48" 8
run_exp "cmp_03_upernet_convnext_tiny" "upernet_convnexttiny" 32
run_exp "cmp_04_segformer_mit_b2" "segformer_mitb2" 32
run_exp "cmp_05_segformer_convnext_tiny" "segformer_convnexttiny" 32
run_exp "cmp_06_segformer_convnext_small" "segformer_convnextsmall" 16
run_exp "cmp_07_pmdfnet_convnext_tiny" "dual_segformer_convnexttiny_chv1_add" 16
run_exp "cmp_08_pmdfnet_convnext_small" "dual_segformer_convnextsmall_chv1_add" 16

if [ "${RUN_HEAVY}" = "1" ]; then
  run_exp "cmp_09_pmdfnet_convnext_base" "dual_segformer_convnextbase_chv1_add" 4
  run_exp "cmp_10_pmdfnet_convnext_large" "dual_segformer_convnextlarge_chv1_add" 2
else
  echo
  echo "Skipping base/large PMDFNet because RUN_HEAVY=${RUN_HEAVY}."
fi

echo
echo "Comparison training finished. Results are under ${OUTPUT_DIR}."
echo "Summary CSV: ${SUMMARY_CSV}"
