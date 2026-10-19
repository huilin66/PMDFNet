#!/usr/bin/env sh
set -eu

# Train PMDFNet ablation experiments for post-release MMLSv2 analysis.
# This script only runs experiments; it does not generate paper text.
#
# Typical use:
#   sh scripts/train_ablation_experiments.sh
#
# Useful overrides:
#   EPOCHS=50 DEVICE=cuda:0 RUN_BACKBONE=0 sh scripts/train_ablation_experiments.sh

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_experiments/ablation}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_DIR}/ablation_summary.csv}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-test}"
EPOCHS="${EPOCHS:-100}"
INPUT_SIZE="${INPUT_SIZE:-128}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SCHEDULER="${SCHEDULER:-cosine}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
NORMALIZATION="${NORMALIZATION:-auto}"
MIXED_PRECISION="${MIXED_PRECISION:-false}"
DETERMINISTIC="${DETERMINISTIC:-true}"
RUN_BACKBONE="${RUN_BACKBONE:-1}"

run_exp() {
  exp_name="$1"
  model_name="$2"
  batch_size="$3"
  loss="$4"
  augmentation="$5"
  mosaic_prob="$6"
  echo
  echo "=== ${exp_name}: ${model_name}, loss=${loss}, aug=${augmentation}, mosaic=${mosaic_prob} ==="

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
      --batch-size "${batch_size}" \
      --loss "${loss}" \
      --augmentation "${augmentation}" \
      --mosaic-prob "${mosaic_prob}" \
      --aug-prob 0.5
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
      --batch-size "${batch_size}" \
      --loss "${loss}" \
      --augmentation "${augmentation}" \
      --mosaic-prob "${mosaic_prob}" \
      --aug-prob 0.5
  fi
}

# 1. Architecture contribution: single-stream baseline vs dual-stream.
run_exp "abl_01_single_segformer_tiny" "segformer_convnexttiny" 32 "unetformer" "mars" 0.5
run_exp "abl_02_dual_tiny_chv1_add" "dual_segformer_convnexttiny_chv1_add" 16 "unetformer" "mars" 0.5

# 2. Channel grouping.
run_exp "abl_03_dual_tiny_chv2_add" "dual_segformer_convnexttiny_chv2_add" 16 "unetformer" "mars" 0.5
run_exp "abl_04_dual_tiny_chv3_add" "dual_segformer_convnexttiny_chv3_add" 16 "unetformer" "mars" 0.5

# 3. Fusion strategy under the same backbone and channel split.
run_exp "abl_05_dual_tiny_chv1_cat" "dual_segformer_convnexttiny_chv1_cat" 16 "unetformer" "mars" 0.5
run_exp "abl_06_dual_tiny_chv1_att" "dual_segformer_convnexttiny_chv1_att" 16 "unetformer" "mars" 0.5
run_exp "abl_07_dual_tiny_chv1_moe" "dual_segformer_convnexttiny_chv1_moe" 16 "unetformer" "mars" 0.5
run_exp "abl_08_dual_tiny_chv1_moev2" "dual_segformer_convnexttiny_chv1_moev2" 16 "unetformer" "mars" 0.5

# 4. Loss function.
run_exp "abl_09_dual_tiny_loss_combined" "dual_segformer_convnexttiny_chv1_add" 16 "combined" "mars" 0.5
run_exp "abl_10_dual_tiny_loss_ce" "dual_segformer_convnexttiny_chv1_add" 16 "ce" "mars" 0.5
run_exp "abl_11_dual_tiny_loss_dice" "dual_segformer_convnexttiny_chv1_add" 16 "dice" "mars" 0.5

# 5. Augmentation and mosaic.
run_exp "abl_12_dual_tiny_no_mosaic" "dual_segformer_convnexttiny_chv1_add" 16 "unetformer" "mars" 0.0
run_exp "abl_13_dual_tiny_no_aug_no_mosaic" "dual_segformer_convnexttiny_chv1_add" 16 "unetformer" "none" 0.0

# 6. Backbone scale. Keep this optional because base/large runs are expensive.
if [ "${RUN_BACKBONE}" = "1" ]; then
  run_exp "abl_14_dual_small_chv1_add" "dual_segformer_convnextsmall_chv1_add" 16 "unetformer" "mars" 0.5
  run_exp "abl_15_dual_base_chv1_add" "dual_segformer_convnextbase_chv1_add" 4 "unetformer" "mars" 0.5
  run_exp "abl_16_dual_large_chv1_add" "dual_segformer_convnextlarge_chv1_add" 2 "unetformer" "mars" 0.5
else
  echo
  echo "Skipping backbone scale ablation because RUN_BACKBONE=${RUN_BACKBONE}."
fi

echo
echo "Ablation training finished. Results are under ${OUTPUT_DIR}."
echo "Summary CSV: ${SUMMARY_CSV}"
