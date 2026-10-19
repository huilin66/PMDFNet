#!/usr/bin/env sh
set -eu

# Chv1 fusion exploration experiments.
# Centered on the best channel split: chv1 ([0,1,2,3] / [4,5,6]).
#   - dataA (competition): chv1_add best (online 0.8665)
#   - dataB (mmlsv2):      chv1_cat best (iou_fg 0.8210)
# This script explores all fusion strategies under chv1 with loss/aug/size variations.
#
# Typical use:
#   sh scripts/train_chv1_fusion_experiments.sh
#
# Useful overrides:
#   EPOCHS=50 DEVICE=cuda:0 RUN_LOSS=0 SEEDS="42 123 7" sh scripts/train_chv1_fusion_experiments.sh

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_experiments/chv1_fusion}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_DIR}/chv1_fusion_summary.csv}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-test}"
ENCODER="${ENCODER:-tu-convnext_tiny}"
PRETRAIN="${PRETRAIN:-true}"
EPOCHS="${EPOCHS:-100}"
INPUT_SIZE="${INPUT_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SCHEDULER="${SCHEDULER:-cosine}"
AUG_PROB="${AUG_PROB:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-42}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
NORMALIZATION="${NORMALIZATION:-auto}"
MIXED_PRECISION="${MIXED_PRECISION:-false}"
DETERMINISTIC="${DETERMINISTIC:-true}"
PRIMARY_METRIC="${PRIMARY_METRIC:-miou}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
MIN_DELTA="${MIN_DELTA:-0.0}"
DATASET_TYPE="${DATASET_TYPE:-mmlsv2}"
RUN_LOSS="${RUN_LOSS:-1}"
RUN_SIZE="${RUN_SIZE:-0}"
RUN_NOAUG="${RUN_NOAUG:-0}"
RUN_BACKBONE="${RUN_BACKBONE:-0}"
SIZE_INPUT="${SIZE_INPUT:-256}"
SIZE_BATCH_SIZE="${SIZE_BATCH_SIZE:-8}"

run_exp() {
  exp_prefix="$1"
  fusion="$2"
  loss="$3"
  augmentation="$4"
  mosaic_prob="$5"
  input_size="$6"
  batch_size="$7"
  encoder="${8:-${ENCODER}}"

  for seed in ${SEEDS}; do
    exp_name="${exp_prefix}_seed${seed}"
    echo
    echo "=== ${exp_name}: chv1 ${fusion}, loss=${loss}, aug=${augmentation}, mosaic=${mosaic_prob}, size=${input_size}, enc=${encoder} ==="

    if [ -n "${DATA_ROOT:-}" ]; then
      "${PYTHON_BIN}" train_ablation.py \
        --data-root "${DATA_ROOT}" \
        --train-split "${TRAIN_SPLIT}" \
        --val-split "${VAL_SPLIT}" \
        --output-dir "${OUTPUT_DIR}" \
        --summary-csv "${SUMMARY_CSV}" \
        --model-name auto \
        --arch pmdfnet \
        --encoder "${encoder}" \
        --pretrain "${PRETRAIN}" \
        --channels1 "0,1,2,3" \
        --channels2 "4,5,6" \
        --fusion "${fusion}" \
        --epochs "${EPOCHS}" \
        --input-size "${input_size}" \
        --batch-size "${batch_size}" \
        --lr "${LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --optimizer "${OPTIMIZER}" \
        --scheduler "${SCHEDULER}" \
        --loss "${loss}" \
        --augmentation "${augmentation}" \
        --aug-prob "${AUG_PROB}" \
        --mosaic-prob "${mosaic_prob}" \
        --num-workers "${NUM_WORKERS}" \
        --device "${DEVICE}" \
        --seed "${seed}" \
        --val-interval "${VAL_INTERVAL}" \
        --save-interval "${SAVE_INTERVAL}" \
        --normalization "${NORMALIZATION}" \
        --mixed-precision "${MIXED_PRECISION}" \
        --deterministic "${DETERMINISTIC}" \
        --primary-metric "${PRIMARY_METRIC}" \
        --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
        --min-delta "${MIN_DELTA}" \
        --dataset-type "${DATASET_TYPE}" \
        --experiment-name "${exp_name}"
    else
      "${PYTHON_BIN}" train_ablation.py \
        --train-split "${TRAIN_SPLIT}" \
        --val-split "${VAL_SPLIT}" \
        --output-dir "${OUTPUT_DIR}" \
        --summary-csv "${SUMMARY_CSV}" \
        --model-name auto \
        --arch pmdfnet \
        --encoder "${encoder}" \
        --pretrain "${PRETRAIN}" \
        --channels1 "0,1,2,3" \
        --channels2 "4,5,6" \
        --fusion "${fusion}" \
        --epochs "${EPOCHS}" \
        --input-size "${input_size}" \
        --batch-size "${batch_size}" \
        --lr "${LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --optimizer "${OPTIMIZER}" \
        --scheduler "${SCHEDULER}" \
        --loss "${loss}" \
        --augmentation "${augmentation}" \
        --aug-prob "${AUG_PROB}" \
        --mosaic-prob "${mosaic_prob}" \
        --num-workers "${NUM_WORKERS}" \
        --device "${DEVICE}" \
        --seed "${seed}" \
        --val-interval "${VAL_INTERVAL}" \
        --save-interval "${SAVE_INTERVAL}" \
        --normalization "${NORMALIZATION}" \
        --mixed-precision "${MIXED_PRECISION}" \
        --deterministic "${DETERMINISTIC}" \
        --primary-metric "${PRIMARY_METRIC}" \
        --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
        --min-delta "${MIN_DELTA}" \
        --dataset-type "${DATASET_TYPE}" \
        --experiment-name "${exp_name}"
    fi
  done
}

# ============================================================
# 1. Core fusion comparison (5 methods, unetformer loss)
# ============================================================
run_exp "exp_01_chv1_add"   "add"   "unetformer" "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
run_exp "exp_02_chv1_cat"   "cat"   "unetformer" "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
run_exp "exp_03_chv1_att"   "att"   "unetformer" "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
run_exp "exp_04_chv1_moe"   "moe"   "unetformer" "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
run_exp "exp_05_chv1_moev2" "moev2" "unetformer" "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"

# ============================================================
# 2. Loss exploration for top-2 fusions (cat, add)
# ============================================================
if [ "${RUN_LOSS}" = "1" ]; then
  run_exp "exp_06_chv1_cat_ce"       "cat" "ce"       "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
  run_exp "exp_07_chv1_cat_dice"     "cat" "dice"     "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
  run_exp "exp_08_chv1_cat_combined" "cat" "combined" "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
  run_exp "exp_09_chv1_add_ce"       "add" "ce"       "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
  run_exp "exp_10_chv1_add_dice"     "add" "dice"     "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
  run_exp "exp_11_chv1_add_combined" "add" "combined" "mars" "0.5" "${INPUT_SIZE}" "${BATCH_SIZE}"
else
  echo
  echo "Skipping loss exploration because RUN_LOSS=${RUN_LOSS}."
fi

# ============================================================
# 3. Mosaic / augmentation ablation for best fusion (cat)
# ============================================================
run_exp "exp_12_chv1_cat_no_mosaic"     "cat" "unetformer" "mars" "0.0" "${INPUT_SIZE}" "${BATCH_SIZE}"

if [ "${RUN_NOAUG}" = "1" ]; then
  run_exp "exp_13_chv1_cat_no_aug_no_mosaic" "cat" "unetformer" "none" "0.0" "${INPUT_SIZE}" "${BATCH_SIZE}"
  run_exp "exp_14_chv1_add_no_aug_no_mosaic" "add" "unetformer" "none" "0.0" "${INPUT_SIZE}" "${BATCH_SIZE}"
else
  echo
  echo "Skipping no-augmentation follow-up because RUN_NOAUG=${RUN_NOAUG}."
fi

# ============================================================
# 4. Input size exploration
# ============================================================
if [ "${RUN_SIZE}" = "1" ]; then
  run_exp "exp_15_chv1_cat_size${SIZE_INPUT}" "cat" "unetformer" "mars" "0.5" "${SIZE_INPUT}" "${SIZE_BATCH_SIZE}"
  run_exp "exp_16_chv1_add_size${SIZE_INPUT}" "add" "unetformer" "mars" "0.5" "${SIZE_INPUT}" "${SIZE_BATCH_SIZE}"
else
  echo
  echo "Skipping input-size follow-up because RUN_SIZE=${RUN_SIZE}."
fi

# ============================================================
# 5. Backbone scale for best fusion (cat)
# ============================================================
if [ "${RUN_BACKBONE}" = "1" ]; then
  run_exp "exp_17_chv1_cat_small" "cat" "unetformer" "mars" "0.5" "${INPUT_SIZE}" "16" "tu-convnext_small"
  run_exp "exp_18_chv1_cat_base"  "cat" "unetformer" "mars" "0.5" "${INPUT_SIZE}" "4"  "tu-convnext_base"
else
  echo
  echo "Skipping backbone scale because RUN_BACKBONE=${RUN_BACKBONE}."
fi

echo
echo "Chv1 fusion exploration finished. Results are under ${OUTPUT_DIR}."
echo "Summary CSV: ${SUMMARY_CSV}"
