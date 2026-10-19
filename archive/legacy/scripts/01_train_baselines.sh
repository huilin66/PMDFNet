#!/usr/bin/env sh
set -eu

# Official Stage 1 baselines. All checkpoint selection happens on val only;
# train_skg.py does not load the test split.
#
# Example:
#   DATA_ROOT=/path/to/mmlsv2 SEEDS="42 123 7" sh scripts/01_train_baselines.sh

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -n "${DATA_ROOT:-}" ]; then
  export MMLSV2_DATA_ROOT="${DATA_ROOT}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-outputs_skg/baselines}"
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-25}"
INPUT_SIZE="${INPUT_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STRICT_DETERMINISM="${STRICT_DETERMINISM:-1}"
DEVICE="${DEVICE:-auto}"
PRETRAIN="${PRETRAIN:-true}"
MIXED_PRECISION="${MIXED_PRECISION:-true}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"

run_baseline() {
  experiment="$1"
  family="$2"
  encoder="$3"
  fusion="$4"
  for seed in ${SEEDS}; do
    echo "=== ${experiment}, seed ${seed} ==="
    PYTHONHASHSEED="${seed}" \
    CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    TRAIN_STRICT_DETERMINISM="${STRICT_DETERMINISM}" \
    "${PYTHON_BIN}" train_skg.py \
      --output-dir "${OUTPUT_DIR}" \
      --stage baselines \
      --experiment-name "${experiment}" \
      --seed "${seed}" \
      --model-family "${family}" \
      --encoder "${encoder}" \
      --fusion "${fusion}" \
      --knowledge-mode none \
      --pretrain "${PRETRAIN}" \
      --epochs "${EPOCHS}" \
      --early-stopping-patience "${PATIENCE}" \
      --input-size "${INPUT_SIZE}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --device "${DEVICE}" \
      --mixed-precision "${MIXED_PRECISION}" \
      --deterministic true \
      --normalization none \
      --max-train-samples "${MAX_TRAIN_SAMPLES}" \
      --max-val-samples "${MAX_VAL_SAMPLES}"
  done
}

# B1: early-fusion SegFormer; B2: single-stream ConvNeXt + SegFormer decoder.
run_baseline "B1_single_segformer" "single" "mit_b2" "none"
run_baseline "B2_single_convnext_tiny" "single" "tu-convnext_tiny" "none"

# B3/B4/B6 share the same dual-stream encoder, training budget, and augmentation.
run_baseline "B3_dual_add" "dual" "tu-convnext_tiny" "add"
run_baseline "B4_dual_concat" "dual" "tu-convnext_tiny" "cat"
run_baseline "B6_dual_ordinary_attention" "dual" "tu-convnext_tiny" "att"

# B5 is an initial parameter-matched single-stream control. Its actual parameter
# count is logged by train_skg.py and must be reported alongside B4 before use.
run_baseline "B5_single_convnext_small_param_control" "single" "tu-convnext_small" "none"
