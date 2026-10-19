#!/usr/bin/env sh
set -eu

# Stage 3 core SKG ablation A0-A6. Run tools/knowledge/build_prototypes.sh first.
# Every run uses train→val checkpoint selection and leaves test untouched.
#
# Example:
#   DATA_ROOT=/path/to/mmlsv2 SEEDS="42" sh scripts/04_train_knowledge_ablation.sh

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -n "${DATA_ROOT:-}" ]; then
  export MMLSV2_DATA_ROOT="${DATA_ROOT}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-outputs_skg/ablation}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/prototypes}"
DESCRIPTOR_CONFIG="${DESCRIPTOR_CONFIG:-configs/descriptors/default.json}"
DESCRIPTOR_NAME="${DESCRIPTOR_NAME:-default}"
PROTOTYPE_K="${PROTOTYPE_K:-4}"
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

run_ablation() {
  experiment="$1"
  mode="$2"
  fusion="$3"
  update="$4"
  lambda_kc="$5"
  lambda_proto="$6"
  require_prototype="$7"
  for seed in ${SEEDS}; do
    prototype_path=""
    if [ "${require_prototype}" = "1" ]; then
      prototype_path="${ARTIFACT_ROOT}/${DESCRIPTOR_NAME}/k${PROTOTYPE_K}/seed${seed}/prototypes.npz"
      if [ ! -f "${prototype_path}" ]; then
        echo "Missing ${prototype_path}. Run tools/knowledge/build_prototypes.sh with matching SEEDS/K first." >&2
        exit 1
      fi
    fi
    echo "=== ${experiment}, seed ${seed} ==="
    PYTHONHASHSEED="${seed}" \
    CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    TRAIN_STRICT_DETERMINISM="${STRICT_DETERMINISM}" \
    "${PYTHON_BIN}" train_skg.py \
      --output-dir "${OUTPUT_DIR}" \
      --stage ablation \
      --experiment-name "${experiment}" \
      --seed "${seed}" \
      --model-family dual \
      --encoder "tu-convnext_tiny" \
      --fusion "${fusion}" \
      --knowledge-mode "${mode}" \
      --knowledge-stages "3,4" \
      --descriptor-config "${DESCRIPTOR_CONFIG}" \
      --prototype-k "${PROTOTYPE_K}" \
      --prototype-distance cosine \
      --prototype-update "${update}" \
      --knowledge-consistency-weight "${lambda_kc}" \
      --prototype-reg-weight "${lambda_proto}" \
      --knowledge-confidence maxprob \
      --knowledge-confidence-threshold 0.7 \
      --pretrain "${PRETRAIN}" \
      --epochs "${EPOCHS}" \
      --early-stopping-patience "${PATIENCE}" \
      --input-size "${INPUT_SIZE}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --device "${DEVICE}" \
      --mixed-precision "${MIXED_PRECISION}" \
      --deterministic true \
      --normalization auto \
      --max-train-samples "${MAX_TRAIN_SAMPLES}" \
      --max-val-samples "${MAX_VAL_SAMPLES}" \
      --prototype-path "${prototype_path}"
  done
}

run_ablation "A0_dual_concat" "none"       "cat"            "fixed"    "0.0" "0.0"   "0"
run_ablation "A1_descriptor"  "descriptor" "cat"            "fixed"    "0.0" "0.0"   "0"
run_ablation "A2_prior"       "prototype"  "cat"            "fixed"    "0.0" "0.0"   "1"
run_ablation "A3_kg_fusion"   "full"       "knowledge_gate" "fixed"    "0.0" "0.0"   "1"
run_ablation "A4_kc_loss"     "full"       "knowledge_gate" "fixed"    "0.1" "0.0"   "1"
run_ablation "A5_residual"    "full"       "knowledge_gate" "residual" "0.1" "0.0"   "1"
run_ablation "A6_full_skg"    "full"       "knowledge_gate" "residual" "0.1" "0.001" "1"
