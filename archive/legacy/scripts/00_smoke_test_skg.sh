#!/usr/bin/env sh
set -eu

# End-to-end smoke test for every currently implemented SKG task. Each model
# receives one optimizer step (2 train samples / batch size 2) and one val pass.
# It creates isolated timestamped outputs and never opens the test split.
#
# Example:
#   DATA_ROOT=/scrinvme/huilin/bdd/cp_data/mmlsv2 sh scripts/00_smoke_test_skg.sh

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -n "${DATA_ROOT:-}" ]; then
  export MMLSV2_DATA_ROOT="${DATA_ROOT}"
fi

SMOKE_TAG="${SMOKE_TAG:-$(date '+%Y%m%d_%H%M%S')}"
SMOKE_ROOT="${SMOKE_ROOT:-outputs_skg/smoke/${SMOKE_TAG}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/prototypes/smoke/${SMOKE_TAG}}"
SEED="${SEED:-42}"
INPUT_SIZE="${INPUT_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DEVICE="${DEVICE:-auto}"

echo "Smoke output: ${SMOKE_ROOT}"
echo "Protocol: train (2 images / 1 step) -> val (2 images); test is never loaded."

OUTPUT_DIR="${SMOKE_ROOT}/baselines" \
SEEDS="${SEED}" EPOCHS=1 PATIENCE=0 PRETRAIN=false MIXED_PRECISION=false \
INPUT_SIZE="${INPUT_SIZE}" BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS=0 DEVICE="${DEVICE}" \
MAX_TRAIN_SAMPLES=2 MAX_VAL_SAMPLES=2 \
sh scripts/01_train_baselines.sh

# Prototype construction is not an epoch-based operation. Eight train images
# make class sampling reliable while still keeping this check lightweight.
ARTIFACT_ROOT="${ARTIFACT_ROOT}" DESCRIPTOR_NAME=smoke K_VALUES=2 SEEDS="${SEED}" \
MAX_SAMPLES=8 MAX_PIXELS_PER_IMAGE=128 NORMALIZATION=none \
sh tools/knowledge/build_prototypes.sh

OUTPUT_DIR="${SMOKE_ROOT}/ablation" ARTIFACT_ROOT="${ARTIFACT_ROOT}" DESCRIPTOR_NAME=smoke \
PROTOTYPE_K=2 SEEDS="${SEED}" EPOCHS=1 PATIENCE=0 PRETRAIN=false MIXED_PRECISION=false \
INPUT_SIZE="${INPUT_SIZE}" BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS=0 DEVICE="${DEVICE}" \
MAX_TRAIN_SAMPLES=2 MAX_VAL_SAMPLES=2 \
sh scripts/04_train_knowledge_ablation.sh

# Exercise the frozen-checkpoint evaluation entry point against val only.
CHECKPOINT="${SMOKE_ROOT}/baselines/B4_dual_concat/${SEED}/checkpoints/best.pth"
"${PYTHON_BIN}" tools/evaluation/eval_skg.py \
  --checkpoint "${CHECKPOINT}" \
  --split val \
  --device "${DEVICE}" \
  --num-workers 0 \
  --output-json "${SMOKE_ROOT}/eval_b4_on_val.json"

echo ""
echo "SKG smoke test passed: ${SMOKE_ROOT}"
