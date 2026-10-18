#!/usr/bin/env sh
set -eu

# Frozen-checkpoint interpretability workflow:
#   1) Grad-CAM for each requested formal run;
#   2) a model-comparison CAM sheet;
#   3) joint GT/Pred prototype t-SNE.
#
# This script never trains, changes a run directory or selects a checkpoint by
# Test performance.  The default fixed sample seed is part of the paper plan.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

if [ -z "${DATA_ROOT:-}" ]; then
  echo "DATA_ROOT is empty; set DATA_ROOT or MMLSV2_DATA_ROOT" >&2
  exit 2
fi

INTERP_IDS="${INTERP_IDS:-M4,L1}"
INTERP_SEED="${INTERP_SEED:-41}"
INTERP_SPLIT="${INTERP_SPLIT:-test}"
INTERP_SELECTION_SEED="${INTERP_SELECTION_SEED:-20261018}"
INTERP_TOP_K="${INTERP_TOP_K:-12}"
INTERP_CLASS_ID="${INTERP_CLASS_ID:-1}"
INTERP_CAM_TARGET="${INTERP_CAM_TARGET:-gt_mask}"
INTERP_CAM_LAYER="${INTERP_CAM_LAYER:-auto}"
INTERP_OUTPUT_ROOT="${INTERP_OUTPUT_ROOT:-outputs_experiments/interpretability}"
INTERP_TSNE_MAX_IMAGES="${INTERP_TSNE_MAX_IMAGES:-80}"

CAM_DIRS=""
set --
OLD_IFS=${IFS}
IFS=,
for MODEL_ID in ${INTERP_IDS}; do
  [ -n "${MODEL_ID}" ] || continue
  MODEL_CAM_DIR="${INTERP_OUTPUT_ROOT}/cam/${MODEL_ID}_s${INTERP_SEED}"
  "${PYTHON_BIN}" tools/evaluation/cam_analysis.py \
    --data-root "${DATA_ROOT}" \
    --run-root "${RUN_ROOT:-outputs_experiments/pmdfnet_mmlsv2}" \
    --config "${CONFIG}" \
    --experiment-id "${MODEL_ID}" \
    --seed "${INTERP_SEED}" \
    --split "${INTERP_SPLIT}" \
    --device "${DEVICE}" \
    --cam-layer "${INTERP_CAM_LAYER}" \
    --cam-target "${INTERP_CAM_TARGET}" \
    --class-id "${INTERP_CLASS_ID}" \
    --selection fixed \
    --selection-seed "${INTERP_SELECTION_SEED}" \
    --top-k "${INTERP_TOP_K}" \
    --output-dir "${MODEL_CAM_DIR}"
  set -- "$@" --model-dir "${MODEL_ID}=${MODEL_CAM_DIR}"
done
IFS=${OLD_IFS}

"${PYTHON_BIN}" tools/evaluation/make_cam_comparison_figure.py \
  "$@" \
  --max-samples "${INTERP_TOP_K}" \
  --output "${INTERP_OUTPUT_ROOT}/cam/cam_comparison_s${INTERP_SEED}.png"

"${PYTHON_BIN}" tools/evaluation/tsne_prototype_analysis.py \
  --data-root "${DATA_ROOT}" \
  --model-ids "${INTERP_IDS}" \
  --run-root "${RUN_ROOT:-outputs_experiments/pmdfnet_mmlsv2}" \
  --config "${CONFIG}" \
  --seed "${INTERP_SEED}" \
  --split "${INTERP_SPLIT}" \
  --device "${DEVICE}" \
  --selection-seed "${INTERP_SELECTION_SEED}" \
  --max-images-per-class "${INTERP_TSNE_MAX_IMAGES}" \
  --output-dir "${INTERP_OUTPUT_ROOT}/tsne/${INTERP_IDS}_s${INTERP_SEED}"

