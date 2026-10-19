#!/usr/bin/env sh
set -eu

# Frozen-checkpoint interpretability workflow:
#   1) Grad-CAM for each requested formal run, with a mostly-foreground and
#      small-background per-sample target plan;
#   2) model-comparison CAM and prediction sheets;
#   3) joint prototype t-SNE plus one foreground/background figure per model;
#      by default, t-SNE uses the complete labeled Test split.
#
# This script never trains or changes a run directory.  The CAM sample list is
# performance-conditioned; t-SNE defaults to all Test images and can be changed
# to the CAM list with INTERP_TSNE_ALL_TEST=false.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

if [ -z "${DATA_ROOT:-}" ]; then
  echo "DATA_ROOT is empty; set DATA_ROOT or MMLSV2_DATA_ROOT" >&2
  exit 2
fi

# INTERP_IDS remains a backwards-compatible shorthand for the t-SNE model set.
# The default CAM/prediction comparison uses the requested M1/M2/M3/M10/M11
# set; use the explicit lists when the CAM and t-SNE subsets differ.
INTERP_IDS="${INTERP_IDS:-M1,M2,M3,M4,L1}"
INTERP_CAM_IDS="${INTERP_CAM_IDS:-M1,M2,M3,M10,M11}"
INTERP_TSNE_IDS="${INTERP_TSNE_IDS:-${INTERP_IDS}}"
INTERP_SELECTION_TARGET_IDS="${INTERP_SELECTION_TARGET_IDS:-L1}"
INTERP_SELECTION_REFERENCE_IDS="${INTERP_SELECTION_REFERENCE_IDS:-M1,M2,M3,M10,M11}"
INTERP_SEED="${INTERP_SEED:-41}"
INTERP_SPLIT="${INTERP_SPLIT:-test}"
INTERP_SELECTION_SEED="${INTERP_SELECTION_SEED:-20261018}"
INTERP_TOP_K="${INTERP_TOP_K:-12}"
INTERP_TARGET_MODE="${INTERP_TARGET_MODE:-mixed}"
INTERP_FOREGROUND_CLASS_ID="${INTERP_FOREGROUND_CLASS_ID:-${INTERP_CLASS_ID:-1}}"
INTERP_BACKGROUND_CLASS_ID="${INTERP_BACKGROUND_CLASS_ID:-0}"
INTERP_BACKGROUND_RATIO="${INTERP_BACKGROUND_RATIO:-0.25}"
INTERP_TARGET_PLAN="${INTERP_TARGET_PLAN:-}"
INTERP_CAM_TARGET="${INTERP_CAM_TARGET:-gt_mask}"
INTERP_CAM_LAYER="${INTERP_CAM_LAYER:-auto}"
INTERP_OUTPUT_ROOT="${INTERP_OUTPUT_ROOT:-outputs_experiments/interpretability}"
INTERP_TSNE_MAX_IMAGES="${INTERP_TSNE_MAX_IMAGES:-80}"
INTERP_TSNE_SOURCES="${INTERP_TSNE_SOURCES:-gt}"
INTERP_TSNE_ALL_TEST="${INTERP_TSNE_ALL_TEST:-true}"
INTERP_SELECTION_METRIC="${INTERP_SELECTION_METRIC:-miou}"
INTERP_SELECTION_REFERENCE_REDUCE="${INTERP_SELECTION_REFERENCE_REDUCE:-max}"
INTERP_SELECTION_TOP_K="${INTERP_SELECTION_TOP_K:-${INTERP_TOP_K}}"
INTERP_SELECTION_MIN_ADVANTAGE="${INTERP_SELECTION_MIN_ADVANTAGE:-1e-6}"
INTERP_SELECTION_REFERENCE_TAG=$(printf '%s' "${INTERP_SELECTION_REFERENCE_IDS}" | tr ',' '_')
INTERP_SELECTION_OUTPUT_DIR="${INTERP_SELECTION_OUTPUT_DIR:-${INTERP_OUTPUT_ROOT}/selection/pmdfnet_advantage_${INTERP_SELECTION_REFERENCE_TAG}_s${INTERP_SEED}}"
INTERP_PAPER_TOP_K="${INTERP_PAPER_TOP_K:-6}"

"${PYTHON_BIN}" tools/evaluation/select_interpretability_samples.py \
  --data-root "${DATA_ROOT}" \
  --target-model-ids "${INTERP_SELECTION_TARGET_IDS}" \
  --reference-model-ids "${INTERP_SELECTION_REFERENCE_IDS}" \
  --run-root "${RUN_ROOT:-outputs_experiments/pmdfnet_mmlsv2}" \
  --config "${CONFIG}" \
  --seed "${INTERP_SEED}" \
  --split "${INTERP_SPLIT}" \
  --device "${DEVICE}" \
  --metric "${INTERP_SELECTION_METRIC}" \
  --reference-reduce "${INTERP_SELECTION_REFERENCE_REDUCE}" \
  --top-k "${INTERP_SELECTION_TOP_K}" \
  --min-advantage "${INTERP_SELECTION_MIN_ADVANTAGE}" \
  --selection-seed "${INTERP_SELECTION_SEED}" \
  --output-dir "${INTERP_SELECTION_OUTPUT_DIR}"

INTERP_SAMPLE_IDS=$(tr '\n' ',' < "${INTERP_SELECTION_OUTPUT_DIR}/selected_ids.txt" | sed 's/,$//')
[ -n "${INTERP_SAMPLE_IDS}" ] || {
  echo "The PMDFNet advantage selector returned no image IDs" >&2
  exit 3
}

CAM_DIRS=""
set --
OLD_IFS=${IFS}
IFS=,
for MODEL_ID in ${INTERP_CAM_IDS}; do
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
    --target-mode "${INTERP_TARGET_MODE}" \
    --foreground-class-id "${INTERP_FOREGROUND_CLASS_ID}" \
    --background-class-id "${INTERP_BACKGROUND_CLASS_ID}" \
    --background-ratio "${INTERP_BACKGROUND_RATIO}" \
    --selection fixed \
    --selection-seed "${INTERP_SELECTION_SEED}" \
    --top-k "${INTERP_TOP_K}" \
    --ids "${INTERP_SAMPLE_IDS}" \
    ${INTERP_TARGET_PLAN:+--target-plan "${INTERP_TARGET_PLAN}"} \
    --output-dir "${MODEL_CAM_DIR}"
  set -- "$@" --model-dir "${MODEL_ID}=${MODEL_CAM_DIR}"
done
IFS=${OLD_IFS}

"${PYTHON_BIN}" tools/evaluation/make_cam_comparison_figure.py \
  "$@" \
  --max-samples "${INTERP_TOP_K}" \
  --output "${INTERP_OUTPUT_ROOT}/cam/cam_comparison_rgb_slope_s${INTERP_SEED}.png"

"${PYTHON_BIN}" tools/evaluation/make_prediction_comparison_figure.py \
  "$@" \
  --max-samples "${INTERP_TOP_K}" \
  --output "${INTERP_OUTPUT_ROOT}/cam/prediction_comparison_rgb_slope_s${INTERP_SEED}.png"

# Keep the complete top-K sheets for audit, while producing a less dense
# manuscript version from the highest-ranked frozen advantage cases.
if [ "${INTERP_PAPER_TOP_K}" -gt 0 ]; then
  INTERP_PAPER_SAMPLE_IDS=$(head -n "${INTERP_PAPER_TOP_K}" "${INTERP_SELECTION_OUTPUT_DIR}/selected_ids.txt" | tr '\n' ',' | sed 's/,$//')
  [ -n "${INTERP_PAPER_SAMPLE_IDS}" ] || {
    echo "The requested paper sample subset is empty" >&2
    exit 4
  }
  "${PYTHON_BIN}" tools/evaluation/make_cam_comparison_figure.py \
    "$@" \
    --sample-ids "${INTERP_PAPER_SAMPLE_IDS}" \
    --output "${INTERP_OUTPUT_ROOT}/cam/cam_comparison_rgb_slope_s${INTERP_SEED}_n${INTERP_PAPER_TOP_K}.png"
  "${PYTHON_BIN}" tools/evaluation/make_prediction_comparison_figure.py \
    "$@" \
    --sample-ids "${INTERP_PAPER_SAMPLE_IDS}" \
    --output "${INTERP_OUTPUT_ROOT}/cam/prediction_comparison_rgb_slope_s${INTERP_SEED}_n${INTERP_PAPER_TOP_K}.png"
fi

if [ "${INTERP_TSNE_ALL_TEST}" = "true" ]; then
  TSNE_OUTPUT_DIR="${INTERP_OUTPUT_ROOT}/tsne/${INTERP_TSNE_IDS}_all_test_s${INTERP_SEED}"
  "${PYTHON_BIN}" tools/evaluation/tsne_prototype_analysis.py \
    --data-root "${DATA_ROOT}" \
    --model-ids "${INTERP_TSNE_IDS}" \
    --run-root "${RUN_ROOT:-outputs_experiments/pmdfnet_mmlsv2}" \
    --config "${CONFIG}" \
    --seed "${INTERP_SEED}" \
    --split "${INTERP_SPLIT}" \
    --device "${DEVICE}" \
    --sources "${INTERP_TSNE_SOURCES}" \
    --selection-seed "${INTERP_SELECTION_SEED}" \
    --max-images-per-class 0 \
    --output-dir "${TSNE_OUTPUT_DIR}"
else
  TSNE_OUTPUT_DIR="${INTERP_OUTPUT_ROOT}/tsne/${INTERP_TSNE_IDS}_s${INTERP_SEED}"
  "${PYTHON_BIN}" tools/evaluation/tsne_prototype_analysis.py \
    --data-root "${DATA_ROOT}" \
    --model-ids "${INTERP_TSNE_IDS}" \
    --run-root "${RUN_ROOT:-outputs_experiments/pmdfnet_mmlsv2}" \
    --config "${CONFIG}" \
    --seed "${INTERP_SEED}" \
    --split "${INTERP_SPLIT}" \
    --device "${DEVICE}" \
    --sources "${INTERP_TSNE_SOURCES}" \
    --ids "${INTERP_SAMPLE_IDS}" \
    --selection-seed "${INTERP_SELECTION_SEED}" \
    --max-images-per-class "${INTERP_TSNE_MAX_IMAGES}" \
    --output-dir "${TSNE_OUTPUT_DIR}"
fi

