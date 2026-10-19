#!/usr/bin/env sh
set -eu

# Construct class spectral prototypes from train only. This script refuses any
# split other than train in the Python implementation.
#
# Example:
#   DATA_ROOT=/path/to/mmlsv2 SEEDS="42 123 7" K_VALUES="1 2 4 8" sh tools/knowledge/build_prototypes.sh

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -n "${DATA_ROOT:-}" ]; then
  export MMLSV2_DATA_ROOT="${DATA_ROOT}"
fi

SEEDS="${SEEDS:-42}"
K_VALUES="${K_VALUES:-4}"
DISTANCE="${DISTANCE:-cosine}"
NORMALIZATION="${NORMALIZATION:-none}"
DESCRIPTOR_CONFIG="${DESCRIPTOR_CONFIG:-configs/descriptors/default.json}"
DESCRIPTOR_NAME="${DESCRIPTOR_NAME:-default}"
STRICT_DETERMINISM="${STRICT_DETERMINISM:-1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/prototypes}"
MAX_PIXELS_PER_IMAGE="${MAX_PIXELS_PER_IMAGE:-2048}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

for seed in ${SEEDS}; do
  for k in ${K_VALUES}; do
    output="${ARTIFACT_ROOT}/${DESCRIPTOR_NAME}/k${k}/seed${seed}/prototypes.npz"
    echo "=== train-only prototypes: descriptor=${DESCRIPTOR_NAME}, K=${k}, seed=${seed} ==="
    PYTHONHASHSEED="${seed}" \
    CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    TRAIN_STRICT_DETERMINISM="${STRICT_DETERMINISM}" \
    "${PYTHON_BIN}" -m knowledge.build_prototypes \
      --descriptor-config "${DESCRIPTOR_CONFIG}" \
      --prototype-k "${k}" \
      --distance "${DISTANCE}" \
      --normalization "${NORMALIZATION}" \
      --max-pixels-per-image "${MAX_PIXELS_PER_IMAGE}" \
      --seed "${seed}" \
      --max-samples "${MAX_SAMPLES}" \
      --output "${output}"
  done
done
