#!/usr/bin/env sh
set -eu

# Frozen paper protocol entry point. One Python process is started per seed so
# PYTHONHASHSEED/CUDA determinism are set before Python and PyTorch initialize.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG=${CONFIG:-configs/mmlsv2_paper_protocol.json}
# Same resolution order as the trainer and tools/evaluation (and .env), minus the
# old baked-in host path that disagreed with the evaluation scripts.
DATA_ROOT=${DATA_ROOT:-${MARS_DATA_ROOT:-${MAPPED_DATA_ROOT:-${MMLSV2_DATA_ROOT:-}}}}
if [ -z "${DATA_ROOT}" ]; then
  echo "DATA_ROOT is not set. Export DATA_ROOT (or MARS_DATA_ROOT/MAPPED_DATA_ROOT/MMLSV2_DATA_ROOT) to the mapped MMLSv2 root." >&2
  exit 1
fi
DEVICE=${DEVICE:-cuda:0}
GROUP=${GROUP:-core}
SEEDS=${SEEDS:-"41 42 43"}
NUM_WORKERS=${NUM_WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-}
EXPERIMENTS=${EXPERIMENTS:-}
# Never relocate completed runs by default. Set FINALIZE_RESULTS=1 only for an
# explicit legacy copy to the dataset output filesystem.
FINALIZE_RESULTS=${FINALIZE_RESULTS:-0}
STRICT_DETERMINISM=${STRICT_DETERMINISM:-1}
DRY_RUN=${DRY_RUN:-0}
SKIP_MODEL_INIT=${SKIP_MODEL_INIT:-0}

case "${STRICT_DETERMINISM}" in
  1|true|TRUE|yes|YES|on|ON) STRICT_VALUE=1 ;;
  *) STRICT_VALUE=0 ;;
esac

if [ ! -d "${DATA_ROOT}/train/images" ] || [ ! -d "${DATA_ROOT}/val/masks" ]; then
  echo "Dataset is incomplete: ${DATA_ROOT}" >&2
  echo "Set DATA_ROOT/MARS_DATA_ROOT/MAPPED_DATA_ROOT to the mapped MMLSv2 root." >&2
  exit 1
fi

export DATA_ROOT
export MARS_DATA_ROOT="${DATA_ROOT}"
export MMLSV2_DATA_ROOT="${DATA_ROOT}"
export TRAIN_STRICT_DETERMINISM="${STRICT_VALUE}"
export PYTHONUNBUFFERED=1

echo "[paper] config=${CONFIG} group=${GROUP}"
echo "[paper] data=${DATA_ROOT} device=${DEVICE} seeds=${SEEDS} batch=${BATCH_SIZE:-protocol-default} experiments=${EXPERIMENTS:-group-default} finalize=${FINALIZE_RESULTS:-config-default}"

for seed in ${SEEDS}; do
  echo "=== paper group=${GROUP}, seed=${seed} ==="
  set -- \
    --config "${CONFIG}" \
    --group "${GROUP}" \
    --seed "${seed}" \
    --data-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --num-workers "${NUM_WORKERS}"
  if [ -n "${BATCH_SIZE}" ]; then
    set -- "$@" --batch-size "${BATCH_SIZE}"
  fi
  if [ -n "${EXPERIMENTS}" ]; then
    set -- "$@" --experiments "${EXPERIMENTS}"
  fi
  if [ -n "${FINALIZE_RESULTS}" ]; then
    set -- "$@" --finalize-results "${FINALIZE_RESULTS}"
  fi
  if [ "${DRY_RUN}" = "1" ]; then
    set -- "$@" --dry-run
  fi
  if [ "${SKIP_MODEL_INIT}" = "1" ]; then
    set -- "$@" --skip-model-init
  fi

  PYTHONHASHSEED="${seed}" \
  CUBLAS_WORKSPACE_CONFIG=":4096:8" \
  "${PYTHON_BIN}" scripts/train_paper_experiments.py "$@"
done

echo "[paper] completed group=${GROUP}"
