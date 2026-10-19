#!/usr/bin/env sh
set -eu

# Data-only smoke check: CLI parsing, data-root resolution, dataset import and
# one sample read. No model is initialised and no training happens.
#
# If DATA_ROOT is empty the probe still runs its syntax/CLI checks and reports
# which environment key to set. Resolution order matches the trainer:
#   DATA_ROOT -> MARS_DATA_ROOT -> MAPPED_DATA_ROOT -> MMLSV2_DATA_ROOT -> .env
#
# Overrides:
#   DATA_ROOT=...            mapped MMLSv2 root
#   TRAIN_SPLIT=train        split used for the read probe
#   VAL_SPLIT=test           second split used for the read probe
#   INPUT_SIZE=64            probe resolution
#   MAX_TRAIN_SAMPLES=2      samples read from the train split
#   MAX_VAL_SAMPLES=2        samples read from the validation split

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

echo "######## data smoke: data_root=${DATA_ROOT:-<unset>} ########"
if [ -z "${DATA_ROOT}" ]; then
  echo "[data-smoke] DATA_ROOT is unset; set DATA_ROOT or MMLSV2_DATA_ROOT to include the read probe." >&2
fi

sh "${REPO_ROOT}/tools/validation/smoke_test_data.sh"
