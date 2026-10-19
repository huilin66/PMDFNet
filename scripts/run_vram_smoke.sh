#!/usr/bin/env sh
set -eu

# Bounded synthetic VRAM probe. It builds one model per selected protocol entry,
# runs one training step and reports peak allocated/reserved memory. It does not
# read the dataset, train, or write checkpoints.
#
# Use it before a long run to confirm new baselines fit, especially the extended
# group (M11/M3LSNet has a bottleneck self-attention and no pretraining).
#
# Overrides:
#   GROUP=extended           core | ablations | extended | all (default all)
#   BATCH_SIZE=32            synthetic probe batch (script default is 8)
#   DEVICE=cuda:0            torch device
#   SEED=41                  probe seed
#   MAX_MEMORY_GB=40         hard safety ceiling
#   ALLOCATOR_LIMIT_GB=38    per-process allocator limit
#   TIMEOUT_SECONDS=300      per-configuration timeout
#
# Examples:
#   GROUP=extended BATCH_SIZE=32 bash scripts/run_vram_smoke.sh
#   GROUP=core     BATCH_SIZE=32 bash scripts/run_vram_smoke.sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

GROUP="${GROUP:-all}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-41}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-40}"
ALLOCATOR_LIMIT_GB="${ALLOCATOR_LIMIT_GB:-38}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_experiments/smoke/vram}"

export GROUP BATCH_SIZE SEED MAX_MEMORY_GB ALLOCATOR_LIMIT_GB TIMEOUT_SECONDS OUTPUT_DIR

echo "######## vram smoke: group=${GROUP} batch=${BATCH_SIZE} device=${DEVICE} ########"
sh "${REPO_ROOT}/tools/validation/smoke_test_vram.sh"
