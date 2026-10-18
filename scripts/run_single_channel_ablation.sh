#!/usr/bin/env sh
set -eu

# Supplementary single-channel removal ablation.  This is deliberately a
# separate protocol from the 57-run formal matrix.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

if [ -z "${DATA_ROOT:-}" ]; then
  echo "DATA_ROOT is empty; set DATA_ROOT or MMLSV2_DATA_ROOT" >&2
  exit 2
fi

SINGLE_CHANNEL_CONFIG="${SINGLE_CHANNEL_CONFIG:-configs/mmlsv2_single_channel_protocol.json}"
SINGLE_CHANNEL_SEEDS="${SINGLE_CHANNEL_SEEDS:-${SEEDS}}"
SINGLE_CHANNEL_EXPERIMENTS="${SINGLE_CHANNEL_EXPERIMENTS:-${EXPERIMENTS:-}}"

echo "######## supplementary single-channel ablation ########"
echo "config=${SINGLE_CHANNEL_CONFIG} data=${DATA_ROOT} seeds=${SINGLE_CHANNEL_SEEDS}"
echo "experiments=${SINGLE_CHANNEL_EXPERIMENTS:-S1,S2,S3,S4}"

CONFIG="${SINGLE_CHANNEL_CONFIG}" \
GROUP="single_channel" \
SEEDS="${SINGLE_CHANNEL_SEEDS}" \
EXPERIMENTS="${SINGLE_CHANNEL_EXPERIMENTS}" \
bash "${SCRIPT_DIR}/train_paper_experiments.sh"
