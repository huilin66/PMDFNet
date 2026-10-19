#!/usr/bin/env sh
set -eu

# Supplementary modality-grouping protocol.
# G0 and G1 are frozen references to L1/M5 and B2 in the main protocol; this
# launcher trains only the two new equal-capacity controls G2 and G3 for the
# configured seeds (default: 41, 42, 43), six runs in total.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

MODALITY_GROUPING_CONFIG="${MODALITY_GROUPING_CONFIG:-${CONFIG}}"
if [ -z "${MODALITY_GROUPING_CONFIG}" ] || [ "${MODALITY_GROUPING_CONFIG}" = "configs/mmlsv2_paper_protocol.json" ]; then
  MODALITY_GROUPING_CONFIG="configs/mmlsv2_modality_grouping_protocol.json"
fi
CONFIG="${MODALITY_GROUPING_CONFIG}"
GROUP="modality_grouping"
export CONFIG GROUP

echo "######## modality grouping: G2,G3; seeds=${SEEDS} ########"
sh "${SCRIPT_DIR}/train_paper_experiments.sh"
