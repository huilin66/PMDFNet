#!/usr/bin/env sh
set -eu

# Group: ablations -- A1 A2 A5 B2 B3 C2 plus the loss/fusion family L1 L2 L3.
#
# Overrides (export before calling):
#   DATA_ROOT=...        mapped MMLSv2 root; falls back to MMLSV2_DATA_ROOT/.env
#   SEEDS="41 42 43"     reproducibility seeds
#   DEVICE=cuda:0        torch device
#   EXPERIMENTS=L1,L2,L3 optional comma-separated subset of this group
#
# Note: L1 is also the paper's main model (M5), A4, B1 and C1. It is trained
# exactly once, here.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

GROUP="${GROUP:-ablations}"
export GROUP

echo "######## group=${GROUP} experiments=${EXPERIMENTS:-all} seeds=${SEEDS} ########"
sh "${SCRIPT_DIR}/train_paper_experiments.sh"
