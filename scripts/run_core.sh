#!/usr/bin/env sh
set -eu

# Group: core -- representative baselines M1-M4 (CE-only protocol).
#
# Overrides (export before calling):
#   DATA_ROOT=...        mapped MMLSv2 root; falls back to MMLSV2_DATA_ROOT/.env
#   SEEDS="41 42 43"     reproducibility seeds
#   DEVICE=cuda:0        torch device
#   BATCH_SIZE=32        default is the protocol value
#   EXPERIMENTS=M4       optional comma-separated subset of this group
#
# Selective rerun example (the single most important baseline):
#   DATA_ROOT=/data/.../mmlsv2_mapped_mars_ls EXPERIMENTS=M4 bash scripts/run_core.sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

GROUP="${GROUP:-core}"
export GROUP

echo "######## group=${GROUP} experiments=${EXPERIMENTS:-all} seeds=${SEEDS} ########"
sh "${SCRIPT_DIR}/train_paper_experiments.sh"
