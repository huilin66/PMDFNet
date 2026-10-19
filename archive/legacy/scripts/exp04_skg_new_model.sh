#!/usr/bin/env bash
set -euo pipefail

# Purpose: new-model experiments. This group is intentionally separate from
# the legacy competition reproduction. By default it uses the native public
# MMLSv2 [0,1] release because the existing SKG scripts use its normalization
# contract. Run this group only after the legacy baseline is frozen.

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${SKG_DATA_ROOT:-${DATA_ROOT:-/scrinvme/huilin/bdd/cp_data/mmlsv2}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_skg}"
STAGES="${STAGES:-baselines prototypes ablation}"
SEEDS="${SEEDS:-42}"

echo "[skg_new_model] data=${DATA_ROOT}"
echo "[skg_new_model] stages=${STAGES} seeds=${SEEDS}"
echo "[skg_new_model] This group must follow the frozen legacy baseline."

DATA_ROOT="${DATA_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
STAGES="${STAGES}" \
SEEDS="${SEEDS}" \
PYTHON_BIN="${PYTHON_BIN}" \
  bash "${REPO_ROOT}/scripts/00_run_skg_training_pipeline.sh"
