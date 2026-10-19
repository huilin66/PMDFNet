#!/usr/bin/env bash
set -euo pipefail

# Purpose: compare model capacity after the historical fusion configuration is
# reproduced. The channel split, fusion, optimizer, and data protocol remain
# fixed at chv1 + add.

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export MODEL_NAMES="${MODEL_NAMES:-dual_segformer_convnexttiny_chv1_add dual_segformer_convnextsmall_chv1_add dual_segformer_convnextbase_chv1_add dual_segformer_convnextlarge_chv1_add}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_experiments/legacy_capacity}"
export GROUP_NAME="legacy_capacity"
export SEEDS="${SEEDS:-42}"

bash "${SCRIPT_DIR}/exp01_reproduce_competition.sh"
