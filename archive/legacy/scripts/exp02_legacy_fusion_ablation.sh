#!/usr/bin/env bash
set -euo pipefail

# Purpose: controlled legacy fusion ablation after the competition baseline is
# available. Only the fusion/model name changes; data and training budget stay
# fixed.

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export MODEL_NAMES="${MODEL_NAMES:-dual_segformer_convnexttiny_chv1_add dual_segformer_convnexttiny_chv1_cat dual_segformer_convnexttiny_chv1_att dual_segformer_convnexttiny_chv1_moe dual_segformer_convnexttiny_chv1_moev2}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_experiments/legacy_fusion}"
export GROUP_NAME="legacy_fusion_ablation"
export SEEDS="${SEEDS:-42}"

bash "${SCRIPT_DIR}/exp01_reproduce_competition.sh"
