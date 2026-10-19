#!/usr/bin/env sh
set -eu

# Group: extended -- E1 extra comparison baselines M6-M11.
#   M6  UPerNet       + ResNet50     (tu-resnet50)
#   M7  FPN           + ResNet50     (tu-resnet50)
#   M8  MANet         + ResNet50     (tu-resnet50)
#   M9  U-Net++       + ResNet50     (tu-resnet50)
#   M10 OCRNet/HRNet-OCR + HRNet-W18 (tu-hrnet_w18, auxiliary head, 0.4 weight)
#   M11 M3LSNet                      (networks.get_model, no pretraining)
#
# 'extended' is a virtual group: these experiments are stored with group 'core'
# because they belong to the main E1 table, but they can be launched separately.
#
# Overrides (export before calling):
#   DATA_ROOT=...             mapped MMLSv2 root; falls back to .env
#   SEEDS="41 42 43"          reproducibility seeds
#   EXPERIMENTS=M6,M7         optional comma-separated subset of this group
#   BATCH_SIZE=16             lower the batch if a baseline does not fit
#
# M11 has no ImageNet pretraining and depends on mamba_ssm being importable
# (otherwise the built-in gated-conv simulator is used). Record which path ran.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

GROUP="${GROUP:-extended}"
export GROUP

echo "######## group=${GROUP} experiments=${EXPERIMENTS:-all} seeds=${SEEDS} ########"
sh "${SCRIPT_DIR}/train_paper_experiments.sh"
