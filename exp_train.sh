#!/usr/bin/env bash
set -euo pipefail

# Current PMDFNet paper-workflow entry point. Historical competition, SKG,
# submission, and analysis entry points are preserved under archive/legacy/.

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage: bash exp_train.sh [paper_core|paper_ablations|paper_extended|paper_all|vram_smoke|data_smoke|test]

Groups:
  paper_core       Run M1-M4 with seeds 41/42/43 (CE-only).
  paper_ablations  Run A1/A2/A5/B2/B3/C2/L1/L2/L3 with seeds 41/42/43.
  paper_extended   Run M6-M11 (UPerNet/FPN/MANet/U-Net++/OCRNet/M3LSNet) with seeds 41/42/43.
  paper_all        Run paper_core, then paper_ablations, then paper_extended (57 runs).
  vram_smoke       Run the bounded synthetic VRAM probe.
  data_smoke       Check the configured dataset and read one sample.
  test             Evaluate a selected checkpoint on the labeled public test split.

Common overrides are read by the paper entry point, for example:
  DATA_ROOT=/path/to/mapped BATCH_SIZE=32 bash exp_train.sh paper_core
  SEEDS="41 42 43" DEVICE=cuda:0 bash exp_train.sh paper_extended
EOF
}

run_paper() {
  group="$1"
  echo "######## paper group=${group} ########"
  GROUP="${group}" bash "${REPO_ROOT}/scripts/train_paper_experiments.sh"
}

GROUP="${1:-paper_core}"
if [ "$#" -gt 1 ]; then
  usage >&2
  exit 2
fi

case "${GROUP}" in
  paper_core)
    run_paper core
    ;;
  paper_ablations)
    run_paper ablations
    ;;
  paper_extended)
    run_paper extended
    ;;
  paper_all|all)
    run_paper core
    run_paper ablations
    run_paper extended
    ;;
  vram_smoke|memory_smoke)
    sh "${REPO_ROOT}/tools/validation/smoke_test_vram.sh"
    ;;
  data_smoke|smoke)
    sh "${REPO_ROOT}/tools/validation/smoke_test_data.sh"
    ;;
  test|evaluate)
    bash "${REPO_ROOT}/tools/evaluation/evaluate_reproduction_test.sh"
    ;;
  help|-h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown experiment group: ${GROUP}" >&2
    usage >&2
    exit 2
    ;;
esac

echo "Experiment group completed: ${GROUP}"
