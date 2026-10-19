#!/usr/bin/env sh
set -eu

# Unified SKG training launcher. It intentionally stops before final test
# evaluation: test is only evaluated manually after all val-based decisions are
# frozen.
#
# Default full pipeline:
#   DATA_ROOT=/path/to/mmlsv2 sh scripts/00_run_skg_training_pipeline.sh
#
# Useful staged runs:
#   STAGES="smoke" sh scripts/00_run_skg_training_pipeline.sh
#   STAGES="baselines" sh scripts/00_run_skg_training_pipeline.sh
#   STAGES="prototypes ablation" SEEDS="42" sh scripts/00_run_skg_training_pipeline.sh
#   SEEDS="42 123 7" K_VALUES="1 2 4 8" sh scripts/00_run_skg_training_pipeline.sh

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STAGES="${STAGES:-baselines prototypes ablation}"
STRICT_DETERMINISM="${STRICT_DETERMINISM:-1}"
export TRAIN_STRICT_DETERMINISM="${STRICT_DETERMINISM}"

has_stage() {
  requested="$1"
  for stage in ${STAGES}; do
    if [ "${stage}" = "${requested}" ]; then
      return 0
    fi
  done
  return 1
}

if has_stage "smoke"; then
  echo ""
  echo "######## Smoke test: all implemented tasks ########"
  sh scripts/00_smoke_test_skg.sh
fi

if has_stage "baselines"; then
  echo ""
  echo "######## Stage 1: fair baselines (train -> val) ########"
  sh scripts/01_train_baselines.sh
fi

if has_stage "prototypes"; then
  echo ""
  echo "######## Stage 2: train-only prototype construction ########"
  sh tools/knowledge/build_prototypes.sh
fi

if has_stage "ablation"; then
  echo ""
  echo "######## Stage 3: knowledge ablation (train -> val) ########"
  sh scripts/04_train_knowledge_ablation.sh
fi

echo ""
echo "SKG training pipeline completed for stages: ${STAGES}"
echo "Test was intentionally not evaluated. Freeze selected val checkpoints first, then run tools/evaluation/08_eval_final_test.sh manually."
