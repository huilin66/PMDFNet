#!/usr/bin/env sh
set -eu

# Batch Test evaluation over the frozen formal runs. Reads each run's config.json
# and checkpoints/best.pth and writes test_metrics.json into the run directory.
# Only run this after model, hyper-parameters and checkpoint selection are frozen.
#
# It expects the complete formal matrix in RUN_ROOT and aborts otherwise:
#   M1-M4 + M6-M11 (10) + A1 A2 A5 B2 B3 C2 (6) + L1 L2 L3 (3) = 19 x 3 = 57 runs
# Backups and hand-made copies under RUN_ROOT (_backup*, _archive*, _old*, dot
# directories, <run>_s41_..._DEBUG) are pruned, so they cannot inflate the count.
#
# Overrides:
#   DATA_ROOT=...            mapped MMLSv2 root with a labeled test split
#   RUN_ROOT=outputs_experiments/pmdfnet_mmlsv2
#   DEVICE=cuda:0
#   BATCH_SIZE=32
#   EXPECTED_RUNS=57         lower this to evaluate a partial matrix on purpose
#
# Partial evaluation example (after training only the core group):
#   EXPECTED_RUNS=30 bash scripts/run_test_eval.sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

. "${SCRIPT_DIR}/_env.sh"

RUN_ROOT="${RUN_ROOT:-outputs_experiments/pmdfnet_mmlsv2}"
EXPECTED_RUNS="${EXPECTED_RUNS:-57}"
export RUN_ROOT EXPECTED_RUNS

echo "######## test evaluation: runs=${RUN_ROOT} expected=${EXPECTED_RUNS} ########"
bash "${REPO_ROOT}/tools/evaluation/evaluate_paper_tests.sh"
