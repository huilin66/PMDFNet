#!/usr/bin/env bash
set -euo pipefail

# Evaluate the validation-selected checkpoint for all formal PMDFNet paper runs.
# Results are written to each project-side run directory. Dataset outputs are
# intentionally not touched; archive them separately after evaluation.

REPO_ROOT="$(CDPATH= cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

. "${REPO_ROOT}/scripts/_env.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
# Same resolution order as the trainer (and .env); no baked-in host path, so a
# stale default can never silently point the evaluation at the wrong dataset.
DATA_ROOT="${DATA_ROOT:-${MARS_DATA_ROOT:-${MAPPED_DATA_ROOT:-${MMLSV2_DATA_ROOT:-}}}}"
if [ -z "${DATA_ROOT}" ]; then
  printf 'DATA_ROOT is not set. Export DATA_ROOT (or MARS_DATA_ROOT/MAPPED_DATA_ROOT/MMLSV2_DATA_ROOT) to the mapped MMLSv2 root.\n' >&2
  exit 1
fi
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RUN_ROOT="${RUN_ROOT:-outputs_experiments/pmdfnet_mmlsv2}"

# Strip a trailing slash so the anchored pattern below stays valid.
RUN_ROOT="${RUN_ROOT%/}"

# Full formal matrix under the CE-only protocol:
#   M1-M4 core baselines              -> 4 x 3 = 12
#   M6-M11 extra E1 baselines         -> 6 x 3 = 18
#   A1 A2 A5 B2 B3 C2                 -> 6 x 3 = 18
#   L1 L2 L3 loss/fusion family       -> 3 x 3 =  9
#   total                                  57
# A3/A4/B1/C1 are display aliases over M4/L1 and are never trained separately.
EXPECTED_RUNS="${EXPECTED_RUNS:-57}"

# Anchor the match to RUN_ROOT and require the directory name to end exactly
# after the run timestamp that make_run_dir writes
# ({experiment_name}_s{seed}_{YYYYMMDD}_{HHMMSS}). This keeps backups
# (_backup*), hidden trees (.ipynb_checkpoints) and hand-made copies such as
# <run>_s41_..._DEBUG out of the formal run set; the -prune below additionally
# avoids walking those trees at all.
RUN_PATTERN="^${RUN_ROOT}/(M1|M2|M3|M4|M6|M7|M8|M9|M10|M11|A1|A2|A5|B2|B3|C2|L1|L2|L3)_[^/]*_s(41|42|43)_[0-9]{8}_[0-9]{6}\$"

mapfile -t RUN_DIRS < <(
  find "$RUN_ROOT" \
    \( -name '_backup*' -o -name '_archive*' -o -name '_old*' -o -name '.*' \) -prune -o \
    -type f -path '*/logs/train.log' \
    -exec grep -l 'Training completed' {} + \
    | sed 's#/logs/train.log##' \
    | grep -E "$RUN_PATTERN" \
    | sort
)

if [[ "${#RUN_DIRS[@]}" -ne "${EXPECTED_RUNS}" ]]; then
  printf 'Expected %s formal paper runs, found %s.\n' "${EXPECTED_RUNS}" "${#RUN_DIRS[@]}" >&2
  printf '%s\n' "${RUN_DIRS[@]}" >&2
  exit 1
fi

echo "[paper-test] data=$DATA_ROOT device=$DEVICE batch=$BATCH_SIZE workers=$NUM_WORKERS"
echo "[paper-test] formal_runs=${#RUN_DIRS[@]}"

for run_dir in "${RUN_DIRS[@]}"; do
  checkpoint="$run_dir/checkpoints/best.pth"
  run_config="$run_dir/config.json"
  if [[ ! -f "$checkpoint" || ! -f "$run_config" ]]; then
    echo "[paper-test] missing checkpoint/config: $run_dir" >&2
    exit 1
  fi

  if [[ -f "$run_dir/test_metrics.json" ]]; then
    echo "[paper-test] skip existing: $run_dir"
    continue
  fi

  echo "[paper-test] start: $run_dir"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" tools/evaluation/evaluate_test_metrics.py \
      --data-root "$DATA_ROOT" \
      --checkpoint "$checkpoint" \
      --run-config "$run_config" \
      --output-dir "$run_dir" \
      --device "$DEVICE" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS"
  echo "[paper-test] done: $run_dir"
done

echo "[paper-test] all formal runs evaluated"
