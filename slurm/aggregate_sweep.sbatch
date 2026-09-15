#!/usr/bin/env bash
# Collect a finished sweep's evaluation JSONs into one report.
#
# Submitted with --dependency=afterany on every evaluation array, not afterok:
# a sweep where one cell failed still has a report worth reading, and the arm
# table makes the gap obvious by showing fewer episodes for that arm. Waiting
# for perfection would mean no report at all on the run most in need of one.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00

set -euo pipefail

: "${OCTVLA_REPO:?Set OCT_VLA checkout path}"
: "${OCTVLA_OUTPUT_ROOT:?Set output root}"
: "${SWEEP_STAGE:?Set the sweep stage name}"

# This stage's results only. The shared eval/ folder also holds one-off probes
# whose checkpoints belong to these same arms; globbing it would merge them in.
EVAL_DIR="$OCTVLA_OUTPUT_ROOT/eval/$SWEEP_STAGE"
[ -d "$EVAL_DIR" ] || { echo "No evaluation results at $EVAL_DIR" >&2; exit 2; }
REPORT_DIR="$OCTVLA_OUTPUT_ROOT/reports"
mkdir -p "$REPORT_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
REPORT="$REPORT_DIR/${SWEEP_STAGE}-${STAMP}.json"

# Stdlib only, so this needs no virtualenv and cannot fail on a missing wheel
# after a sweep that has already cost hundreds of GPU-hours.
PYTHON=$(command -v python3)

# Check the script reproduces the summaries the eval jobs themselves wrote
# before trusting anything new it says. Every arm comparison is built on the
# same episode parsing, so if this disagrees the report is meaningless.
echo "=== verifying aggregation against stored summaries ==="
"$PYTHON" "$OCTVLA_REPO/scripts/aggregate_eval.py" "$EVAL_DIR" --verify

echo
echo "=== $SWEEP_STAGE ==="
"$PYTHON" "$OCTVLA_REPO/scripts/aggregate_eval.py" "$EVAL_DIR" \
  --run-metadata "$OCTVLA_OUTPUT_ROOT/run_metadata" \
  --baseline rgb \
  --output "$REPORT"

echo
echo "report: $REPORT"
