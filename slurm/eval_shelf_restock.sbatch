#!/usr/bin/env bash
# Closed-loop evaluation: simulator server and policy client in one job, each in
# its own interpreter. Runs after training, never during it -- SAPIEN and the
# policy would otherwise contend for the same GPU.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
# No --mem: see the note in collect_shelf_restock_array.sbatch.

set -uo pipefail

: "${OCTVLA_REPO:?Set OCT_VLA checkout path}"
: "${OCTVLA_ROBOTWIN_ROOT:?Set RoboTwin checkout path}"
: "${OCTVLA_ROBOTWIN_PYTHON:?Set RoboTwin Python 3.10 executable}"
: "${OCTVLA_POLICY_PYTHON:?Set LeRobot Python executable}"
: "${OCTVLA_DATASET_ROOT:?Set LeRobot dataset root}"
: "${OCTVLA_OUTPUT_ROOT:?Set output root}"
: "${EVAL_CHECKPOINT:?Set the trained policy directory}"
: "${EVAL_VARIANT:?Set rgb or object}"
: "${EVAL_SEEDS:?Set evaluation seeds, e.g. 800-829}"

case "$EVAL_VARIANT" in rgb|object) ;; *) echo "Invalid EVAL_VARIANT" >&2; exit 2 ;; esac

export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID"

# A free port per job, so two evaluations on one node cannot collide.
EVAL_PORT="${EVAL_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
READY="${TMPDIR:-/tmp}/octvla-eval-ready-$SLURM_JOB_ID"
RESULTS="$OCTVLA_OUTPUT_ROOT/eval/${EVAL_VARIANT}_${SLURM_JOB_ID}.json"
rm -f "$READY"

cd "$OCTVLA_ROBOTWIN_ROOT"
"$OCTVLA_ROBOTWIN_PYTHON" -u -B "$OCTVLA_REPO/scripts/robotwin_eval_server.py" \
  --robotwin-root "$OCTVLA_ROBOTWIN_ROOT" \
  --host 127.0.0.1 --port "$EVAL_PORT" --ready-file "$READY" &
SERVER_PID=$!
# Kill the simulator however this script exits, or it survives as an orphan
# holding a GPU render context for the rest of the allocation.
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT

# Wait on the ready file rather than sleeping: SAPIEN plus cuRobo warm-up takes
# tens of seconds and varies, so any fixed delay is a race or a waste.
for _ in $(seq 1 120); do
  [ -f "$READY" ] && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died during startup" >&2; exit 1; }
  sleep 5
done
[ -f "$READY" ] || { echo "server did not become ready in 600s" >&2; exit 1; }
echo "server ready on port $EVAL_PORT"

OBJECT_FLAG=()
[ "$EVAL_VARIANT" = "object" ] && OBJECT_FLAG=(--object-tokens)

cd "$OCTVLA_REPO"
"$OCTVLA_POLICY_PYTHON" -u -B scripts/eval_shelf_restock_policy.py \
  --checkpoint "$EVAL_CHECKPOINT" \
  --dataset-root "$OCTVLA_DATASET_ROOT/three_object_${EVAL_VARIANT}" \
  --repo-id "local/oct-vla-shelf-restock-three_object-${EVAL_VARIANT}" \
  --host 127.0.0.1 --port "$EVAL_PORT" \
  --profiles "${EVAL_PROFILES:-two_object,three_object,four_object}" \
  --seeds "$EVAL_SEEDS" \
  --max-steps "${EVAL_MAX_STEPS:-600}" \
  "${OBJECT_FLAG[@]}" \
  --output "$RESULTS"
STATUS=$?
echo "eval exit=$STATUS; results at $RESULTS"
exit "$STATUS"
