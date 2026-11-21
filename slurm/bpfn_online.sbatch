#!/usr/bin/env bash
# BimanualPFN closed-loop pilot (stage 2b): the eval server for the assigned rule plus the
# direct controller client, one job. Object colours: OCTVLA_COLOURS (fixed across the
# supports compared, so a seed renders the same scene for both).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
set -uo pipefail
: "${OCTVLA_REPO:?}"; : "${OCTVLA_ROBOTWIN_ROOT:?}"; : "${OCTVLA_ROBOTWIN_PYTHON:?}"
: "${OCTVLA_POLICY_PYTHON:?}"; : "${BPFN_RULE:?assigned rule}"; : "${BPFN_OUT:?result json}"
: "${BPFN_CONTROLLER:?}"; : "${BPFN_GRIPPER:?}"; : "${BPFN_SEEDS:?}"
export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH OCTVLA_MODEL_IDS
PORT="${EVAL_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
READY="${TMPDIR:-/tmp}/bpfn-online-ready-$SLURM_JOB_ID"; rm -f "$READY"
cd "$OCTVLA_ROBOTWIN_ROOT"
"$OCTVLA_ROBOTWIN_PYTHON" -u -B "$OCTVLA_REPO/scripts/robotwin_eval_server.py" \
  --robotwin-root "$OCTVLA_ROBOTWIN_ROOT" --host 127.0.0.1 --port "$PORT" \
  --ready-file "$READY" --rule "$BPFN_RULE" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null' EXIT
for _ in $(seq 1 120); do
  [ -f "$READY" ] && break
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died during startup" >&2; exit 1; }
  sleep 5
done
[ -f "$READY" ] || { echo "server not ready in 600s" >&2; exit 1; }
EXTRA=()
[ -n "${BPFN_RUN:-}" ] && EXTRA+=(--run "$BPFN_RUN")
[ -n "${BPFN_SUPPORT:-}" ] && EXTRA+=(--support "$BPFN_SUPPORT")
"$OCTVLA_POLICY_PYTHON" -u "$OCTVLA_REPO/scripts/bpfn/stage2_online_pilot.py" \
  --controller "$BPFN_CONTROLLER" --gripper "$BPFN_GRIPPER" --seeds "$BPFN_SEEDS" \
  --profile "${BPFN_PROFILE:-four_object}" --k "${BPFN_K:-1}" --port "$PORT" \
  --post-success-steps "${BPFN_POST_SUCCESS:-30}" --out "$BPFN_OUT" \
  --trace-dir "${BPFN_OUT%.json}_traces" --tag "${BPFN_TAG:-}" "${EXTRA[@]}"
