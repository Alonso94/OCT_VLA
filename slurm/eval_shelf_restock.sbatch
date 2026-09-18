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
: "${EVAL_VARIANT:?Set rgb, object or privileged}"
: "${EVAL_SEEDS:?Set evaluation seeds, e.g. 800-829}"

# `privileged` selects the vision-free dataset, whose observation.environment_state
# is what tells the evaluator this checkpoint reads privileged scene state rather
# than cameras. Pointing it at an rgb dataset would leave that undetected and the
# policy would be fed an observation it was never trained on.
case "$EVAL_VARIANT" in rgb|object|privileged) ;; *) echo "Invalid EVAL_VARIANT" >&2; exit 2 ;; esac

export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID"
# No LD_LIBRARY_PATH here, deliberately. This job runs the simulator, and the
# torchcodec FFmpeg symlinks belong to the policy virtualenv -- exposing them to
# SAPIEN stops it importing. The policy client needs them anyway: video is
# decoded at training time, while here frames arrive over the bridge as raw RGB.
unset LD_LIBRARY_PATH

# A free port per job, so two evaluations on one node cannot collide.
EVAL_PORT="${EVAL_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
READY="${TMPDIR:-/tmp}/octvla-eval-ready-$SLURM_JOB_ID"
# EVAL_TAG names the result after the cell it scores, not after the job that
# happened to run it. A sweep produces dozens of these, and a job id alone
# cannot say which arm, seed or profile a file describes once the Slurm
# accounting database has aged out.
RESULTS="$OCTVLA_OUTPUT_ROOT/eval/${EVAL_TAG:-${EVAL_VARIANT}_${SLURM_JOB_ID}}.json"
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
# The shuffled control: same weights as the object arm, tokens permuted across
# objects. A policy that genuinely reads them degrades; one that ignores them
# scores the same. Evaluation only -- training with shuffled tokens would teach
# the policy to ignore them, which is the opposite of what this detects.
if [ "${EVAL_SHUFFLE_TOKENS:-0}" = "1" ]; then
  [ "$EVAL_VARIANT" = "object" ] || { echo "EVAL_SHUFFLE_TOKENS needs EVAL_VARIANT=object" >&2; exit 2; }
  OBJECT_FLAG+=(--shuffle-tokens)
fi

VIDEO_FLAG=()
if [ -n "${EVAL_VIDEO_DIR:-}" ]; then
  VIDEO_FLAG+=(--video-dir "$EVAL_VIDEO_DIR")
  [ -n "${EVAL_VIDEO_LABEL:-}" ] && VIDEO_FLAG+=(--video-label "$EVAL_VIDEO_LABEL")
  [ -n "${EVAL_VIDEO_SEEDS:-}" ] && VIDEO_FLAG+=(--video-seeds "$EVAL_VIDEO_SEEDS")
  [ -n "${EVAL_VIDEO_NAME:-}" ] && VIDEO_FLAG+=(--video-name "$EVAL_VIDEO_NAME")
  [ -n "${EVAL_VIDEO_CRF:-}" ] && VIDEO_FLAG+=(--video-crf "$EVAL_VIDEO_CRF")
fi

cd "$OCTVLA_REPO"
# The dataset supplies the action width, the control space and the state
# encoding, so it has to be the one the checkpoint was fit on -- not merely one
# of the same variant. EVAL_DATASET names it directly; the default reproduces
# the original three_object_<variant> convention.
EVAL_DATASET="${EVAL_DATASET:-three_object_${EVAL_VARIANT}}"
EVAL_DATASET_DIR="$OCTVLA_DATASET_ROOT/$EVAL_DATASET"
[ -d "$EVAL_DATASET_DIR" ] || { echo "No dataset at $EVAL_DATASET_DIR" >&2; exit 2; }
# Matches how build_shelf_restock_splits.sbatch names the repo it created; a
# dataset built under the old convention keeps its three_object- form.
if [ "$EVAL_DATASET" = "three_object_${EVAL_VARIANT}" ]; then
  EVAL_REPO_ID="local/oct-vla-shelf-restock-three_object-${EVAL_VARIANT}"
else
  EVAL_REPO_ID="local/oct-vla-shelf-restock-${EVAL_DATASET#three_object_}"
fi
echo "eval dataset : $EVAL_DATASET_DIR ($EVAL_REPO_ID)"

"$OCTVLA_POLICY_PYTHON" -u -B scripts/eval_shelf_restock_policy.py \
  --checkpoint "$EVAL_CHECKPOINT" \
  --dataset-root "$EVAL_DATASET_DIR" \
  --repo-id "$EVAL_REPO_ID" \
  --host 127.0.0.1 --port "$EVAL_PORT" \
  --profiles "${EVAL_PROFILES:-two_object,three_object,four_object}" \
  --seeds "$EVAL_SEEDS" \
  --max-steps "${EVAL_MAX_STEPS:-600}" \
  "${OBJECT_FLAG[@]}" \
  "${VIDEO_FLAG[@]}" \
  --output "$RESULTS"
STATUS=$?
echo "eval exit=$STATUS; results at $RESULTS"
exit "$STATUS"
