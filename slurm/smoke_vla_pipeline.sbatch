#!/usr/bin/env bash
# End-to-end smoke test of the two-stage VLA pipeline for one backbone, before
# any real run is queued. Everything heavy lives in node-local $TMPDIR and is
# discarded with the job; only reports are copied back (SMOKE_REPORTS).
#
#   BACKBONE=pi05|smolvla|groot sbatch ... slurm/smoke_vla_pipeline.sbatch
#
# Exercises, in the order a real run would:
#   stage 1 (rgb, a few steps)           -> slim save for GR00T, final-only save,
#                                           optimiser state removed
#   merge (pi05, smolvla)                -> scripts/merge_stage1_adapter.py
#   INIT_CHECK for rgb_cont, kv,         -> stage 2 starts at stage 1; gradient
#     kv_adaln, kv_tokens                   reaches each branch
#   stage 2, each arm, a few steps       -> verified load, PEFT keeps the branch,
#                                           checkpoint saves and reloads
#   rollout of kv_adaln and kv_tokens    -> the branches at inference, through
#     (1 scene, a few steps)                the KV-cache / denoising path
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00

set -euo pipefail
: "${BACKBONE:?Set pi05, smolvla or groot}"
: "${OCTVLA_REPO:?}"; : "${OCTVLA_POLICY_PYTHON:?}"; : "${OCTVLA_OUTPUT_ROOT:?}"
DATASET_NAME="${SMOKE_DATASET:-three_object_identity_eeabs}"
REPORTS="${SMOKE_REPORTS:-$OCTVLA_OUTPUT_ROOT/smoke_vla/$BACKBONE}"
mkdir -p "$REPORTS"
LOCAL="${TMPDIR:?node-local TMPDIR required}/smoke-$BACKBONE"
mkdir -p "$LOCAL"
df -h "$TMPDIR" | tail -1 | sed 's/^/node-local : /'
export OCTVLA_OUTPUT_ROOT="$LOCAL"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-g107ea}"
cd "$OCTVLA_REPO"

common=(TRAIN_DATASET="$DATASET_NAME" TRAIN_STEPS="${SMOKE_STEPS:-10}" BATCH_SIZE=2
        NUM_WORKERS=2 EVAL_STEPS=5 MAX_EVAL_SAMPLES=4 N_ACTION_STEPS=25 BACKBONE="$BACKBONE")
step() { echo; echo "===== $* ====="; }
status=0
record() { echo "$1: $2" | tee -a "$REPORTS/summary.txt"; [ "$2" = ok ] || status=1; }
: > "$REPORTS/summary.txt"

step "stage 1"
env "${common[@]}" TRAIN_VARIANT=rgb bash slurm/train_shelf_restock.sbatch && record stage1 ok || record stage1 FAILED
RUN1="$LOCAL/${BACKBONE}_rgb_s1000"
STAGE1="$RUN1/checkpoints/last/pretrained_model"
ls "$STAGE1" | sed 's/^/  stage1 file: /'
du -sh "$RUN1" | sed 's/^/  stage1 size: /' | tee -a "$REPORTS/summary.txt"
[ -d "$RUN1/checkpoints/last/training_state" ] && record optimiser-state-removed FAILED || record optimiser-state-removed ok

if [ "$BACKBONE" != groot ]; then
  step "merge"
  "$OCTVLA_POLICY_PYTHON" -u scripts/merge_stage1_adapter.py --stage1 "$STAGE1" \
    --dataset-root "$OCTVLA_DATASET_ROOT/$DATASET_NAME" \
    --repo-id "local/oct-vla-shelf-restock-${DATASET_NAME#three_object_}" \
    --output "$RUN1/merged" && record merge ok || record merge FAILED
  cp "$RUN1/merged/merge_provenance.json" "$REPORTS/" 2>/dev/null || true
  STAGE1="$RUN1/merged"
else
  [ -f "$STAGE1/slim_checkpoint.json" ] && record slim-save ok || record slim-save FAILED
fi

arm_env() {
  case "$1" in
    rgb_cont) echo TRAIN_VARIANT=rgb ;;
    *)        echo TRAIN_VARIANT=object CONDITIONING="$1" ;;
  esac
}
for arm in rgb_cont kv kv_adaln kv_tokens; do
  step "init check: $arm"
  # shellcheck disable=SC2046
  env "${common[@]}" $(arm_env $arm) STAGE1_CHECKPOINT="$STAGE1" TRAIN_TAG="check_$arm" \
    INIT_CHECK=1 bash slurm/train_shelf_restock.sbatch && record "init-$arm" ok || record "init-$arm" FAILED
done
cp "$LOCAL"/run_metadata/*_init_check.json "$REPORTS/" 2>/dev/null || true

for arm in rgb_cont kv kv_adaln kv_tokens; do
  step "stage 2: $arm"
  # shellcheck disable=SC2046
  env "${common[@]}" $(arm_env $arm) STAGE1_CHECKPOINT="$STAGE1" TRAIN_TAG="s2_$arm" \
    bash slurm/train_shelf_restock.sbatch && record "stage2-$arm" ok || record "stage2-$arm" FAILED
done
du -sh "$LOCAL"/*_s2_* 2>/dev/null | sed 's/^/  stage2 size: /' | tee -a "$REPORTS/summary.txt"

for arm in kv_adaln kv_tokens; do
  step "rollout: $arm"
  RUN2=$(ls -d "$LOCAL"/*_s2_"$arm" | head -1)
  EVAL_CHECKPOINT="$RUN2/checkpoints/last/pretrained_model" \
    EVAL_DATASET="$DATASET_NAME" EVAL_SEEDS=800 EVAL_PROFILES=three_object \
    EVAL_MAX_STEPS="${SMOKE_ROLLOUT_STEPS:-3}" EVAL_N_ACTION_STEPS=25 \
    EVAL_MODEL_IDS=1,2,3,4 EVAL_TAG="smoke-$BACKBONE-$arm" \
    bash slurm/eval_shelf_restock.sbatch && record "rollout-$arm" ok || record "rollout-$arm" FAILED
  cp "$LOCAL/eval/smoke-$BACKBONE-$arm.json" "$REPORTS/" 2>/dev/null || true
done

echo; echo "===== summary ====="; cat "$REPORTS/summary.txt"
exit "$status"
