#!/usr/bin/env bash
# Evaluate several checkpoints of ONE training run closed-loop, to find the
# step count at which success saturates.
#
# This exists because offline loss is a weak proxy for closed-loop success on a
# VLA: eval loss on this task plateaus by ~step 1500 while the policy still
# cannot complete an episode. Sizing the sweep off the loss curve would either
# waste ~4x the GPU-hours or quietly under-train every arm, and only a
# closed-loop measurement distinguishes the two.
#
# Example: ./slurm/submit_checkpoint_sweep.sh $OCTVLA_OUTPUT_ROOT/pi05_rgb_full_s1000 rgb a40

set -euo pipefail

RUN_DIR="${1:?Usage: $0 <run_dir> rgb|object a40|a100 [--dry-run]}"
VARIANT="${2:?Missing variant}"
GPU="${3:?Missing GPU profile}"
DRY_RUN="${4:-}"

case "$VARIANT" in rgb|object) ;; *) echo "Variant must be rgb or object" >&2; exit 2;; esac
case "$GPU" in a40|a100) ;; *) echo "GPU must be a40 or a100" >&2; exit 2;; esac
case "$DRY_RUN" in ""|--dry-run) ;; *) echo "Only --dry-run is supported" >&2; exit 2;; esac

: "${SLURM_ACCOUNT:?Set cluster account}"
: "${SLURM_PARTITION:?Set cluster partition}"
: "${OCTVLA_OUTPUT_ROOT:?Set output root}"
export OCTVLA_REPO="${OCTVLA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"

[ -d "$RUN_DIR/checkpoints" ] || { echo "No checkpoints under $RUN_DIR" >&2; exit 2; }

# Every checkpoint is scored on the SAME scene seeds and profiles. Comparing
# step counts across different scenes would confound "more training" with
# "easier draw", which is the same pairing argument the arms rely on.
SEEDS="${SWEEP_EVAL_SEEDS:-800-819}"
PROFILES="${SWEEP_EVAL_PROFILES:-three_object}"

# `last` is a symlink to one of the numbered directories; following it too
# would score that checkpoint twice under two names.
mapfile -t CHECKPOINTS < <(find "$RUN_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
[ "${#CHECKPOINTS[@]}" -gt 0 ] || { echo "No numbered checkpoints yet in $RUN_DIR" >&2; exit 2; }

RUN_NAME=$(basename "$RUN_DIR")
echo "run       : $RUN_NAME"
echo "variant   : $VARIANT"
echo "seeds     : $SEEDS"
echo "profiles  : $PROFILES"
echo "steps     : ${CHECKPOINTS[*]}"

mkdir -p "$OCTVLA_OUTPUT_ROOT/slurm_logs"
for step in "${CHECKPOINTS[@]}"; do
  CHECKPOINT="$RUN_DIR/checkpoints/$step/pretrained_model"
  [ -d "$CHECKPOINT" ] || { echo "  skip $step (no pretrained_model)"; continue; }
  if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "  would submit $RUN_NAME @ $step"
    continue
  fi
  sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --gres="gpu:$GPU:1" \
    --time="${SWEEP_EVAL_TIME:-04:00:00}" --job-name="octvla-ckpt-$step" \
    --output="$OCTVLA_OUTPUT_ROOT/slurm_logs/ckpt-${RUN_NAME}-${step}-%j.out" \
    --error="$OCTVLA_OUTPUT_ROOT/slurm_logs/ckpt-${RUN_NAME}-${step}-%j.out" \
    --export="ALL,EVAL_CHECKPOINT=$CHECKPOINT,EVAL_VARIANT=$VARIANT,EVAL_SEEDS=$SEEDS,EVAL_PROFILES=$PROFILES" \
    "$OCTVLA_REPO/slurm/eval_shelf_restock.sbatch"
done
[ "$DRY_RUN" = "--dry-run" ] && echo "dry run: nothing submitted"
exit 0
