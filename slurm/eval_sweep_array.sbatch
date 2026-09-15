#!/usr/bin/env bash
# One array task = the closed-loop evaluation of one sweep cell on ONE object
# profile. Submitted with --dependency=aftercorr on the training array, so task
# N evaluates exactly the run trained by task N and only if that run succeeded.
#
# One profile per job, deliberately. A cell evaluated on all three profiles at
# 30 seeds needs ~12 h against this job's 8 h limit (docs/cluster_envelope.md),
# so the profiles are split rather than truncated -- count shift is the sweep's
# free generalisation axis and dropping it would be the wrong saving.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
# No --mem: see the note in collect_shelf_restock_array.sbatch.

set -euo pipefail

: "${OCTVLA_REPO:?Set OCT_VLA checkout path}"
: "${OCTVLA_OUTPUT_ROOT:?Set output root}"
: "${SWEEP_MANIFEST:?Set the sweep manifest path}"
: "${EVAL_PROFILE:?Set the object-count profile to evaluate}"
: "${SLURM_ARRAY_TASK_ID:?This script must be submitted as an array job}"

[ -f "$SWEEP_MANIFEST" ] || { echo "No manifest at $SWEEP_MANIFEST" >&2; exit 2; }

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SWEEP_MANIFEST")
[ -n "$LINE" ] || { echo "No manifest row for array task $SLURM_ARRAY_TASK_ID" >&2; exit 2; }
IFS=$'\t' read -r VARIANT TOKEN_MODE SEED <<< "$LINE"

RUN_NAME="pi05_${VARIANT}_${TOKEN_MODE}_s${SEED}"
RUN_DIR="$OCTVLA_OUTPUT_ROOT/$RUN_NAME"

# `last` rather than a hard-coded step: it is whatever the run actually reached.
# Naming the final step here would silently evaluate a stale checkpoint if the
# run were ever resized, and evaluate nothing at all if it stopped early.
CHECKPOINT="$RUN_DIR/checkpoints/last/pretrained_model"
[ -d "$CHECKPOINT" ] || { echo "No checkpoint at $CHECKPOINT" >&2; exit 2; }

SUFFIX=""
[ "${EVAL_SHUFFLE_TOKENS:-0}" = "1" ] && SUFFIX="__shuffled"

# Results go in a per-stage directory. Aggregation globs a directory, so a
# one-off exploratory evaluation left in the shared eval/ folder would be
# merged into whichever arm its checkpoint belongs to and silently shift that
# arm's success rate -- a 3-episode probe of an early checkpoint diluting a
# 90-episode cell.
: "${SWEEP_STAGE:?Set the sweep stage name}"

export EVAL_CHECKPOINT="$CHECKPOINT"
export EVAL_VARIANT="$VARIANT"
export EVAL_PROFILES="$EVAL_PROFILE"
export EVAL_SEEDS="${EVAL_SEEDS:-800-829}"
export EVAL_TAG="${SWEEP_STAGE}/${RUN_NAME}__${EVAL_PROFILE}${SUFFIX}"

echo "eval cell  : task=$SLURM_ARRAY_TASK_ID run=$RUN_NAME profile=$EVAL_PROFILE shuffled=${EVAL_SHUFFLE_TOKENS:-0}"
echo "checkpoint : $(readlink -f "$CHECKPOINT")"

# Hand off to the single-run evaluation script for the same reason the training
# array does: that is the path that has been run end to end.
exec bash "$OCTVLA_REPO/slurm/eval_shelf_restock.sbatch"
