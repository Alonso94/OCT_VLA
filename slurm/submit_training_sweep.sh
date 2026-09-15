#!/usr/bin/env bash
# Submit a sweep stage as one Slurm array. A --dry-run first is the safe habit:
# it prints the cells and the manifest without allocating anything.
#
# Example: ./slurm/submit_training_sweep.sh stage2 a40 --dry-run
#          TRAIN_STEPS=6000 ./slurm/submit_training_sweep.sh stage2 a40

set -euo pipefail

STAGE="${1:?Usage: $0 stage2|stage3-smolvla a40|a100 [--dry-run]}"
GPU="${2:?Missing GPU profile}"
DRY_RUN="${3:-}"

case "$GPU" in a40|a100) ;; *) echo "GPU must be a40 or a100" >&2; exit 2;; esac
case "$DRY_RUN" in ""|--dry-run) ;; *) echo "Only --dry-run is supported" >&2; exit 2;; esac

: "${SLURM_ACCOUNT:?Set cluster account}"
: "${SLURM_PARTITION:?Set cluster partition}"
: "${OCTVLA_DATASET_ROOT:?Set durable LeRobot dataset root}"
: "${OCTVLA_OUTPUT_ROOT:?Set training output root}"
: "${OCTVLA_POLICY_PYTHON:?Set LeRobot Python executable}"
export OCTVLA_REPO="${OCTVLA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"

# Three training seeds per arm. Seeds are the sweep's only source of error bars
# on the training side, so they are fixed here rather than left to the caller:
# two arms compared at different seeds are not comparable.
SEEDS=(${SWEEP_SEEDS:-1000 1001 1002})

# Cells as "backbone<TAB>variant<TAB>token_mode". The arms are stated here,
# once, so the manifest and the write-up cannot disagree about what was run.
#
# A: RGB baseline. B: full object tokens -- the headline claim. C:
# role-stripped -- is B real, or task-state leakage? Without C a win for B is
# uninterpretable, so C is part of a stage, not an extra.
case "$STAGE" in
  stage2)
    ARMS=("pi05	rgb	full" "pi05	object	full" "pi05	object	role_stripped")
    ;;
  stage3-smolvla)
    # Deliberately the same three arms as stage2, so the backbone is the only
    # thing that differs between the two stages. Six cells (dropping C) would
    # be cheaper, but then a SmolVLA win could not be checked for the same
    # leakage the pi0.5 arm has to rule out.
    ARMS=("smolvla	rgb	full" "smolvla	object	full" "smolvla	object	role_stripped")
    ;;
  *)
    echo "Unknown stage: $STAGE (known: stage2, stage3-smolvla)" >&2; exit 2 ;;
esac

MANIFEST="$OCTVLA_OUTPUT_ROOT/sweeps/${STAGE}.tsv"
mkdir -p "$(dirname "$MANIFEST")"
: > "$MANIFEST"
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    printf '%s\t%s\n' "$arm" "$seed" >> "$MANIFEST"
  done
done
CELLS=$(wc -l < "$MANIFEST")

echo "stage    : $STAGE"
echo "cells    : $CELLS"
echo "steps    : ${TRAIN_STEPS:-20000}"
echo "manifest : $MANIFEST"
awk -F'\t' '{printf "  [%d] %s_%s_%s_s%s\n", NR-1, $1, $2, $3, $4}' "$MANIFEST"

# Refuse to submit a stage whose cells already exist, rather than letting each
# array task discover it and fail. One partly-submitted array is far harder to
# reason about afterwards than a refusal now.
CLASHES=$(awk -F'\t' -v root="$OCTVLA_OUTPUT_ROOT" \
  '{d = root "/" $1 "_" $2 "_" $3 "_s" $4; if (system("[ -e \"" d "\" ]") == 0) print "  " d}' "$MANIFEST")
if [ -n "$CLASHES" ]; then
  echo "Refusing to submit; these run directories already exist:" >&2
  echo "$CLASHES" >&2
  exit 2
fi

if [ "$DRY_RUN" = "--dry-run" ]; then
  echo "dry run: nothing submitted"
  exit 0
fi

mkdir -p "$OCTVLA_OUTPUT_ROOT/slurm_logs"
LOGS="$OCTVLA_OUTPUT_ROOT/slurm_logs"
JOB_ID=$(sbatch --parsable \
  --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --gres="gpu:$GPU:1" \
  --array="0-$((CELLS - 1))" --job-name="octvla-sweep-$STAGE" \
  --output="$LOGS/%x-%A_%a.out" --error="$LOGS/%x-%A_%a.out" \
  --export="ALL,SWEEP_MANIFEST=$MANIFEST" \
  "$OCTVLA_REPO/slurm/train_sweep_array.sbatch")
echo "training array: $JOB_ID"

# Evaluation is chained, not run by hand later. `aftercorr` pairs array task N
# of the evaluation with array task N of the training array, so each cell is
# scored by the job that trained it and only if that job succeeded -- no
# checkpoint-path bookkeeping, and a failed cell is skipped rather than scored
# against a stale or absent checkpoint.
#
# One evaluation array per profile, because a cell on all three profiles needs
# ~12 h against the evaluation job's 8 h limit.
EVAL_PROFILES_LIST=(${SWEEP_EVAL_PROFILES:-two_object three_object four_object})
EVAL_JOBS=()
for profile in "${EVAL_PROFILES_LIST[@]}"; do
  EVAL_JOBS+=("$(sbatch --parsable \
    --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --gres="gpu:$GPU:1" \
    --array="0-$((CELLS - 1))" --job-name="octvla-eval-$profile" \
    --dependency="aftercorr:$JOB_ID" --kill-on-invalid-dep=yes \
    --output="$LOGS/%x-%A_%a.out" --error="$LOGS/%x-%A_%a.out" \
    --export="ALL,SWEEP_MANIFEST=$MANIFEST,SWEEP_STAGE=$STAGE,EVAL_PROFILE=$profile,EVAL_SEEDS=${SWEEP_EVAL_SEEDS:-800-829}" \
    "$OCTVLA_REPO/slurm/eval_sweep_array.sbatch")")
done
echo "eval arrays   : ${EVAL_JOBS[*]}"

# The shuffled control reuses the object arm's weights, so it trains nothing
# and is submitted over exactly those cells. Cheap, and it is the only check
# that the policy reads the tokens at all rather than having learned to ignore
# a channel that never helped.
SHUFFLE_CELLS=$(awk -F'\t' '$2 == "object" && $3 == "full" {print NR - 1}' "$MANIFEST" | paste -sd,)
if [ -n "$SHUFFLE_CELLS" ]; then
  for profile in "${EVAL_PROFILES_LIST[@]}"; do
    EVAL_JOBS+=("$(sbatch --parsable \
      --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --gres="gpu:$GPU:1" \
      --array="$SHUFFLE_CELLS" --job-name="octvla-shuf-$profile" \
      --dependency="aftercorr:$JOB_ID" --kill-on-invalid-dep=yes \
      --output="$LOGS/%x-%A_%a.out" --error="$LOGS/%x-%A_%a.out" \
      --export="ALL,SWEEP_MANIFEST=$MANIFEST,SWEEP_STAGE=$STAGE,EVAL_PROFILE=$profile,EVAL_SHUFFLE_TOKENS=1,EVAL_SEEDS=${SWEEP_EVAL_SEEDS:-800-829}" \
      "$OCTVLA_REPO/slurm/eval_sweep_array.sbatch")")
  done
  echo "shuffled ctrl : cells $SHUFFLE_CELLS"
fi

# afterany, not afterok: a sweep with one failed cell still has a report worth
# reading, and the arm table shows the gap as missing episodes.
DEPENDENCY=$(printf "afterany:%s," "${EVAL_JOBS[@]}")
AGG_JOB=$(sbatch --parsable \
  --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --gres="gpu:$GPU:1" \
  --job-name="octvla-aggregate-$STAGE" \
  --dependency="${DEPENDENCY%,}" --kill-on-invalid-dep=yes \
  --output="$LOGS/%x-%j.out" --error="$LOGS/%x-%j.out" \
  --export="ALL,SWEEP_STAGE=$STAGE" \
  "$OCTVLA_REPO/slurm/aggregate_sweep.sbatch")
echo "aggregation   : $AGG_JOB"
echo
echo "results   : $OCTVLA_OUTPUT_ROOT/eval/$STAGE/<run>__<profile>[__shuffled].json"
echo "report    : $OCTVLA_OUTPUT_ROOT/reports/${STAGE}-<timestamp>.json"
