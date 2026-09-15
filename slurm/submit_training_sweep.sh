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

# Cells as "variant<TAB>token_mode". The arms are stated here, once, so the
# manifest and the write-up cannot disagree about what was run.
case "$STAGE" in
  stage2)
    # A: RGB baseline. B: full object tokens -- the headline claim.
    # C: role-stripped -- is B real, or task-state leakage? Without C a win
    # for B is uninterpretable, so C is part of the stage, not an extra.
    ARMS=("rgb	full" "object	full" "object	role_stripped")
    ;;
  *)
    echo "Unknown stage: $STAGE (known: stage2)" >&2; exit 2 ;;
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
awk -F'\t' '{printf "  [%d] pi05_%s_%s_s%s\n", NR-1, $1, $2, $3}' "$MANIFEST"

# Refuse to submit a stage whose cells already exist, rather than letting each
# array task discover it and fail. One partly-submitted array is far harder to
# reason about afterwards than a refusal now.
CLASHES=$(awk -F'\t' -v root="$OCTVLA_OUTPUT_ROOT" \
  '{d = root "/pi05_" $1 "_" $2 "_s" $3; if (system("[ -e \"" d "\" ]") == 0) print "  " d}' "$MANIFEST")
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
JOB_ID=$(sbatch --parsable \
  --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --gres="gpu:$GPU:1" \
  --array="0-$((CELLS - 1))" --job-name="octvla-sweep-$STAGE" \
  --output="$OCTVLA_OUTPUT_ROOT/slurm_logs/%x-%A_%a.out" \
  --error="$OCTVLA_OUTPUT_ROOT/slurm_logs/%x-%A_%a.out" \
  --export="ALL,SWEEP_MANIFEST=$MANIFEST" \
  "$OCTVLA_REPO/slurm/train_sweep_array.sbatch")
echo "sweep job: $JOB_ID"
