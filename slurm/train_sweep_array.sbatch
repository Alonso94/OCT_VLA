#!/usr/bin/env bash
# One array task = one sweep cell. Resources are supplied at submission, as for
# collection and single training runs.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
# No --mem: see the note in collect_shelf_restock_array.sbatch.

set -euo pipefail

: "${OCTVLA_REPO:?Set OCT_VLA checkout path}"
: "${SWEEP_MANIFEST:?Set the sweep manifest path}"
: "${SLURM_ARRAY_TASK_ID:?This script must be submitted as an array job}"

[ -f "$SWEEP_MANIFEST" ] || { echo "No manifest at $SWEEP_MANIFEST" >&2; exit 2; }

# The manifest is read here rather than baked into --export because a sweep's
# cells differ from each other, while --export is shared by the whole array.
# Task ids are 0-based; manifest lines are 1-based.
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SWEEP_MANIFEST")
[ -n "$LINE" ] || { echo "No manifest row for array task $SLURM_ARRAY_TASK_ID" >&2; exit 2; }

IFS=$'\t' read -r BACKBONE TRAIN_VARIANT OBJECT_TOKEN_MODE TRAIN_SEED <<< "$LINE"
: "${BACKBONE:?Malformed manifest row: $LINE}"
: "${TRAIN_VARIANT:?Malformed manifest row: $LINE}"
: "${OBJECT_TOKEN_MODE:?Malformed manifest row: $LINE}"
: "${TRAIN_SEED:?Malformed manifest row: $LINE}"
export BACKBONE TRAIN_VARIANT OBJECT_TOKEN_MODE TRAIN_SEED

echo "sweep cell : task=$SLURM_ARRAY_TASK_ID backbone=$BACKBONE variant=$TRAIN_VARIANT mode=$OBJECT_TOKEN_MODE seed=$TRAIN_SEED"

# Hand off to the single-run script rather than duplicating its body. That
# script is the one that has actually been run end to end, and a sweep whose
# cells were trained by a near-copy of it would be comparing against a baseline
# nobody verified. Its #SBATCH lines are inert comments when run this way; the
# allocation is the one this array job already holds.
exec bash "$OCTVLA_REPO/slurm/train_shelf_restock.sbatch"
