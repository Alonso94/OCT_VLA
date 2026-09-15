#!/usr/bin/env bash
# One RoboTwin scene seed per array task. Resources are supplied at submission.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
# No --mem: on GPU clusters that allocate memory in proportion to the GPU
# count (NHR@FAU Alex among them) an explicit --mem is rejected outright --
# "Do not specify --mem for GPU jobs!" -- and submission fails. One a40/a100
# GPU carries ~64 GB there, comfortably above the ~3 GB a collection task
# actually uses. Add --mem at submission time on sites that require it.

set -euo pipefail

: "${OCTVLA_REPO:?Set OCT_VLA checkout path}"
: "${OCTVLA_ROBOTWIN_ROOT:?Set RoboTwin checkout path}"
: "${OCTVLA_ROBOTWIN_PYTHON:?Set RoboTwin Python 3.10 executable}"
: "${OCTVLA_COLLECTION_ROOT:?Set durable canonical collection root}"
: "${COLLECTION_PROFILE:?Set two_object, three_object or four_object}"

case "$COLLECTION_PROFILE" in
  two_object|three_object|four_object) ;;
  *) echo "Invalid COLLECTION_PROFILE=$COLLECTION_PROFILE" >&2; exit 2 ;;
esac

SEED="${SLURM_ARRAY_TASK_ID:?Slurm array index is required}"
DESTINATION="$OCTVLA_COLLECTION_ROOT/$COLLECTION_PROFILE/seed_$SEED"
mkdir -p "$DESTINATION"

export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID"

cd "$OCTVLA_ROBOTWIN_ROOT"
srun "$OCTVLA_ROBOTWIN_PYTHON" -u -B "$OCTVLA_REPO/scripts/collect_shelf_restock.py" \
  --robotwin-root "$OCTVLA_ROBOTWIN_ROOT" \
  --output "$DESTINATION" \
  --profile "$COLLECTION_PROFILE" \
  --seeds "$SEED"
