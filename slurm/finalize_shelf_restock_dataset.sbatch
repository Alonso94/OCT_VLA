#!/usr/bin/env bash
# One writer process finalizes canonical clips after collection has completed.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

: "${OCTVLA_REPO:?Set OCT_VLA checkout path}"
: "${OCTVLA_POLICY_PYTHON:?Set LeRobot Python executable}"
: "${OCTVLA_COLLECTION_ROOT:?Set canonical collection root}"
: "${OCTVLA_DATASET_ROOT:?Set durable LeRobot dataset root}"
: "${COLLECTION_PROFILE:?Set two_object, three_object or four_object}"
: "${DATASET_REPO_PREFIX:?Set Hub-style dataset prefix, e.g. org/oct-vla-three-object}"

CANONICAL="$OCTVLA_COLLECTION_ROOT/$COLLECTION_PROFILE"
RGB_OUTPUT="$OCTVLA_DATASET_ROOT/${COLLECTION_PROFILE}_rgb"
OBJECT_OUTPUT="$OCTVLA_DATASET_ROOT/${COLLECTION_PROFILE}_object"
export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

if [ -e "$RGB_OUTPUT" ] || [ -e "$OBJECT_OUTPUT" ]; then
  echo "Refusing to overwrite finalized dataset output" >&2
  exit 2
fi

srun "$OCTVLA_POLICY_PYTHON" -u -B "$OCTVLA_REPO/scripts/finalize_shelf_restock_dataset.py" \
  --canonical-root "$CANONICAL" \
  --output "$RGB_OUTPUT" \
  --repo-id "$DATASET_REPO_PREFIX-rgb"
srun "$OCTVLA_POLICY_PYTHON" -u -B "$OCTVLA_REPO/scripts/finalize_shelf_restock_dataset.py" \
  --canonical-root "$CANONICAL" \
  --output "$OBJECT_OUTPUT" \
  --repo-id "$DATASET_REPO_PREFIX-object" \
  --object-tokens
