#!/usr/bin/env bash
# Submit data generation. A one-seed feasibility job is the safe first run.
# Example: ./slurm/submit_shelf_restock_collection.sh three_object a100 1000 --finalize

set -euo pipefail

PROFILE="${1:?Usage: $0 two_object|three_object|four_object a40|a100 seeds [--finalize]}"
GPU="${2:?Missing GPU profile}" 
SEEDS="${3:?Missing Slurm array seed expression, e.g. 1000-1009}"
FINALIZE="${4:-}"

case "$PROFILE" in two_object|three_object|four_object) ;; *) echo "Invalid profile" >&2; exit 2;; esac
case "$GPU" in a40|a100) ;; *) echo "GPU must be a40 or a100" >&2; exit 2;; esac
case "$FINALIZE" in ""|--finalize) ;; *) echo "Only --finalize is supported" >&2; exit 2;; esac

: "${SLURM_ACCOUNT:?Set cluster account}"
: "${SLURM_PARTITION:?Set cluster partition}"
: "${OCTVLA_REPO:=$(cd "$(dirname "$0")/.." && pwd)}"
: "${OCTVLA_ROBOTWIN_ROOT:?Set RoboTwin checkout path}"
: "${OCTVLA_ROBOTWIN_PYTHON:?Set RoboTwin Python executable}"
: "${OCTVLA_COLLECTION_ROOT:?Set durable canonical collection root}"
: "${OCTVLA_DATASET_ROOT:?Set durable LeRobot dataset root}"
: "${OCTVLA_POLICY_PYTHON:?Set LeRobot Python executable}"
: "${DATASET_REPO_PREFIX:=local/oct-vla-shelf-restock-$PROFILE}"

mkdir -p "$OCTVLA_COLLECTION_ROOT/slurm_logs"
JOB_ID=$(sbatch --parsable \
  --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --gres="gpu:$GPU:1" \
  --array="$SEEDS" --job-name="octvla-collect-$PROFILE" \
  --output="$OCTVLA_COLLECTION_ROOT/slurm_logs/%x-%A_%a.out" \
  --error="$OCTVLA_COLLECTION_ROOT/slurm_logs/%x-%A_%a.err" \
  --export="ALL,COLLECTION_PROFILE=$PROFILE" \
  "$OCTVLA_REPO/slurm/collect_shelf_restock_array.sbatch")
echo "collection job: $JOB_ID"

if [ "$FINALIZE" = "--finalize" ]; then
  sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --dependency="afterok:$JOB_ID" --job-name="octvla-finalize-$PROFILE" \
    --output="$OCTVLA_COLLECTION_ROOT/slurm_logs/%x-%j.out" \
    --error="$OCTVLA_COLLECTION_ROOT/slurm_logs/%x-%j.err" \
    --export="ALL,COLLECTION_PROFILE=$PROFILE" \
    "$OCTVLA_REPO/slurm/finalize_shelf_restock_dataset.sbatch"
fi
