#!/usr/bin/env bash
# Submit data generation. A one-seed feasibility job is the safe first run.
# Example: ./slurm/submit_shelf_restock_collection.sh three_object a100 1000

set -euo pipefail

PROFILE="${1:?Usage: $0 two_object|three_object|four_object a40|a100 seeds}"
GPU="${2:?Missing GPU profile}" 
SEEDS="${3:?Missing Slurm array seed expression, e.g. 1000-1009}"

case "$PROFILE" in two_object|three_object|four_object) ;; *) echo "Invalid profile" >&2; exit 2;; esac
case "$GPU" in a40|a100) ;; *) echo "GPU must be a40 or a100" >&2; exit 2;; esac

: "${SLURM_ACCOUNT:?Set cluster account}"
: "${SLURM_PARTITION:?Set cluster partition}"
: "${OCTVLA_ROBOTWIN_ROOT:?Set RoboTwin checkout path}"
: "${OCTVLA_ROBOTWIN_PYTHON:?Set RoboTwin Python executable}"
: "${OCTVLA_COLLECTION_ROOT:?Set durable canonical collection root}"
: "${OCTVLA_DATASET_ROOT:?Set durable LeRobot dataset root}"
: "${OCTVLA_POLICY_PYTHON:?Set LeRobot Python executable}"
# Exported, not merely assigned: `sbatch --export=ALL` propagates the
# environment, and `${VAR:=default}` only creates a shell variable, so the
# job's own `${VAR:?}` assertion would fail.
export OCTVLA_REPO="${OCTVLA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
export DATASET_REPO_PREFIX="${DATASET_REPO_PREFIX:-local/oct-vla-shelf-restock-$PROFILE}"

mkdir -p "$OCTVLA_COLLECTION_ROOT/slurm_logs"
JOB_ID=$(sbatch --parsable \
  --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --gres="gpu:$GPU:1" \
  --array="$SEEDS" --job-name="octvla-collect-$PROFILE" \
  --output="$OCTVLA_COLLECTION_ROOT/slurm_logs/%x-%A_%a.out" \
  --error="$OCTVLA_COLLECTION_ROOT/slurm_logs/%x-%A_%a.err" \
  --export="ALL,COLLECTION_PROFILE=$PROFILE" \
  "$OCTVLA_REPO/slurm/collect_shelf_restock_array.sbatch")
echo "collection job: $JOB_ID"

echo "then build the dataset with slurm/build_shelf_restock_splits.sbatch (split-aware)"
