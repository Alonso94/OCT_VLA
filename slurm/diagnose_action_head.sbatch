#!/usr/bin/env bash
# Offline action-prediction diagnostic for a trained checkpoint.
#
# Re-run after every change to the action encoding or the observation: this is
# the measurement that says whether a model learned its target at all, and it
# costs minutes against the hours a closed-loop sweep costs to return another
# zero.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

set -euo pipefail

: "${OCTVLA_REPO:?Set OCT_VLA checkout path}"
: "${OCTVLA_POLICY_PYTHON:?Set LeRobot Python executable}"
: "${OCTVLA_DATASET_ROOT:?Set durable LeRobot dataset root}"
: "${OCTVLA_OUTPUT_ROOT:?Set training output root}"
: "${DIAG_DATASET:?Set the dataset directory name, e.g. three_object_privileged}"

DATASET="$OCTVLA_DATASET_ROOT/$DIAG_DATASET"
[ -d "$DATASET" ] || { echo "No dataset at $DATASET" >&2; exit 2; }

export HF_HOME="${HF_HOME:?Set HF_HOME}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID"
if [ -n "${OCTVLA_TORCHCODEC_LIBS:-}" ]; then
  export LD_LIBRARY_PATH="$OCTVLA_TORCHCODEC_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

ARGS=(
  --dataset-root "$DATASET"
  --repo-id "local/oct-vla-shelf-restock-${DIAG_DATASET#three_object_}"
)
# Without a checkpoint the script still reports the reference predictors, which
# is a useful thing to ask of a dataset on its own.
if [ -n "${DIAG_CHECKPOINT:-}" ]; then
  ARGS+=(--checkpoint "$DIAG_CHECKPOINT")
fi
if [ -n "${DIAG_OUTPUT:-}" ]; then
  mkdir -p "$(dirname "$DIAG_OUTPUT")"
  ARGS+=(--output "$DIAG_OUTPUT")
fi

srun "$OCTVLA_POLICY_PYTHON" -u -B "$OCTVLA_REPO/scripts/diagnose_action_head.py" "${ARGS[@]}"
