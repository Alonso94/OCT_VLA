#!/usr/bin/env bash
# Build one backbone end to end and push a real batch through it.
#
# Cheap by design: the failures it catches -- a missing processor module, a
# feature-name mismatch, LoRA attaching to nothing, an object residual wired to
# the wrong module -- all surface only after a full training run otherwise.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

set -uo pipefail

: "${OCTVLA_REPO:?Set OCT_VLA checkout path}"
: "${OCTVLA_POLICY_PYTHON:?Set LeRobot Python executable}"
: "${OCTVLA_DATASET_ROOT:?Set LeRobot dataset root}"
: "${SMOKE_POLICY_TYPE:?Set the registered policy type}"
: "${SMOKE_DATASET:?Set the dataset directory name}"

export HF_HOME="${HF_HOME:?Set HF_HOME}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID"
if [ -n "${OCTVLA_TORCHCODEC_LIBS:-}" ]; then
  export LD_LIBRARY_PATH="$OCTVLA_TORCHCODEC_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# Downloads a base checkpoint on first use, so online unless told otherwise.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

ARGS=(
  --policy-type "$SMOKE_POLICY_TYPE"
  --dataset-root "$OCTVLA_DATASET_ROOT/$SMOKE_DATASET"
  --repo-id "local/oct-vla-shelf-restock-${SMOKE_DATASET#three_object_}"
)
if [ -n "${SMOKE_PRETRAINED:-}" ]; then ARGS+=(--pretrained "$SMOKE_PRETRAINED"); fi
if [ "${SMOKE_PEFT:-0}" = "1" ]; then ARGS+=(--peft); fi
if [ -n "${SMOKE_INJECTION_MODE:-}" ]; then ARGS+=(--injection-mode "$SMOKE_INJECTION_MODE"); fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun "$OCTVLA_POLICY_PYTHON" -u -B "$OCTVLA_REPO/scripts/smoke_test_backbone.py" "${ARGS[@]}"
