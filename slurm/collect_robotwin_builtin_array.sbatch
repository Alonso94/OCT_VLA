#!/usr/bin/env bash
# One seed per array task. The array index is the scene seed, so a rerun of the
# same index reproduces the same scene exactly.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

set -euo pipefail

: "${OCTVLA_REPO:?}"; : "${OCTVLA_ROBOTWIN_ROOT:?}"; : "${OCTVLA_ROBOTWIN_PYTHON:?}"
: "${BUILTIN_COLLECTION_ROOT:?Set the collection root for this task}"
TASK="${BUILTIN_TASK:-place_container_plate}"
SEED="${SLURM_ARRAY_TASK_ID:?Slurm array index is required}"

export PYTHONUNBUFFERED=1
# No OCTVLA_TORCHCODEC_LIBS here, unlike the training and export jobs.
# Torchcodec belongs to the *policy* environment; prepending its libraries to
# LD_LIBRARY_PATH shadows the copies SAPIEN bundles, and every seed then dies on
# `ImportError: libXau-...so: cannot open shared object file` while importing
# sapien.core -- which surfaces as "Could not load RoboTwin task", naming
# neither the library nor the cause. The shelf-restock collector sets neither
# variable for the same reason.

srun "$OCTVLA_ROBOTWIN_PYTHON" -u -B "$OCTVLA_REPO/scripts/collect_robotwin_builtin.py" \
  --robotwin-root "$OCTVLA_ROBOTWIN_ROOT" --task "$TASK" --seed "$SEED" \
  --output "$BUILTIN_COLLECTION_ROOT"
