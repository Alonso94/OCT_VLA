#!/usr/bin/env bash
# Stage 3 wiring check: each camera reaches the model slot it is named for.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
set -euo pipefail
: "${OCTVLA_REPO:?}"; : "${OCTVLA_POLICY_PYTHON:?}"; : "${BPFN_CHECKPOINT:?}"; : "${BPFN_VIEW:?}"; : "${BPFN_OUT:?}"
export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
[ -n "${OCTVLA_TORCHCODEC_LIBS:-}" ] && export LD_LIBRARY_PATH="$OCTVLA_TORCHCODEC_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
srun "$OCTVLA_POLICY_PYTHON" -u "$OCTVLA_REPO/scripts/bpfn/stage3_check_camera_routes.py" \
  --checkpoint "$BPFN_CHECKPOINT" --dataset-root "$BPFN_VIEW" --out "$BPFN_OUT"
