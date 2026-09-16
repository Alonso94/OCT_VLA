#!/usr/bin/env bash
# Acceptance gate for the evaluation controller: replay one successful recorded
# oracle transfer through the exact socket/IK/servo path used by policies.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

set -euo pipefail

: "${OCTVLA_REPO:?Set OCT_VLA checkout path}"
: "${OCTVLA_ROBOTWIN_ROOT:?Set RoboTwin checkout path}"
: "${OCTVLA_ROBOTWIN_PYTHON:?Set RoboTwin Python executable}"
: "${ORACLE_EPISODE:?Set a canonical transfer_index=0 episode directory}"

export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH

cd "$OCTVLA_ROBOTWIN_ROOT"
exec "$OCTVLA_ROBOTWIN_PYTHON" -u -B "$OCTVLA_REPO/scripts/replay_oracle_episode.py" \
  --robotwin-root "$OCTVLA_ROBOTWIN_ROOT" \
  --episode "$ORACLE_EPISODE"

