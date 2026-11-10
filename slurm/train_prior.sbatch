#!/bin/bash
# Train one arm of the interaction prior (PFN-VLA Step 1). CPU-heavy data generation in
# the loader workers, a small model on the GPU; no simulator.
#
#   sbatch $SB --export=ALL,RUN=P1-interaction-s0,ARGS="--head interaction --seed 0" slurm/train_prior.sbatch
#
# RUN names the output directory under $OCTVLA_PRIOR_ROOT/runs; ARGS go to prior_train.py
# (--lang and --eval-set default to the current language table and train eval set).
#SBATCH --job-name=octvla-prior
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
: "${RUN:?set RUN}"
source "$HOME/octvla/env.sh"
PRIOR_ROOT="${OCTVLA_PRIOR_ROOT:-$HPCVAULT/octvla-prior}"
cd "${OCTVLA_REPO}"
WORKERS=$(( ${SLURM_CPUS_ON_NODE:-8} - 2 ))
PYTHONPATH=src "$OCTVLA_POLICY_PYTHON" scripts/prior_train.py \
  --out "$PRIOR_ROOT/runs/$RUN" \
  --lang "${LANG_TABLE:-$PRIOR_ROOT/language/minilm.pt}" \
  --eval-set "${EVAL_SET:-$PRIOR_ROOT/eval/v1/train.npz}" \
  --workers "$WORKERS" ${ARGS:-}
