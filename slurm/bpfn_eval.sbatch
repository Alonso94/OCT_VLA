#!/usr/bin/env bash
# BimanualPFN evaluation of one trained run on every synthetic pack (array element = run).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
set -euo pipefail
: "${OCTVLA_REPO:?}"; : "${OCTVLA_POLICY_PYTHON:?}"; : "${BPFN_ROOT:?}"
: "${BPFN_ARMS:?}"; : "${BPFN_SEEDS:?}"
read -r -a ARMS <<< "$BPFN_ARMS"; read -r -a SEEDS <<< "$BPFN_SEEDS"
I="${SLURM_ARRAY_TASK_ID:?}"
ARM="${ARMS[$(( I / ${#SEEDS[@]} ))]}"; SEED="${SEEDS[$(( I % ${#SEEDS[@]} ))]}"
RUN="$BPFN_ROOT/runs/${ARM}_s${SEED}"
[ -f "$RUN/DONE" ] || { echo "$RUN has no DONE receipt; train first" >&2; exit 2; }
export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
for PACK in ${BPFN_PACKS:-dev heldout_handover_colour count_eval}; do
  OUT="$RUN/eval_${PACK}.json"
  [ -f "$OUT" ] && { echo "$OUT exists; skipping"; continue; }
  srun "$OCTVLA_POLICY_PYTHON" -u "$OCTVLA_REPO/scripts/bimanual_pfn.py" eval \
    --run "$RUN" --pack "$BPFN_ROOT/packs/$PACK.pkl" --out "$OUT" --device cuda
done
