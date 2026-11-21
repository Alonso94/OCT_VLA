#!/usr/bin/env bash
# BimanualPFN prior training: one (arm, seed) per array element (slurm/submit_bpfn.sh).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
set -euo pipefail
: "${OCTVLA_REPO:?}"; : "${OCTVLA_POLICY_PYTHON:?}"; : "${BPFN_ROOT:?}"
: "${BPFN_ARMS:?space-separated arms}"; : "${BPFN_SEEDS:?space-separated seeds}"
read -r -a ARMS <<< "$BPFN_ARMS"; read -r -a SEEDS <<< "$BPFN_SEEDS"
I="${SLURM_ARRAY_TASK_ID:?}"
ARM="${ARMS[$(( I / ${#SEEDS[@]} ))]}"; SEED="${SEEDS[$(( I % ${#SEEDS[@]} ))]}"
RUN="$BPFN_ROOT/${BPFN_RUNS_DIR:-runs}/${ARM}_s${SEED}${BPFN_RUN_SUFFIX:-}"
if [ -f "$RUN/DONE" ]; then echo "$RUN already complete; skipping"; exit 0; fi
if [ -e "$RUN" ]; then echo "$RUN exists without DONE: resolve by hand" >&2; exit 2; fi
export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
srun "$OCTVLA_POLICY_PYTHON" -u "$OCTVLA_REPO/scripts/bimanual_pfn.py" train \
  --out "$RUN" --arm "$ARM" --seed "$SEED" --steps "${BPFN_STEPS:-80000}" \
  --batch "${BPFN_BATCH:-64}" --workers "$(( ${SLURM_CPUS_PER_TASK:-16} - 2 ))" \
  --reuse "${BPFN_REUSE:-8}" ${BPFN_PAIR_EXTRA:+--pair-extra "$BPFN_PAIR_EXTRA"} \
  ${BPFN_CONFLICT:+--conflict-fraction "$BPFN_CONFLICT"} \
  --dev "$BPFN_ROOT/packs/dev.pkl" --device cuda
