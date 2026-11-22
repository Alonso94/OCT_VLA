#!/usr/bin/env bash
# BimanualPFN stage 1, prior training: one (arm, seed) per array element (slurm/submit_bpfn.sh).
# BPFN_BANK: draw items from a pre-generated bank (bpfn_bank.sbatch) instead of
# generating episodes on the fly; it is loaded into RAM once and shared by the workers.
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
srun "$OCTVLA_POLICY_PYTHON" -u "$OCTVLA_REPO/scripts/bpfn/stage1_train_prior.py" \
  --out "$RUN" --arm "$ARM" --seed "$SEED" --steps "${BPFN_STEPS:-80000}" \
  --batch "${BPFN_BATCH:-64}" --workers "$(( ${SLURM_CPUS_PER_TASK:-16} - 2 ))" \
  ${BPFN_REUSE:+--reuse "$BPFN_REUSE"} ${BPFN_PAIR_EXTRA:+--pair-extra "$BPFN_PAIR_EXTRA"} \
  ${BPFN_BANK:+--bank "$BPFN_BANK"} \
  ${BPFN_PER_SUPPORT:+--per-support "$BPFN_PER_SUPPORT"} ${BPFN_TF32:+--tf32} \
  ${BPFN_FRESH_SUPPORTS:+--fresh-supports} ${BPFN_EVAL_EVERY:+--eval-every "$BPFN_EVAL_EVERY"} \
  ${BPFN_LR:+--lr "$BPFN_LR"} ${BPFN_LR_FLOOR:+--lr-floor "$BPFN_LR_FLOOR"} ${BPFN_WARMUP:+--warmup "$BPFN_WARMUP"} \
  ${BPFN_WIDTH:+--width "$BPFN_WIDTH"} ${BPFN_LAYERS:+--layers "$BPFN_LAYERS"} ${BPFN_COMPILE:+--compile} \
  ${BPFN_ARCH:+--arch "$BPFN_ARCH"} ${BPFN_TASK_READ:+--task-read "$BPFN_TASK_READ"} \
  ${BPFN_OUTPUT:+--output "$BPFN_OUTPUT"} ${BPFN_TASK_TOKENS:+--task-tokens "$BPFN_TASK_TOKENS"} \
  --val "heldout_handover_colour=$BPFN_ROOT/packs/heldout_handover_colour.pkl" \
  --val "count_eval=$BPFN_ROOT/packs/count_eval.pkl" \
  ${BPFN_CONFLICT:+--conflict-fraction "$BPFN_CONFLICT"} \
  --dev "$BPFN_ROOT/packs/dev.pkl" --device cuda
