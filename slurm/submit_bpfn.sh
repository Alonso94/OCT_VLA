#!/usr/bin/env bash
# BimanualPFN stages (plan: stage 1). Every stage prints its sbatch lines; DRY_RUN=1
# prints without submitting. Nothing here deletes or overwrites: train/eval skip runs
# with a DONE receipt / existing output, and collection writes only new seed dirs.
#
#   DRY_RUN=1 slurm/submit_bpfn.sh canary     # colour canary: 2 seeds x 2 rules + 1 swap
#   DRY_RUN=1 slurm/submit_bpfn.sh probe      # built-in coverage probes (10 seeds each)
#   DRY_RUN=1 slurm/submit_bpfn.sh train      # 5 arms x 3 seeds, bounded concurrency
#   DRY_RUN=1 slurm/submit_bpfn.sh eval       # synthetic packs, bound to train receipts
#   DRY_RUN=1 slurm/submit_bpfn.sh pilot      # wave 2, only after the canary passes
#
# Seeds of the new corpus (docs/data_protocol.md): colour source 2000-2199, colour
# dev 2200-2249, support banks 2250-2299, count queries 2300-2399. Evaluation scenes
# 800-819 and every v4 block stay untouched.
set -euo pipefail
STAGE="${1:?stage: canary | probe | train | eval | pilot}"
source "${OCTVLA_ENV:-$HOME/octvla/env-v4.sh}"
export OCTVLA_REPO="${OCTVLA_REPO:-$HOME/OCT_VLA}"
export BPFN_ROOT="${BPFN_ROOT:-$HPCVAULT/octvla-bpfn}"
COLLECT_ROOT="${BPFN_COLLECT_ROOT:-$HPCVAULT/octvla-collection-bpfn}"
ACCOUNT="${SLURM_ACCOUNT:-g101ea}"
SB=(--account="$ACCOUNT" --partition=a40 --gres=gpu:a40:1 --exclude=a0423,a0425)
LOGS="$BPFN_ROOT/slurm_logs"
export BPFN_ARMS="${BPFN_ARMS:-full equal_count scene_only no_handover nominal_only}"
export BPFN_SEEDS="${BPFN_SEEDS:-0 1 2}"
read -r -a _A <<< "$BPFN_ARMS"; read -r -a _S <<< "$BPFN_SEEDS"
N=$(( ${#_A[@]} * ${#_S[@]} ))

submit() {  # submit ENV... -- SBATCH_ARGS...  -> prints the job id (0 in a dry run)
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "would submit: env ${envs[*]} sbatch $*" >&2; echo 0; return
  fi
  mkdir -p "$LOGS"
  env "${envs[@]}" sbatch --parsable "$@"
}

collect() {  # collect RULE PROFILE SEEDS ROOT [SWAP] [COUNTS]
  # COUNTS: objects per colour, the rule's colour first (3,1 = a colour support with
  # three eligible and one distractor); empty = balanced (the source pool).
  local rule="$1" profile="$2" seeds="$3" root="$4" swap="${5:-0}" counts="${6:-}" colours=""
  case "$rule" in restock_red) colours="red,blue" ;; restock_blue) colours="blue,red" ;; esac
  submit OCTVLA_COLLECTION_ROOT="$root" COLLECTION_PROFILE="$profile" EPISODE_KIND=full_run \
         RULE="$rule" OCTVLA_COLOURS="$colours" OCTVLA_COLOUR_SWAP="$swap" \
         OCTVLA_COLOUR_COUNTS="$counts" -- \
    "${SB[@]}" --array="$seeds" --job-name="bpfn-collect-$rule" \
    --output="$LOGS/collect-$rule-%A_%a.out" --export=ALL slurm/collect_shelf_restock_array.sbatch
}

case "$STAGE" in
  canary)
    collect restock_red  four_object 2000-2001 "$COLLECT_ROOT/canary/restock_red"
    collect restock_blue four_object 2000-2001 "$COLLECT_ROOT/canary/restock_blue"
    collect restock_red  four_object 2000       "$COLLECT_ROOT/canary/restock_red_swap" 1
    collect restock_red  four_object 2002       "$COLLECT_ROOT/canary/restock_red_support" 0 3,1
    ;;
  probe)
    for task in handover_mic place_cans_plasticbox; do
      submit BUILTIN_TASK="$task" BUILTIN_COLLECTION_ROOT="$COLLECT_ROOT/builtin_probe" -- \
        "${SB[@]}" --array=0-9 --job-name="bpfn-probe-$task" \
        --output="$LOGS/probe-$task-%A_%a.out" --export=ALL slurm/collect_robotwin_builtin_array.sbatch
    done
    ;;
  train)
    submit BPFN_ROOT="$BPFN_ROOT" BPFN_ARMS="$BPFN_ARMS" BPFN_SEEDS="$BPFN_SEEDS" -- \
      "${SB[@]}" --array="0-$((N - 1))%${BPFN_CONCURRENCY:-4}" --job-name=bpfn-train \
      --output="$LOGS/train-%A_%a.out" --export=ALL slurm/bpfn_train.sbatch
    ;;
  eval)
    DEP=""; [ -n "${AFTER_TRAIN:-}" ] && DEP="--dependency=aftercorr:$AFTER_TRAIN"
    submit BPFN_ROOT="$BPFN_ROOT" BPFN_ARMS="$BPFN_ARMS" BPFN_SEEDS="$BPFN_SEEDS" -- \
      "${SB[@]}" --array="0-$((N - 1))%${BPFN_CONCURRENCY:-4}" $DEP --job-name=bpfn-eval \
      --output="$LOGS/eval-%A_%a.out" --export=ALL slurm/bpfn_eval.sbatch
    ;;
  pilot)
    # Wave 2 (plan 1c): only after the canary report passes.
    [ -f "$COLLECT_ROOT/canary/CANARY_PASSED" ] || [ "${DRY_RUN:-0}" = 1 ] || {
      echo "canary has not passed ($COLLECT_ROOT/canary/CANARY_PASSED missing)" >&2; exit 2; }
    collect restock_red  four_object 2000-2069 "$COLLECT_ROOT/colour/restock_red"
    collect restock_blue four_object 2000-2069 "$COLLECT_ROOT/colour/restock_blue"
    collect restock_red  four_object 2250-2257 "$COLLECT_ROOT/supports/restock_red" 0 3,1
    collect restock_blue four_object 2250-2257 "$COLLECT_ROOT/supports/restock_blue" 0 3,1
    collect leftmost     three_object 2260-2267 "$COLLECT_ROOT/supports/count"
    collect leftmost     two_object   2300-2319 "$COLLECT_ROOT/count_queries/leftmost"
    collect leftmost     four_object  2300-2319 "$COLLECT_ROOT/count_queries/leftmost"
    submit BUILTIN_TASK=handover_mic BUILTIN_COLLECTION_ROOT="$COLLECT_ROOT/builtin" -- \
      "${SB[@]}" --array=2000-2029 --job-name=bpfn-handover \
      --output="$LOGS/handover-%A_%a.out" --export=ALL slurm/collect_robotwin_builtin_array.sbatch
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
