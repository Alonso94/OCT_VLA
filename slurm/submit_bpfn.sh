#!/usr/bin/env bash
# BimanualPFN, one step per paper stage. Every step prints its sbatch lines; DRY_RUN=1
# prints without submitting. Nothing here deletes or overwrites: train/eval skip runs
# with a DONE receipt / existing output, and collection writes only new seed dirs.
#
#   Stage 1, data and prior:
#   DRY_RUN=1 slurm/submit_bpfn.sh stage1-canary    # colour canary: 2 seeds x 2 rules + 1 swap
#   DRY_RUN=1 slurm/submit_bpfn.sh stage1-probe     # built-in coverage probes (10 seeds each)
#   DRY_RUN=1 slurm/submit_bpfn.sh stage1-collect   # wave 2, only after the canary passes
#   DRY_RUN=1 slurm/submit_bpfn.sh stage1-generate  # the training episode bank
#   DRY_RUN=1 slurm/submit_bpfn.sh stage1-train     # 5 arms x 3 seeds (BPFN_BANK: from the bank)
#   Stage 2, the frozen PFN alone:
#   DRY_RUN=1 slurm/submit_bpfn.sh stage2-evaluate  # synthetic packs, bound to train receipts
#   DRY_RUN=1 BPFN_RUN=... SUPPORT_RED=... SUPPORT_BLUE=... slurm/submit_bpfn.sh stage2-online
#   Stage 3, executors (after the stage-2 gate):
#   DRY_RUN=1 BPFN_PFN_RUN=... slurm/submit_bpfn.sh stage3-views          # budgets 4 / 8 / 16
#   DRY_RUN=1 SUPPORT_RED=... SUPPORT_BLUE=... slurm/submit_bpfn.sh stage3-act   # 3 budgets x 3 seeds
#   DRY_RUN=1 slurm/submit_bpfn.sh stage3-groot                           # budget 16, 3 seeds
#   DRY_RUN=1 BPFN_CHECKPOINT=... BPFN_VIEW=... slurm/submit_bpfn.sh stage3-camera-check
#
# Seeds of the new corpus (docs/data_protocol.md): colour source 2000-2199, colour
# dev 2200-2249, support banks 2250-2299, count queries 2300-2399. Evaluation scenes
# 800-819 and every v4 block stay untouched.
set -euo pipefail
STAGE="${1:?step: stage1-canary, -probe, -collect, -generate, -train; stage2-evaluate, -online; stage3-views, -act, -groot, -camera-check}"
source "${OCTVLA_ENV:-$HOME/octvla/env-v4.sh}"
export OCTVLA_REPO="${OCTVLA_REPO:-$HOME/OCT_VLA}"
export BPFN_ROOT="${BPFN_ROOT:-$HPCVAULT/octvla-bpfn}"
#: The episode bank is re-creatable (deterministic), so it lives on $WORK, not the vault.
BANK_DEFAULT="$WORK/octvla-scratch/bpfn-bank/v1"
COLLECT_ROOT="${BPFN_COLLECT_ROOT:-$HPCVAULT/octvla-collection-bpfn}"
#: Stage 3: executor views and runs, apart from the v4 roots (same filesystem: hard links).
BPFN_DATA="${BPFN_DATA:-$BPFN_ROOT/datasets}"
BPFN_EXEC="${BPFN_EXEC:-$BPFN_ROOT/executors}"
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
  stage1-canary)
    collect restock_red  four_object 2000-2001 "$COLLECT_ROOT/canary/restock_red"
    collect restock_blue four_object 2000-2001 "$COLLECT_ROOT/canary/restock_blue"
    collect restock_red  four_object 2000       "$COLLECT_ROOT/canary/restock_red_swap" 1
    collect restock_red  four_object 2002       "$COLLECT_ROOT/canary/restock_red_support" 0 3,1
    ;;
  stage1-probe)
    for task in handover_mic place_cans_plasticbox; do
      submit BUILTIN_TASK="$task" BUILTIN_COLLECTION_ROOT="$COLLECT_ROOT/builtin_probe" -- \
        "${SB[@]}" --array=0-9 --job-name="bpfn-probe-$task" \
        --output="$LOGS/probe-$task-%A_%a.out" --export=ALL slurm/collect_robotwin_builtin_array.sbatch
    done
    ;;
  stage1-generate)
    submit BPFN_BANK="${BPFN_BANK:-$BANK_DEFAULT}" -- \
      "${SB[@]}" --job-name=bpfn-bank --output="$LOGS/bank-%j.out" --export=ALL \
      slurm/bpfn_bank.sbatch
    ;;
  stage1-train)
    if [ -n "${BPFN_BANK:-}" ] && [ ! -f "$BPFN_BANK/bank.json" ] && [ -z "${AFTER_BANK:-}" ]; then
      echo "BPFN_BANK=$BPFN_BANK has no bank.json (build it: submit_bpfn.sh stage1-generate, or set AFTER_BANK)" >&2
      exit 2
    fi
    DEP=""; [ -n "${AFTER_BANK:-}" ] && DEP="--dependency=afterok:$AFTER_BANK"
    submit BPFN_ROOT="$BPFN_ROOT" BPFN_ARMS="$BPFN_ARMS" BPFN_SEEDS="$BPFN_SEEDS" -- \
      "${SB[@]}" --array="0-$((N - 1))${BPFN_CONCURRENCY:+%$BPFN_CONCURRENCY}" $DEP --job-name=bpfn-train \
      --output="$LOGS/train-%A_%a.out" --export=ALL slurm/bpfn_train.sbatch
    ;;
  stage2-evaluate)
    DEP=""; [ -n "${AFTER_TRAIN:-}" ] && DEP="--dependency=aftercorr:$AFTER_TRAIN"
    submit BPFN_ROOT="$BPFN_ROOT" BPFN_ARMS="$BPFN_ARMS" BPFN_SEEDS="$BPFN_SEEDS" -- \
      "${SB[@]}" --array="0-$((N - 1))${BPFN_CONCURRENCY:+%$BPFN_CONCURRENCY}" $DEP --job-name=bpfn-eval \
      --output="$LOGS/eval-%A_%a.out" --export=ALL slurm/bpfn_eval.sbatch
    ;;
  stage1-collect)
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
  stage2-online)
    # Stage 2b pilot on 10 evaluation resets (800-809), every condition on the same
    # resets and the same palette: bridge checks (expert windows, oracle / observable
    # gripper), then the frozen PFN with the correct and the swapped support. The paired
    # endpoint is the two 'same' cells (red support / red server, blue support / blue
    # server, identical resets); 'swapped' (support contradicts the server's rule) is the
    # control that must fail if success depends on the demonstration.
    : "${BPFN_RUN:?checkpoint run directory}"
    : "${SUPPORT_RED:?a restock_red support episode dir}"; : "${SUPPORT_BLUE:?a restock_blue one}"
    OUT="$BPFN_ROOT/online/$(basename "$BPFN_RUN")"
    for rule in restock_red restock_blue; do
      own="$SUPPORT_RED"; other="$SUPPORT_BLUE"
      [ "$rule" = restock_blue ] && { own="$SUPPORT_BLUE"; other="$SUPPORT_RED"; }
      for cond in "expert_window oracle -" "expert_window observable -" "pfn observable $own" \
                  "pfn observable $other"; do
        read -r ctrl grip sup <<< "$cond"
        tag="$rule-$ctrl-$grip"; [ "$sup" != - ] && tag="$tag-$([ "$sup" = "$own" ] && echo same || echo swapped)"
        submit BPFN_RULE="$rule" OCTVLA_COLOURS=red,blue BPFN_CONTROLLER="$ctrl" \
               BPFN_GRIPPER="$grip" BPFN_SEEDS=800-809 BPFN_RUN="$BPFN_RUN" \
               BPFN_SUPPORT="$([ "$sup" = - ] && echo || echo "$sup")" BPFN_TAG="$tag" \
               BPFN_OUT="$OUT/$tag.json" -- \
          "${SB[@]}" --job-name="bpfn-online-$tag" --output="$LOGS/online-$tag-%j.out" \
          --export=ALL slurm/bpfn_online.sbatch
      done
    done
    ;;
  stage3-views)
    # Stage 3 data, one job per budget: corpus -> eeabs -> pose-window and support banks.
    : "${BPFN_PFN_RUN:?the selected PFN run (stage 2 gate)}"
    for n in ${BPFN_BUDGETS:-4 8 16}; do
      submit BPFN_BUDGET="$n" BPFN_COLLECT_ROOT="$COLLECT_ROOT" BPFN_PFN_RUN="$BPFN_PFN_RUN" \
             OCTVLA_DATASET_ROOT="$BPFN_DATA" -- \
        "${SB[@]}" --job-name="bpfn-views-r$n" --output="$LOGS/views-r$n-%j.out" \
        --export=ALL slurm/bpfn_views.sbatch
    done
    ;;
  stage3-act)
    # Stage 3 ACT: per budget and seed, stage-1 RGB, then four conditioned arms from it
    # (scene-only kv; direct-demo support table; PFN window full actions; PFN window
    # residual), then the paired rollouts (support red / server red, support blue /
    # server blue, resets 800-809, one palette). AFTER_VIEWS=<jobid> chains to the views.
    : "${SUPPORT_RED:?a restock_red support run}"; : "${SUPPORT_BLUE:?a restock_blue support run}"
    mkdir -p "$BPFN_EXEC/run_metadata" 2>/dev/null || true
    for n in ${BPFN_BUDGETS:-4 8 16}; do
      base="bpfn_colour_r${n}_eeabs"
      for seed in ${BPFN_EXEC_SEEDS:-1000 1001 1002}; do
        dep=""; [ -n "${AFTER_VIEWS:-}" ] && dep="--dependency=afterok:$AFTER_VIEWS"
        common=(OCTVLA_DATASET_ROOT="$BPFN_DATA" OCTVLA_OUTPUT_ROOT="$BPFN_EXEC" TRAIN_SEED="$seed"
                TRAIN_STEPS="${ACT_STEPS:-40000}" CHUNK_SIZE=50 N_ACTION_STEPS=25)
        s1=$(submit "${common[@]}" ACT_DATASET="$base" ACT_POLICY_TYPE=act -- \
          "${SB[@]}" $dep --job-name="bpfn-act1-r$n-s$seed" \
          --output="$LOGS/act1-r$n-s$seed-%j.out" --export=ALL slurm/train_act_shelf_restock.sbatch)
        stage1="$BPFN_EXEC/act_${base}_s${seed}/checkpoints/last/pretrained_model"
        for arm in "scene_only $base 0" "direct_demo ${base}_st4 0" "pfn_full ${base}_pw4 0" \
                   "pfn_residual ${base}_pw4 1"; do
          read -r name view residual <<< "$arm"
          suffix=""; [ "$residual" = 1 ] && suffix=residual
          t=$(submit "${common[@]}" ACT_DATASET="$view" ACT_POLICY_TYPE=control_act CONDITIONING=kv \
                     ACT_PRETRAINED="$stage1" ACTION_RESIDUAL="$residual" RUN_SUFFIX="$suffix" -- \
            "${SB[@]}" --dependency=afterok:"$s1" --job-name="bpfn-act-$name-r$n-s$seed" \
            --output="$LOGS/act-$name-r$n-s$seed-%j.out" --export=ALL slurm/train_act_shelf_restock.sbatch)
          run="control_act_${view}_s${seed}"; [ -n "$suffix" ] && run="${run}_${suffix}"
          ckpt="$BPFN_EXEC/$run/checkpoints/last/pretrained_model"
          for rule in restock_red restock_blue; do
            sup="$SUPPORT_RED"; [ "$rule" = restock_blue ] && sup="$SUPPORT_BLUE"
            [ "$name" = scene_only ] && sup=""
            submit EVAL_CHECKPOINT="$ckpt" EVAL_DATASET="$view" OCTVLA_DATASET_ROOT="$BPFN_DATA" \
                   OCTVLA_OUTPUT_ROOT="$BPFN_EXEC" EVAL_SEEDS=800-809 EVAL_PROFILES=four_object \
                   EVAL_RULE="$rule" OCTVLA_COLOURS=red,blue EVAL_POSE_SUPPORT="$sup" \
                   EVAL_TAG="act-$name-r$n-s$seed-$rule" EVAL_MAX_STEPS=900 \
                   EVAL_TRACE_DIR="$BPFN_EXEC/traces/act-$name-r$n-s$seed-$rule" -- \
              "${SB[@]}" --dependency=afterok:"$t" --job-name="bpfn-eval-$name-r$n-s$seed-$rule" \
              --time=06:00:00 --output="$LOGS/eval-act-$name-r$n-s$seed-$rule-%j.out" \
              --export=ALL slurm/eval_shelf_restock.sbatch >/dev/null
          done
        done
      done
    done
    ;;
  stage3-groot)
    # Stage 3 GR00T at one budget (BPFN_GROOT_BUDGET, default 16), 3 seeds: stage 1 on the
    # eeabs base, then kv / kv_residual on the pose-window view and kv on the support view.
    n="${BPFN_GROOT_BUDGET:-16}"
    export OCTVLA_DATASET_ROOT="$BPFN_DATA" OCTVLA_OUTPUT_ROOT="$BPFN_EXEC" SLURM_ACCOUNT="$ACCOUNT"
    PHASE=stage1 CORPUS="bpfn_r$n" BACKBONES=groot REGIMES=F SEEDS="${BPFN_EXEC_SEEDS:-1000 1001 1002}" \
      ROLLOUTS=" " bash slurm/submit_vla_experiments.sh
    PHASE=stage2 CORPUS="bpfn_r${n}_pw" BACKBONES=groot REGIME_groot=F \
      SEEDS="${BPFN_EXEC_SEEDS:-1000 1001 1002}" ARMS="kv kv_residual" ROLLOUTS=" " \
      bash slurm/submit_vla_experiments.sh
    PHASE=stage2 CORPUS="bpfn_r${n}_st" BACKBONES=groot REGIME_groot=F \
      SEEDS="${BPFN_EXEC_SEEDS:-1000 1001 1002}" ARMS="kv" ROLLOUTS=" " \
      bash slurm/submit_vla_experiments.sh
    ;;
  stage3-camera-check)
    : "${BPFN_CHECKPOINT:?a trained executor checkpoint}"; : "${BPFN_VIEW:?its dataset view}"
    submit BPFN_CHECKPOINT="$BPFN_CHECKPOINT" BPFN_VIEW="$BPFN_VIEW" \
           BPFN_OUT="$BPFN_ROOT/camera_check/$(basename "$(dirname "$(dirname "$(dirname "$BPFN_CHECKPOINT")")")")" -- \
      "${SB[@]}" --job-name=bpfn-camera-check --output="$LOGS/camera-check-%j.out" \
      --export=ALL slurm/bpfn_camera_check.sbatch
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
