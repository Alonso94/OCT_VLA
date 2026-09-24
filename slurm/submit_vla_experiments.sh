#!/usr/bin/env bash
# Stage B: the VLA matrix, in the phases the questions need, each queued with
# dependencies so it runs itself.
#
#   source ~/octvla/env-leftmost.sh
#   PHASE=stage1 slurm/submit_vla_experiments.sh             # Q4: RGB, both regimes
#   PHASE=stage2 REGIME_pi05=F REGIME_smolvla=J REGIME_groot=F \
#     ARMS="rgb_cont kv kv_adaln" slurm/submit_vla_experiments.sh   # Q1-Q3
#
# DRY_RUN=1 prints what would be submitted.
#
# Cells are named <P><R>-<arm>-s<seed>: P the backbone (P pi0.5, S SmolVLA,
# G GR00T), R the regime (F absolute EE, J absolute joint), arm as in the ACT
# matrix. scripts/collect_final_results.py --prefix PF.
#
# Storage is the binding constraint (1 TB of vault): every run keeps its final
# checkpoint only and drops its optimiser state (train_shelf_restock.sbatch);
# GR00T saves its trained head only (slim_groot); pi0.5/SmolVLA stage-1 LoRA
# adapters are merged for stage 2 only for the regime that goes on to it.
#
# Rollouts are ~0.8 s per step for a VLA, so ~2.7 GPU-h for one 20-scene
# three-object job and ~5.3 for the 2/4-object job. Stage 1 answers the regime
# question on seen identities alone; stage 2 adds tiers and object count only
# as far as ROLLOUTS asks (default: whatever the ACT matrix showed informative).
set -euo pipefail

: "${OCTVLA_REPO:?}"; : "${OCTVLA_DATASET_ROOT:?}"; : "${OCTVLA_OUTPUT_ROOT:?}"
PHASE="${PHASE:?Set PHASE=stage1 or stage2}"
BACKBONES="${BACKBONES:-pi05 smolvla groot}"
SEEDS="${SEEDS:-1000 1001 1002}"
TRAIN_STEPS="${TRAIN_STEPS:-8000}"   # the budget every earlier VLA cell used
BATCH="${BATCH:-16}"
LOGS="$OCTVLA_OUTPUT_ROOT/slurm_logs"
SB=(--account="${SLURM_ACCOUNT:-g107ea}" --partition="${SLURM_PARTITION:-a40}" --gres=gpu:a40:1)
mkdir -p "$LOGS"
VIDEO_STAGE="${VIDEO_STAGE:-$HPCVAULT/octvla-rollout-videos/final}"

declare -A LETTER=([pi05]=P [smolvla]=S [groot]=G)
# CORPUS picks the training data. `paired_full` (the default since 2026-09-24):
# continuous runs from the paired collection, the format that chains
# (research_questions.md §4.6). `identity`: the September atomic-clip corpus
# Stage A used. The tag keeps the two apart on disk -- a run of one must never
# be "reused" as the other's stage 1.
CORPUS="${CORPUS:-paired_full}"
case "$CORPUS" in
  paired_full)
    declare -A DATASET=([F]=three_object_paired_full_eeabs [J]=three_object_paired_full_abs)
    declare -A TAG=([F]=pf_eeabs [J]=pf_abs) ;;
  identity)
    # `id_`: `groot_rgb_full_s1000_eeabs` already exists, trained on the old
    # leftmost corpus, and a bare `eeabs` tag would have had stage 1 reuse it.
    declare -A DATASET=([F]=three_object_identity_eeabs [J]=three_object_identity_abs)
    declare -A TAG=([F]=id_eeabs [J]=id_abs) ;;
  *) echo "CORPUS must be paired_full or identity" >&2; exit 2 ;;
esac
for regime in F J; do
  [ -d "$OCTVLA_DATASET_ROOT/${DATASET[$regime]}" ] || { echo "no dataset ${DATASET[$regime]}" >&2; exit 2; }
done
# Must match TIER_IDS in scripts/collect_final_results.py.
declare -A TIER=([seen]="1,2,3,4" [heldout]="0,6" [novel]="5")

submit() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "    would submit: $*" >&2; echo 0; else sbatch --parsable "$@"; fi; }
after() { [ -n "$1" ] && [ "$1" != 0 ] && echo "--dependency=afterok:$1" || true; }
exists() { [ -d "$OCTVLA_OUTPUT_ROOT/$1/checkpoints/last/pretrained_model" ]; }

# stage1_run BACKBONE REGIME SEED -> the stage-1 run directory name
stage1_run() { echo "${1}_rgb_s${3}_${TAG[$2]}"; }

# rollout TAG CHECKPOINT DATASET PROFILES MODEL_IDS AFTER [VAR=VALUE...]
rollout() {
  local tag="$1" ckpt="$2" dataset="$3" profiles="$4" ids="$5" dep="$6"; shift 6
  [ "$#" -gt 0 ] && export "$@"
  EVAL_CHECKPOINT="$ckpt" EVAL_DATASET="$dataset" EVAL_SEEDS=800-819 \
  EVAL_PROFILES="$profiles" EVAL_N_ACTION_STEPS=25 EVAL_MODEL_IDS="$ids" EVAL_TAG="$tag" \
  EVAL_VIDEO_DIR="$VIDEO_STAGE/$tag" EVAL_VIDEO_LABEL="$tag" \
    submit "${SB[@]}" --job-name="octvla-$tag" --time=08:00:00 $(after "$dep") \
      --output="$LOGS/$tag-%j.out" --export=ALL slurm/eval_shelf_restock.sbatch
}

train() {  # train NAME AFTER HOURS VAR=VALUE... -> job id
  # A subshell, so the exported cell variables reach sbatch's --export=ALL and
  # never leak into the next cell. (`env VAR=... submit` cannot call a function.)
  local name="$1" dep="$2" hours="$3"; shift 3
  (
    export "$@" TRAIN_STEPS="$TRAIN_STEPS" BATCH_SIZE="$BATCH" N_ACTION_STEPS=25
    submit "${SB[@]}" --job-name="octvla-$name" --time="$hours:00:00" $(after "$dep") \
      --output="$LOGS/$name-%j.out" --export=ALL slurm/train_shelf_restock.sbatch
  )
}

jobs=0
if [ "$PHASE" = stage1 ]; then
  REGIMES="${REGIMES:-F J}"
  for backbone in $BACKBONES; do
    for regime in $REGIMES; do
      prefix="${LETTER[$backbone]}$regime"
      for seed in $SEEDS; do
        run=$(stage1_run "$backbone" "$regime" "$seed")
        echo "$prefix-rgb-s$seed  ($run)"
        if exists "$run"; then echo "    reuse (on disk)"; dep=""
        elif [ -e "$OCTVLA_OUTPUT_ROOT/$run" ]; then
          echo "    $run exists without a final checkpoint: resolve first" >&2; exit 2
        else
          dep=$(train "$prefix-rgb-s$seed" "" 24 BACKBONE="$backbone" TRAIN_VARIANT=rgb \
            TRAIN_DATASET="${DATASET[$regime]}" TRAIN_SEED="$seed" TRAIN_TAG="${TAG[$regime]}")
          echo "    train  $dep"; jobs=$((jobs + 1))
        fi
        j=$(rollout "$prefix-rgb-s$seed-seen" \
          "$OCTVLA_OUTPUT_ROOT/$run/checkpoints/last/pretrained_model" "${DATASET[$regime]}" \
          three_object "${TIER[seen]}" "$dep" EVAL_MAX_STEPS=600)
        echo "    eval   seen -> $j"; jobs=$((jobs + 1))
      done
    done
  done
elif [ "$PHASE" = stage2 ]; then
  ARMS="${ARMS:?Set ARMS, e.g. \"rgb_cont kv kv_adaln\"}"
  ROLLOUTS="${ROLLOUTS:-seen count}"
  for backbone in $BACKBONES; do
    var="REGIME_$backbone"; regime="${!var:?Set $var to F or J, the regime stage 1 favoured}"
    prefix="${LETTER[$backbone]}$regime"
    for seed in $SEEDS; do
      run1=$(stage1_run "$backbone" "$regime" "$seed")
      exists "$run1" || { echo "no stage-1 checkpoint for $run1" >&2; exit 2; }
      stage1="$OCTVLA_OUTPUT_ROOT/$run1/checkpoints/last/pretrained_model"
      merge_dep=""
      if [ "$backbone" != groot ]; then
        merged="$OCTVLA_OUTPUT_ROOT/$run1/merged"
        if [ ! -f "$merged/model.safetensors" ]; then
          merge_dep=$(submit "${SB[@]}" --job-name="octvla-$prefix-merge-s$seed" --time=02:00:00 \
            --output="$LOGS/$prefix-merge-s$seed-%j.out" --export=ALL --wrap="set -euo pipefail
cd $OCTVLA_REPO; export PYTHONPATH=src HF_HUB_OFFLINE=1
$OCTVLA_POLICY_PYTHON -u scripts/merge_stage1_adapter.py --stage1 $stage1 \
  --dataset-root $OCTVLA_DATASET_ROOT/${DATASET[$regime]} \
  --repo-id local/oct-vla-shelf-restock-${DATASET[$regime]#three_object_} --output $merged")
          echo "$prefix-merge-s$seed -> $merge_dep"; jobs=$((jobs + 1))
        fi
        stage1="$merged"
      fi
      for arm in $ARMS; do
        case "$arm" in
          rgb_cont)               arm_env=(TRAIN_VARIANT=rgb) ;;
          kv|kv_adaln|kv_tokens)  arm_env=(TRAIN_VARIANT=object CONDITIONING="$arm") ;;
          *) echo "Unknown arm $arm" >&2; exit 2 ;;
        esac
        cell="$prefix-$arm-s$seed"
        tag="${TAG[$regime]}_s2_${arm}"
        common=(BACKBONE="$backbone" TRAIN_DATASET="${DATASET[$regime]}" TRAIN_SEED="$seed"
                STAGE1_CHECKPOINT="$stage1" TRAIN_TAG="$tag" "${arm_env[@]}")
        # The init check gates the training: if stage 2 does not start at
        # stage 1, or a branch gets no gradient, nothing trains.
        check=$(train "$cell-check" "$merge_dep" 2 "${common[@]}" INIT_CHECK=1 TRAIN_TAG="${tag}_check")
        dep=$(train "$cell" "$check" 24 "${common[@]}")
        echo "$cell: check $check -> train $dep"; jobs=$((jobs + 2))
        # The name train_shelf_restock.sbatch will give the run.
        case "$arm" in
          rgb_cont) run2="${backbone}_rgb_s${seed}_${tag}" ;;
          *)        run2="${backbone}_${arm}_s${seed}_${tag}" ;;
        esac
        ckpt="$OCTVLA_OUTPUT_ROOT/$run2/checkpoints/last/pretrained_model"
        for r in $ROLLOUTS; do
          case "$r" in
            seen|heldout|novel)
              j=$(rollout "$cell-$r" "$ckpt" "${DATASET[$regime]}" three_object "${TIER[$r]}" "$dep" EVAL_MAX_STEPS=600) ;;
            count)
              j=$(rollout "$cell-count" "$ckpt" "${DATASET[$regime]}" two_object,four_object "${TIER[seen]}" "$dep" EVAL_STEPS_PER_OBJECT=200) ;;
            *) echo "Unknown rollout $r" >&2; exit 2 ;;
          esac
          echo "    eval   $r -> $j"; jobs=$((jobs + 1))
        done
      done
    done
  done
else
  echo "PHASE must be stage1 or stage2" >&2; exit 2
fi
echo; echo "$jobs jobs."
