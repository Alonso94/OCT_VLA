#!/usr/bin/env bash
# The final ACT matrix: three representations x three seeds, each evaluated on
# three identity tiers. Submits everything with dependencies so the whole thing
# can be queued now and will run itself when the cluster returns.
#
#   source ~/octvla/env-leftmost.sh && slurm/submit_final_experiments.sh
#
# DRY_RUN=1 prints what would be submitted and exits.
set -euo pipefail

: "${OCTVLA_REPO:?}"; : "${OCTVLA_DATASET_ROOT:?}"; : "${OCTVLA_OUTPUT_ROOT:?}"
SEEDS="${SEEDS:-1000 1001 1002}"
LOGS="$OCTVLA_OUTPUT_ROOT/slurm_logs"
SB=(--account="${SLURM_ACCOUNT:-g107ea}" --partition="${SLURM_PARTITION:-a40}" --gres=gpu:a40:1)
mkdir -p "$LOGS"

# ControlVLA is a *two-stage* recipe and we were about to run only its failing
# ablation. The paper pre-trains a general policy and adds object conditioning
# afterwards; its "w/o pretrain" arm -- conditioning a from-scratch policy --
# reports severe jitter and large drops, hypothesised to be object features
# offering a low-loss shortcut before stable visual features are learned.
# Training control_act from scratch is exactly that, and our one conditioned
# cell (2/20 against an RGB 6/20) points the same way.
#
# So `rgb` is stage one and the conditioned arms fine-tune from its checkpoint,
# per seed. `scratch` keeps the ablation as one deliberate cell rather than an
# accident.
#   rgb        stock ACT: images and proprioception            (stage 1)
#   entity     layerwise conditioning, geometry only           (stage 2)
#   adaln      entity + AdaLN-Zero on every encoder/decoder block (stage 2)
#   incontext  entity + entity tokens in the encoder sequence  (stage 2)
#   scratch    entity conditioning with no stage 1             (w/o pretrain)
#
# adaln and incontext follow LPWM (arXiv:2603.04553). ACT has one decoder
# layer, so the layerwise branch alone is a single seam and the encoder sees
# no objects; each of these adds one encoder-side path and nothing else, so
# against `entity` it isolates exactly that addition.
#
# There is no `semantic` arm. There used to be, and it set a variable nothing
# read, so it trained the entity model under another name. The datasets are v2
# with no variant code; the arm returns with a v3 export and embedding.
#
# ARMS selects a subset, e.g. ARMS="adaln incontext" to add two arms to a
# matrix whose stage 1 already exists. A stage-1 checkpoint already on disk is
# reused instead of retrained. SKIP_EVAL=1 submits training only, for a task
# with no rollout harness yet. PREFIX names the cells (default F) so two tasks'
# matrices cannot collide in job names or evaluation tags.
ARMS="${ARMS:-rgb entity adaln incontext scratch}"
PREFIX="${PREFIX:-F}"
declare -A ARM_POLICY=(
  [rgb]=act [entity]=control_act [adaln]=control_act [incontext]=control_act [scratch]=control_act
)
declare -A ARM_FLAGS=(
  [rgb]="" [entity]="" [adaln]="OBJECT_ADALN=1" [incontext]="OBJECT_INCONTEXT=1" [scratch]=""
)
# The EE-absolute *view*, not the unified export it derives from: the unified
# dataset's canonical pair is absolute joint, and absolute EE is the only
# encoding that has produced a non-zero closed-loop result here.
DATASET="${FINAL_DATASET:-three_object_identity_eeabs}"

# Two arms that would train the same model are refused: that is exactly how
# the semantic arm went unnoticed.
declare -A SIGNATURE
for arm in $ARMS; do
  [ -n "${ARM_POLICY[$arm]:-}" ] || { echo "Unknown arm: $arm" >&2; exit 2; }
  stage=$([ "$arm" = scratch ] && echo scratch || echo finetune)
  sig="${ARM_POLICY[$arm]}|$stage|${ARM_FLAGS[$arm]}"
  [ "$arm" = rgb ] && sig="act"
  if [ -n "${SIGNATURE[$sig]:-}" ]; then
    echo "Arms ${SIGNATURE[$sig]} and $arm would train the same model" >&2; exit 2
  fi
  SIGNATURE[$sig]="$arm"
done

# The identity tiers a rollout is pinned to. Mixing them into one number is
# what makes an identity holdout meaningless.
declare -A TIER=(
  [seen]="1,2,3,4"       # trained on
  [heldout]="0,6"        # excluded from the dataset
  [novel]="5"            # in zero collected runs
)

submit() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "    would submit: $*" >&2; echo 0; else sbatch --parsable "$@"; fi; }

# Stage one first: the conditioned arms cannot start until their own seed's rgb
# checkpoint exists, so each depends on it rather than on a wall-clock guess.
declare -A STAGE1_JOB STAGE1_RUN
for seed in $SEEDS; do
  cell="${PREFIX}-rgb-s${seed}"
  run="act_${DATASET}_s${seed}"
  STAGE1_RUN[$seed]="$run"
  echo "$cell"
  if [ -d "$OCTVLA_OUTPUT_ROOT/$run/checkpoints/last/pretrained_model" ]; then
    STAGE1_JOB[$seed]=""
    echo "    reuse  $run  (stage 1 on disk)"
  elif [ -e "$OCTVLA_OUTPUT_ROOT/$run" ]; then
    echo "    $run exists without a final checkpoint: running or failed, resolve first" >&2
    exit 2
  else
    STAGE1_JOB[$seed]=$(ACT_DATASET="$DATASET" ACT_POLICY_TYPE=act \
      TRAIN_SEED="$seed" N_ACTION_STEPS=25 \
      submit "${SB[@]}" --job-name="octvla-$cell" --time=04:00:00 \
        --output="$LOGS/$cell-%j.out" --export=ALL slurm/train_act_shelf_restock.sbatch)
    echo "    train  ${STAGE1_JOB[$seed]}  (stage 1)"
  fi
done

after() { [ -n "$1" ] && echo "--dependency=afterok:$1" || true; }
cells=0; rollouts=0
for arm in $ARMS; do
  for seed in $SEEDS; do
    cell="${PREFIX}-${arm}-s${seed}"
    run="${ARM_POLICY[$arm]}_${DATASET}_s${seed}"
    case "$arm" in
      scratch) run="${run}_scratch" ;;
      adaln|incontext) run="${run}_${arm}" ;;
    esac
    if [ "$arm" = rgb ]; then
      train="${STAGE1_JOB[$seed]}"; run="${STAGE1_RUN[$seed]}"
      [ -n "$train" ] && cells=$((cells + 1))
    else
      echo "$cell"
      stage1="$OCTVLA_OUTPUT_ROOT/${STAGE1_RUN[$seed]}/checkpoints/last/pretrained_model"
      train=$(OBJECT_ADALN=$([ "$arm" = adaln ] && echo 1 || echo 0) \
        OBJECT_INCONTEXT=$([ "$arm" = incontext ] && echo 1 || echo 0) \
        ACT_DATASET="$DATASET" ACT_POLICY_TYPE="${ARM_POLICY[$arm]}" \
        TRAIN_SEED="$seed" N_ACTION_STEPS=25 \
        RUN_SUFFIX=$(case "$arm" in adaln|incontext) echo "$arm" ;; esac) \
        ACT_PRETRAINED=$([ "$arm" = scratch ] && echo "" || echo "$stage1") \
        submit "${SB[@]}" --job-name="octvla-$cell" --time=04:00:00 \
          $([ "$arm" = scratch ] || after "${STAGE1_JOB[$seed]}") \
          --output="$LOGS/$cell-%j.out" --export=ALL slurm/train_act_shelf_restock.sbatch)
      cells=$((cells + 1))
      echo "    train  $train  (${ARM_FLAGS[$arm]:-stage 2}${STAGE1_JOB[$seed]:+ after ${STAGE1_JOB[$seed]}})"
    fi
    [ "${SKIP_EVAL:-0}" = 1 ] && continue
    for tier in seen heldout novel; do
      evalj=$(EVAL_CHECKPOINT="$OCTVLA_OUTPUT_ROOT/$run/checkpoints/last/pretrained_model" \
        EVAL_VARIANT=rgb EVAL_DATASET="$DATASET" EVAL_SEEDS=800-819 \
        EVAL_PROFILES=three_object EVAL_MAX_STEPS=600 EVAL_N_ACTION_STEPS=25 \
        EVAL_MODEL_IDS="${TIER[$tier]}" EVAL_TAG="${cell}-${tier}" \
        submit "${SB[@]}" --job-name="octvla-$cell-$tier" --time=04:00:00 \
          $(after "$train") --output="$LOGS/$cell-$tier-%j.out" \
          --export=ALL slurm/eval_shelf_restock.sbatch)
      rollouts=$((rollouts + 1))
      echo "    eval   $tier -> $evalj"
    done
  done
done
echo
echo "$cells training cells, $rollouts rollouts."
echo "Collect with scripts/collect_final_results.py"
