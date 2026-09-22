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
#   rgb       stock ACT: images and proprioception            (stage 1)
#   entity    layerwise conditioning, geometry only           (stage 2)
#   semantic  the same, plus the variant code                 (stage 2)
#   scratch   entity conditioning with no stage 1             (w/o pretrain)
declare -A ARM_POLICY=([rgb]=act [entity]=control_act [semantic]=control_act [scratch]=control_act)
# The EE-absolute *view*, not the unified export it derives from: the unified
# dataset's canonical pair is absolute joint, and absolute EE is the only
# encoding that has produced a non-zero closed-loop result here.
DATASET="${FINAL_DATASET:-three_object_identity_eeabs}"
declare -A ARM_DATASET=(
  [rgb]="$DATASET" [entity]="$DATASET" [semantic]="$DATASET" [scratch]="$DATASET"
)

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
  cell="F-rgb-s${seed}"
  run="act_${ARM_DATASET[rgb]}_s${seed}"
  STAGE1_RUN[$seed]="$run"
  echo "$cell"
  STAGE1_JOB[$seed]=$(ACT_DATASET="${ARM_DATASET[rgb]}" ACT_POLICY_TYPE=act \
    TRAIN_SEED="$seed" N_ACTION_STEPS=25 \
    submit "${SB[@]}" --job-name="octvla-$cell" --time=04:00:00 \
      --output="$LOGS/$cell-%j.out" --export=ALL slurm/train_act_shelf_restock.sbatch)
  echo "    train  ${STAGE1_JOB[$seed]}  (stage 1)"
done

for arm in rgb entity semantic scratch; do
  for seed in $SEEDS; do
    cell="F-${arm}-s${seed}"
    run="${ARM_POLICY[$arm]}_${ARM_DATASET[$arm]}_s${seed}"
    [ "$arm" = semantic ] && run="${run}_semantic"
    [ "$arm" = scratch ] && run="${run}_scratch"
    if [ "$arm" = rgb ]; then
      train="${STAGE1_JOB[$seed]}"; run="${STAGE1_RUN[$seed]}"
    else
      echo "$cell"
      # `scratch` deliberately passes no ACT_PRETRAINED: that is the ablation.
      stage1="$OCTVLA_OUTPUT_ROOT/${STAGE1_RUN[$seed]}/checkpoints/last/pretrained_model"
      train=$(ACT_DATASET="${ARM_DATASET[$arm]}" ACT_POLICY_TYPE="${ARM_POLICY[$arm]}" \
        TRAIN_SEED="$seed" N_ACTION_STEPS=25 \
        OBJECT_SEMANTICS=$([ "$arm" = semantic ] && echo 1 || echo 0) \
        ACT_PRETRAINED=$([ "$arm" = scratch ] && echo "" || echo "$stage1") \
        submit "${SB[@]}" --job-name="octvla-$cell" --time=04:00:00 \
          --dependency=afterok:"${STAGE1_JOB[$seed]}" \
          --output="$LOGS/$cell-%j.out" --export=ALL slurm/train_act_shelf_restock.sbatch)
      echo "    train  $train  (stage 2 after ${STAGE1_JOB[$seed]})"
    fi
    for tier in seen heldout novel; do
      evalj=$(EVAL_CHECKPOINT="$OCTVLA_OUTPUT_ROOT/$run/checkpoints/last/pretrained_model" \
        EVAL_VARIANT=rgb EVAL_DATASET="${ARM_DATASET[$arm]}" EVAL_SEEDS=800-819 \
        EVAL_PROFILES=three_object EVAL_MAX_STEPS=600 EVAL_N_ACTION_STEPS=25 \
        EVAL_MODEL_IDS="${TIER[$tier]}" EVAL_TAG="${cell}-${tier}" \
        submit "${SB[@]}" --job-name="octvla-$cell-$tier" --time=04:00:00 \
          --dependency=afterok:"$train" --output="$LOGS/$cell-$tier-%j.out" \
          --export=ALL slurm/eval_shelf_restock.sbatch)
      echo "    eval   $tier -> $evalj"
    done
  done
done
echo
echo "12 training cells (3 stage-1, 9 stage-2/ablation), 36 rollouts."
echo "Collect with scripts/collect_final_results.py"
