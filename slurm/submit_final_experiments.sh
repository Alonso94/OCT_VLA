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

# One arm per representation, on one dataset, differing only in what the policy
# reads. Three seeds each, because a single seed is what produced the 6/20
# result this project had to retract.
#   rgb       stock ACT: images and proprioception
#   entity    layerwise conditioning on geometry alone (schema v2 columns)
#   semantic  the same, plus the variant code (schema v3)
declare -A ARM_POLICY=([rgb]=act [entity]=control_act [semantic]=control_act)
declare -A ARM_DATASET=(
  [rgb]=three_object_identity_entity
  [entity]=three_object_identity_entity
  [semantic]=three_object_identity_entity
)

# The identity tiers a rollout is pinned to. Mixing them into one number is
# what makes an identity holdout meaningless.
declare -A TIER=(
  [seen]="1,2,3,4"       # trained on
  [heldout]="0,6"        # excluded from the dataset
  [novel]="5"            # in zero collected runs
)

submit() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "    would submit: $*" >&2; echo 0; else sbatch --parsable "$@"; fi; }

for arm in rgb entity semantic; do
  for seed in $SEEDS; do
    cell="F-${arm}-s${seed}"
    run="${ARM_POLICY[$arm]}_${ARM_DATASET[$arm]}_s${seed}"
    [ "$arm" = semantic ] && run="${run}_semantic"
    echo "$cell"
    train=$(ACT_DATASET="${ARM_DATASET[$arm]}" ACT_POLICY_TYPE="${ARM_POLICY[$arm]}" \
      TRAIN_SEED="$seed" N_ACTION_STEPS=25 \
      OBJECT_SEMANTICS=$([ "$arm" = semantic ] && echo 1 || echo 0) \
      submit "${SB[@]}" --job-name="octvla-$cell" --time=04:00:00 \
        --output="$LOGS/$cell-%j.out" --export=ALL slurm/train_act_shelf_restock.sbatch)
    echo "    train  $train"
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
echo "9 training cells, 27 rollouts. Collect with scripts/collect_final_results.py"
