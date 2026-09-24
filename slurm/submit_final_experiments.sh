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

# The arms (ControlVLA's two-stage recipe: train RGB first, then add objects
# from that checkpoint, with its "w/o pretrain" ablation kept as one cell):
#
#   rgb         stock ACT, images + proprioception             stage 1
#   rgb_cont    rgb continued for the steps stage 2 adds       budget control
#   kv          + ControlVLA's KV term                         stage 2
#   kv_adaln    + KV and scene AdaLN on every block            stage 2
#   kv_tokens   + KV and entity tokens in the encoder          stage 2
#   scratch_kv  kv trained with no stage 1                     w/o pretrain
#   rgb_novae   stock ACT with the CVAE latent off (use_vae=false), as rgb
#               otherwise: 40k steps from scratch. Its KL collapses to 0 anyway
#   rgb_short   stock ACT with a short chunk (20) executed 8 at a time
#   rgb_hist    rgb_short plus a two-frame observation history (history_act),
#               so against rgb_short it isolates the history
#
# rgb_cont is the fair baseline: every stage-2 arm trains 40k steps on top of
# rgb's 40k, so against rgb alone a gain could be the extra training.
#
# kv_adaln and kv_tokens are LPWM-inspired (arXiv:2603.04553). LeRobot ACT has
# one decoder layer, so KV alone is a single seam and the encoder never sees
# the objects; each of these adds one encoder-side path to kv, so against kv
# it isolates exactly that addition.
#
# Run directories keep their pre-rename suffixes (_adaln, _incontext,
# _scratch), which is how the checkpoints already on disk are found.
#
# ARMS selects a subset. Any cell whose checkpoint is on disk is reused, so
# re-running this re-queues only rollouts. SKIP_EVAL=1 submits training only.
# PREFIX names the cells (default F) so two matrices cannot collide.
ARMS="${ARMS:-rgb rgb_cont kv kv_adaln kv_tokens scratch_kv}"
PREFIX="${PREFIX:-F}"
JOBS="${JOBS:-seen heldout novel count}"
declare -A ARM_POLICY=(
  [rgb]=act [rgb_cont]=act [rgb_novae]=act [rgb_short]=act [rgb_hist]=history_act
  [kv]=control_act [kv_adaln]=control_act [kv_tokens]=control_act [scratch_kv]=control_act
)
declare -A ARM_CONDITIONING=([kv]=kv [kv_adaln]=kv_adaln [kv_tokens]=kv_tokens [scratch_kv]=kv)
# Directory suffix per arm, as named before the rename.
declare -A ARM_SUFFIX=([rgb]="" [rgb_cont]=_cont [rgb_novae]=_novae [rgb_short]=_short [rgb_hist]="" [kv]="" [kv_adaln]=_adaln [kv_tokens]=_incontext [scratch_kv]=_scratch)
# Chunk and execution horizon per arm; 50 and 25 otherwise. The rollout uses the
# horizon the arm trained with, so the short-chunk arms are not scored at 25.
declare -A ARM_CHUNK=([rgb_short]=20 [rgb_hist]=20)
declare -A ARM_EXEC=([rgb_short]=8 [rgb_hist]=8)
# The EE-absolute *view*, not the unified export it derives from: the unified
# dataset's canonical pair is absolute joint, and absolute EE is the only
# encoding that has produced a non-zero closed-loop result here.
DATASET="${FINAL_DATASET:-three_object_identity_eeabs}"

for arm in $ARMS; do
  [ -n "${ARM_POLICY[$arm]:-}" ] || { echo "Unknown arm: $arm (have: ${!ARM_POLICY[*]})" >&2; exit 2; }
done

# The identity tiers a rollout is pinned to. Mixing them into one number is
# what makes an identity holdout meaningless.
# Must match TIER_IDS in scripts/collect_final_results.py, which refuses a
# rollout whose pin disagrees with its name.
declare -A TIER=(
  [seen]="1,2,3,4"       # trained on
  [heldout]="0,6"        # excluded from the dataset
  [novel]="5"            # in zero collected runs
)
# Compositional generalisation: the same checkpoint on two and four objects,
# seen identities only, so a change is attributable to count and not identity.
# A fixed budget per object (3 x 200 = the 600 the tier rollouts get) rather
# than 600 for every count, which would hand two objects half again the time
# per transfer and four objects two thirds of it.
STEPS_PER_OBJECT=200

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
    # AFTER_JOB: a job stage one must wait for, e.g. the one projecting the
    # dataset view it trains on.
    STAGE1_JOB[$seed]=$(ACT_DATASET="$DATASET" ACT_POLICY_TYPE=act \
      TRAIN_SEED="$seed" N_ACTION_STEPS=25 \
      submit "${SB[@]}" --job-name="octvla-$cell" --time=04:00:00 \
        ${AFTER_JOB:+--dependency=afterok:$AFTER_JOB} \
        --output="$LOGS/$cell-%j.out" --export=ALL slurm/train_act_shelf_restock.sbatch)
    echo "    train  ${STAGE1_JOB[$seed]}  (stage 1)"
  fi
done

after() { [ -n "$1" ] && echo "--dependency=afterok:$1" || true; }
cells=0; rollouts=0

# Every episode is recorded, to a staging area rather than the repository:
# which episode is *representative* is only known once all seeds are in, and
# the rollouts are deterministic, so recording now saves a second pass.
# scripts/select_rollout_videos.py picks one per reported row into docs/.
VIDEO_STAGE="${VIDEO_STAGE:-$HPCVAULT/octvla-rollout-videos/final}"

# rollout TAG PROFILES MODEL_IDS [VAR=VALUE...] -- one pinned evaluation of
# $run after $train, 20 scenes per profile. Prints the job id. Runs in the
# caller's $(...), so the variables it exports never leak into the next cell.
rollout() {
  local tag="$1" profiles="$2" ids="$3"; shift 3
  [ "$#" -gt 0 ] && export "$@"
  EVAL_CHECKPOINT="$OCTVLA_OUTPUT_ROOT/$run/checkpoints/last/pretrained_model" \
  EVAL_DATASET="$DATASET" EVAL_SEEDS=800-819 \
  EVAL_PROFILES="$profiles" EVAL_N_ACTION_STEPS="${ARM_EXEC[$arm]:-25}" \
  EVAL_MODEL_IDS="$ids" EVAL_TAG="$tag" \
  EVAL_VIDEO_DIR="$VIDEO_STAGE/$tag" EVAL_VIDEO_LABEL="$tag" \
    submit "${SB[@]}" --job-name="octvla-$tag" --time=04:00:00 \
      $(after "$train") --output="$LOGS/$tag-%j.out" \
      --export=ALL slurm/eval_shelf_restock.sbatch
}
for arm in $ARMS; do
  for seed in $SEEDS; do
    cell="${PREFIX}-${arm}-s${seed}"
    run="${ARM_POLICY[$arm]}_${DATASET}_s${seed}${ARM_SUFFIX[$arm]}"
    if [ "$arm" = rgb ]; then
      train="${STAGE1_JOB[$seed]}"; run="${STAGE1_RUN[$seed]}"
      [ -n "$train" ] && cells=$((cells + 1))
    elif [ -d "$OCTVLA_OUTPUT_ROOT/$run/checkpoints/last/pretrained_model" ]; then
      echo "$cell"
      echo "    reuse  $run  (on disk)"
      train=""
    elif [ -e "$OCTVLA_OUTPUT_ROOT/$run" ]; then
      echo "    $run exists without a final checkpoint: running or failed, resolve first" >&2
      exit 2
    else
      echo "$cell"
      stage1="$OCTVLA_OUTPUT_ROOT/${STAGE1_RUN[$seed]}/checkpoints/last/pretrained_model"
      # The sbatch derives _scratch itself from an unset ACT_PRETRAINED, so the
      # scratch arm passes no suffix; the others pass theirs without the "_".
      suffix="${ARM_SUFFIX[$arm]#_}"; [ "$arm" = scratch_kv ] && suffix=""
      # Arms trained from scratch (no stage-1 checkpoint, no dependency on it).
      from_scratch=$(case "$arm" in scratch_kv|rgb_novae|rgb_short|rgb_hist) echo 1 ;; *) echo 0 ;; esac)
      train=$(CONDITIONING="${ARM_CONDITIONING[$arm]:-}" \
        ACT_DATASET="$DATASET" ACT_POLICY_TYPE="${ARM_POLICY[$arm]}" \
        TRAIN_SEED="$seed" N_ACTION_STEPS="${ARM_EXEC[$arm]:-25}" RUN_SUFFIX="$suffix" \
        CHUNK_SIZE="${ARM_CHUNK[$arm]:-50}" \
        USE_VAE=$([ "$arm" = rgb_novae ] && echo 0 || echo 1) \
        ACT_PRETRAINED=$([ "$from_scratch" = 1 ] && echo "" || echo "$stage1") \
        submit "${SB[@]}" --job-name="octvla-$cell" --time=04:00:00 \
          $([ "$from_scratch" = 1 ] || after "${STAGE1_JOB[$seed]}") \
          --output="$LOGS/$cell-%j.out" --export=ALL slurm/train_act_shelf_restock.sbatch)
      cells=$((cells + 1))
      label=$([ "$from_scratch" = 1 ] && echo "from scratch" || echo "${ARM_CONDITIONING[$arm]:-continue}${STAGE1_JOB[$seed]:+ after ${STAGE1_JOB[$seed]}}")
      echo "    train  $train  ($label)"
    fi
    [ "${SKIP_EVAL:-0}" = 1 ] && continue
    # JOBS selects rollouts, e.g. JOBS=count to re-run only the object-count
    # job after the two-object layout changed.
    for tier in seen heldout novel; do
      [[ " $JOBS " == *" $tier "* ]] || continue
      evalj=$(rollout "$cell-$tier" three_object "${TIER[$tier]}" EVAL_MAX_STEPS=600)
      rollouts=$((rollouts + 1))
      echo "    eval   $tier -> $evalj"
    done
    [[ " $JOBS " == *" count "* ]] || continue
    evalj=$(rollout "$cell-count" two_object,four_object "${TIER[seen]}" \
      EVAL_STEPS_PER_OBJECT=$STEPS_PER_OBJECT)
    rollouts=$((rollouts + 1))
    echo "    eval   count (2, 4 objects) -> $evalj"
  done
done
echo
echo "$cells training cells, $rollouts rollouts."
echo "Collect with scripts/collect_final_results.py"
