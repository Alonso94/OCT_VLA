#!/usr/bin/env bash
# Record one video per trained cell, so the grid can be watched rather than read.
#
# The cells span three output roots because each control space was swept into
# its own, and a checkpoint can only be scored against the dataset it was fit
# on. Rather than assume a naming rule, every cell names its environment
# overlay, its run and its dataset explicitly -- the combination that produced
# it is the thing being recorded, and getting it wrong yields a plausible video
# of the wrong policy.
#
# Usage:  bash slurm/submit_rollout_videos.sh [seed]
set -uo pipefail

SEED="${1:-800}"
STAGE="${VIDEO_STAGE:-$HPCVAULT/octvla-rollout-videos}"

# env overlay | run directory | EVAL_VARIANT | EVAL_DATASET | label
#
# pi0.5 arms: three observation variants x three action spaces, seed 1000.
# `object` is the ControlVLA arm; the role-stripped variant differs only in the
# checkpoint's own token mode, so both share EVAL_VARIANT=object.
CELLS=(
  "env.sh|pi05_rgb_full_s1000|rgb|three_object_rgb|pi05 RGB | EE delta"
  "env.sh|pi05_object_full_s1000|object|three_object_object|pi05 ControlVLA | EE delta"
  "env.sh|pi05_object_role_stripped_s1000|object|three_object_object|pi05 ControlVLA role-stripped | EE delta"

  "env-jointspace.sh|pi05_rgb_full_s1000|rgb|three_object_rgb|pi05 RGB | absolute joint"
  "env-jointspace.sh|pi05_object_full_s1000|object|three_object_object|pi05 ControlVLA | absolute joint"
  "env-jointspace.sh|pi05_object_role_stripped_s1000|object|three_object_object|pi05 ControlVLA role-stripped | absolute joint"

  "env-jointdelta.sh|pi05_rgb_full_s1000|rgb|three_object_rgb|pi05 RGB | joint delta"
  "env-jointdelta.sh|pi05_object_full_s1000|object|three_object_object|pi05 ControlVLA | joint delta"
  "env-jointdelta.sh|pi05_object_role_stripped_s1000|object|three_object_object|pi05 ControlVLA role-stripped | joint delta"

  # ACT cells, all on the r75 corpus, seed 1000.
  "env-jointdelta.sh|act_three_object_rgb_r75_s1000|rgb|three_object_rgb_r75|ACT RGB | joint delta"
  "env-jointdelta.sh|act_three_object_rgb_r75_abs_s1000|rgb|three_object_rgb_r75_abs|ACT RGB | absolute joint"
  "env-jointdelta.sh|act_three_object_rgb_r75_velo_s1000|rgb|three_object_rgb_r75_velo|ACT RGB | joint delta + velocity"
  "env-jointdelta.sh|act_three_object_rgb_r75_eedelta_s1000|rgb|three_object_rgb_r75_eedelta|ACT RGB | EE delta"
  "env-jointdelta.sh|act_three_object_rgb_r75_eeabs_s1000|rgb|three_object_rgb_r75_eeabs|ACT RGB | absolute EE"
  "env-jointdelta.sh|act_three_object_privileged_r75_s1000|privileged|three_object_privileged_r75|ACT privileged | joint delta"
  "env-jointdelta.sh|act_three_object_privileged_r75_abs_s1000|privileged|three_object_privileged_r75_abs|ACT privileged | absolute joint"
  "env-jointdelta.sh|act_three_object_privileged_r75_velo_s1000|privileged|three_object_privileged_r75_velo|ACT privileged | joint delta + velocity"
)

submitted=0
skipped=0
for cell in "${CELLS[@]}"; do
  IFS='|' read -r envfile run variant dataset label <<< "$cell"
  # Each cell in its own subshell: the overlays export the same variable names
  # at different values, and leaking one into the next would score a checkpoint
  # against another root's dataset.
  (
    # shellcheck disable=SC1090
    source "$HOME/octvla/$envfile" >/dev/null 2>&1

    RUN_DIR="$OCTVLA_OUTPUT_ROOT/$run"
    CKPT="$RUN_DIR/checkpoints/best/pretrained_model"
    [ -d "$CKPT" ] || CKPT="$RUN_DIR/checkpoints/last/pretrained_model"
    if [ ! -d "$CKPT" ]; then
      echo "SKIP  $label -- no checkpoint under $RUN_DIR" >&2
      exit 3
    fi
    if [ ! -d "$OCTVLA_DATASET_ROOT/$dataset" ]; then
      echo "SKIP  $label -- no dataset $OCTVLA_DATASET_ROOT/$dataset" >&2
      exit 3
    fi

    slug=$(echo "$label" | tr 'A-Z ' 'a-z_' | tr -cd 'a-z0-9_|' | tr '|' '-')
    # One file per cell, named for the cell and not the seed, so re-recording
    # overwrites in place. The alternative accumulates a file per seed per
    # refresh, which is what fills a quota.
    out="$STAGE"
    mkdir -p "$out"

    jid=$(sbatch --parsable \
      --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --gres=gpu:a40:1 \
      --time=01:00:00 --job-name="octvla-vid-$slug" \
      --output="$OCTVLA_OUTPUT_ROOT/slurm_logs/vid-$slug-%j.out" \
      --export=ALL,EVAL_CHECKPOINT="$CKPT",EVAL_VARIANT="$variant",EVAL_DATASET="$dataset",EVAL_SEEDS="$SEED",EVAL_PROFILES=three_object,EVAL_MAX_STEPS=600,EVAL_TAG="videos/$slug-$SEED",EVAL_VIDEO_DIR="$out",EVAL_VIDEO_LABEL="$label",EVAL_VIDEO_SEEDS="$SEED",EVAL_VIDEO_NAME="$slug" \
      "$OCTVLA_REPO/slurm/eval_shelf_restock.sbatch")
    echo "$jid  $label"
  )
  case $? in
    0) submitted=$((submitted + 1)) ;;
    *) skipped=$((skipped + 1)) ;;
  esac
done

echo
echo "submitted $submitted, skipped $skipped, seed $SEED -> $STAGE"
