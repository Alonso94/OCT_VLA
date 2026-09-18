#!/usr/bin/env bash
# LoRA finetune a VLA on a shelf-restock dataset, in the LeRobot policy
# environment. Resources are supplied at submission, as for collection.
#
# BACKBONE selects pi05 or smolvla. The two differ in more than their weights:
# pi0.5 needs openpi's camera names and supports gradient checkpointing, while
# SmolVLA takes the dataset's own camera keys and has no such config field. All
# three differences are handled below rather than by a second copy of this
# script, so the arms stay comparable in everything else.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
# No --mem: see the note in collect_shelf_restock_array.sbatch.

set -euo pipefail

: "${OCTVLA_REPO:?Set OCT_VLA checkout path}"
: "${OCTVLA_POLICY_PYTHON:?Set LeRobot Python executable}"
: "${OCTVLA_DATASET_ROOT:?Set durable LeRobot dataset root}"
: "${OCTVLA_OUTPUT_ROOT:?Set training output root}"
: "${TRAIN_VARIANT:?Set rgb or object}"

case "$TRAIN_VARIANT" in rgb|object) ;; *) echo "Invalid TRAIN_VARIANT" >&2; exit 2 ;; esac

BACKBONE="${BACKBONE:-pi05}"
case "$BACKBONE" in
  pi05)
    DEFAULT_BASE_MODEL="lerobot/pi05_base"
    OBJECT_POLICY_TYPE="control_pi05"
    # masked_pi05, not stock pi05: identical network, but it excludes padded
    # action targets from the loss the way ACT already does. Without it the RGB
    # arm would fit repeated terminal actions that the object arms do not,
    # confounding "conditioning" with "which arm trained on padding".
    STOCK_POLICY_TYPE="masked_pi05"
    # Measured: batch 16 fits in 15.6 GB of an A40's 49 GB with both of these.
    PRECISION_ARGS=(--policy.dtype=bfloat16 --policy.gradient_checkpointing=true)
    ;;
  smolvla)
    DEFAULT_BASE_MODEL="lerobot/smolvla_base"
    OBJECT_POLICY_TYPE="control_smolvla"
    STOCK_POLICY_TYPE="smolvla"
    # SmolVLAConfig has neither `dtype` nor `gradient_checkpointing`; passing
    # pi0.5's flags here fails the run at config parsing, before anything
    # loads. Its equivalent lever is automatic mixed precision. SmolVLA is also
    # far smaller than pi0.5, so the memory that gradient checkpointing buys
    # back is not needed.
    PRECISION_ARGS=(--policy.use_amp=true)
    ;;
  *) echo "Invalid BACKBONE: $BACKBONE (expected pi05 or smolvla)" >&2; exit 2 ;;
esac

# Arm C of the sweep ("is the object win real, or task-state leakage?") trains
# on the *same* dataset as arm B and differs only in how the policy reads the
# stored tokens, so this is a separate knob from TRAIN_VARIANT rather than a
# third variant with a dataset of its own.
OBJECT_TOKEN_MODE="${OBJECT_TOKEN_MODE:-full}"
case "$OBJECT_TOKEN_MODE" in full|role_stripped) ;; *) echo "Invalid OBJECT_TOKEN_MODE" >&2; exit 2 ;; esac
if [ "$TRAIN_VARIANT" = "rgb" ] && [ "$OBJECT_TOKEN_MODE" != "full" ]; then
  echo "OBJECT_TOKEN_MODE=$OBJECT_TOKEN_MODE is meaningless for TRAIN_VARIANT=rgb" >&2
  exit 2
fi

TRAIN_SEED="${TRAIN_SEED:-1000}"

DATASET="$OCTVLA_DATASET_ROOT/three_object_${TRAIN_VARIANT}"
[ -d "$DATASET" ] || { echo "No dataset at $DATASET" >&2; exit 2; }

# eval_split is a derived quantity, not a hyperparameter: it is whatever value
# makes LeRobot's positional hold-out select exactly the reserved validation
# seeds. Read it from the dataset rather than restating it, so the two can
# never drift apart. See scripts/build_shelf_restock_splits.py.
SPLIT_MANIFEST="$DATASET/split_manifest.json"
[ -f "$SPLIT_MANIFEST" ] || { echo "No split_manifest.json in $DATASET" >&2; exit 2; }
EVAL_SPLIT=$("$OCTVLA_POLICY_PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['eval_split'])" "$SPLIT_MANIFEST")

BASE_MODEL="${OCTVLA_BASE_MODEL:-$DEFAULT_BASE_MODEL}"

# The variant selects the policy CLASS, not just the dataset. `object` swaps in
# the object-conditioned plugin, which reads the dataset's object-token columns
# through a zero-initialised residual; `rgb` is the stock backbone and ignores
# them. Getting this wrong is silent and costly, and it has already happened
# once: a stale copy of this script trained the "object" arm as stock pi0.5 on
# a dataset that merely *contained* tokens, and the two arms produced identical
# losses to four decimal places for 2.5 GPU-hours before anyone noticed. The
# arms must differ here, not only in --dataset.root.
#
# The stock policy loads via --policy.path (type comes from the checkpoint).
# The plugin cannot: its type is not in LeRobot's registry until its package is
# imported, so it needs --policy.type plus --policy.discover_packages_path,
# with the base weights supplied separately as --policy.pretrained_path.
if [ "$TRAIN_VARIANT" = "object" ]; then
  POLICY_ARGS=(
    --policy.type="$OBJECT_POLICY_TYPE"
    --policy.pretrained_path="$BASE_MODEL"
    --policy.discover_packages_path=oct_vla.policies
    --policy.object_token_mode="$OBJECT_TOKEN_MODE"
  )
elif [ "$STOCK_POLICY_TYPE" = "$BACKBONE" ]; then
  # A stock LeRobot type: load via --policy.path and take the type from the
  # checkpoint, as before.
  POLICY_ARGS=(--policy.path="$BASE_MODEL")
else
  # A plugin type, so the same treatment the object arm gets: the type is not
  # in LeRobot's registry until this package is imported, and the base weights
  # come in separately.
  POLICY_ARGS=(
    --policy.type="$STOCK_POLICY_TYPE"
    --policy.pretrained_path="$BASE_MODEL"
    --policy.discover_packages_path=oct_vla.policies
  )
fi

# Name the cell, not the job. A job-id-only directory makes a sweep's runs
# mutually indistinguishable after the fact -- you cannot tell arm B seed 1001
# from arm C seed 1002 without going back to the Slurm accounting database,
# and aggregation has nothing to group by. Job id goes in run_metadata.json
# below instead, where it is still available but is not the identity.
RUN_NAME="${BACKBONE}_${TRAIN_VARIANT}_${OBJECT_TOKEN_MODE}_s${TRAIN_SEED}"
OUTPUT="$OCTVLA_OUTPUT_ROOT/$RUN_NAME"
# Refuse to touch an existing cell rather than resuming or overwriting it.
# Two runs of one cell silently sharing a directory is the failure that would
# quietly corrupt the whole comparison, and it is cheap to rule out here.
if [ -e "$OUTPUT" ]; then
  echo "Run directory already exists: $OUTPUT" >&2
  echo "Remove it to re-run this cell, or change TRAIN_SEED." >&2
  exit 2
fi
# Create only the parent: lerobot-train refuses to start if its own output
# directory already exists and resume is False, so pre-creating it fails the
# run at cfg.validate() before anything loads.
mkdir -p "$(dirname "$OUTPUT")"

# The base checkpoint and dataset are already local; keep the job off the network
# so a hub hiccup cannot kill a 24-hour run.
export HF_HOME="${HF_HOME:?Set HF_HOME}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID"
# torchcodec decodes ~9.5x faster than the PyAV fallback; without it the
# dataloader, not the GPU, sets training throughput. Opt in here rather than
# globally: these libraries belong to the policy environment, and exposing them
# to the simulator environment stops SAPIEN importing at all.
if [ -n "${OCTVLA_TORCHCODEC_LIBS:-}" ]; then
  export LD_LIBRARY_PATH="$OCTVLA_TORCHCODEC_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "backbone   : $BACKBONE"
echo "dataset    : $DATASET"
echo "variant    : $TRAIN_VARIANT"
echo "token_mode : $OBJECT_TOKEN_MODE"
echo "seed       : $TRAIN_SEED"
echo "eval_split : $EVAL_SPLIT"
echo "output     : $OUTPUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The cell's identity, recorded rather than left in a job log that ages out of
# the Slurm database; aggregate_eval.py reads it to group evaluations by arm
# and seed. Deliberately written to a sibling registry, NOT into $OUTPUT --
# creating $OUTPUT here is exactly what the note above says fails the run at
# cfg.validate() before anything loads.
mkdir -p "$OCTVLA_OUTPUT_ROOT/run_metadata"
cat > "$OCTVLA_OUTPUT_ROOT/run_metadata/${RUN_NAME}.json" <<JSON
{
  "run_name": "$RUN_NAME",
  "backbone": "$BACKBONE",
  "conditioning": "$TRAIN_VARIANT",
  "token_mode": "$OBJECT_TOKEN_MODE",
  "seed": $TRAIN_SEED,
  "steps": ${TRAIN_STEPS:-20000},
  "batch_size": ${BATCH_SIZE:-4},
  "dataset": "$DATASET",
  "job_id": "${SLURM_JOB_ID:-none}"
}
JSON

# pi0.5 was pretrained with openpi's camera names, and make_policy hard-fails on
# any mismatch rather than guessing which of our views is the base view. SmolVLA
# takes the dataset's own keys and needs no map at all. Look it up by policy
# type rather than restating it here: evaluation reads the type back from the
# checkpoint and resolves the same map, and inference would not complain about
# a wrong one -- every camera shares shape and dtype, so a mismatch shows up
# only as an unexplained drop in success rate.
POLICY_TYPE_FOR_MAP="$STOCK_POLICY_TYPE"
[ "$TRAIN_VARIANT" = "object" ] && POLICY_TYPE_FOR_MAP="$OBJECT_POLICY_TYPE"
RENAME_MAP=$(PYTHONPATH="$OCTVLA_REPO/src" "$OCTVLA_POLICY_PYTHON" -c \
  "import json,sys; from oct_vla.data.policy_inputs import rename_map_for; print(json.dumps(rename_map_for(sys.argv[1])))" \
  "$POLICY_TYPE_FOR_MAP")

# The plugin lives in this repo, so the training process itself needs src on
# the path -- previously only the rename-map subshell above did.
export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

srun "$OCTVLA_POLICY_PYTHON" -u -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="local/oct-vla-shelf-restock-three_object-${TRAIN_VARIANT}" \
  --dataset.root="$DATASET" \
  --dataset.eval_split="$EVAL_SPLIT" \
  --rename_map="$RENAME_MAP" \
  "${POLICY_ARGS[@]}" \
  "${PRECISION_ARGS[@]}" \
  --policy.push_to_hub=false \
  --peft.method_type=LORA \
  --peft.r="${LORA_R:-16}" \
  --peft.lora_alpha="${LORA_ALPHA:-32}" \
  --steps="${TRAIN_STEPS:-20000}" \
  --batch_size="${BATCH_SIZE:-4}" \
  --num_workers="${NUM_WORKERS:-8}" \
  --eval_steps="${EVAL_STEPS:-50}" \
  --max_eval_samples="${MAX_EVAL_SAMPLES:-256}" \
  --save_freq="${SAVE_FREQ:-2000}" \
  --log_freq=100 \
  --seed="$TRAIN_SEED" \
  --output_dir="$OUTPUT" \
  --job_name="$RUN_NAME" \
  --wandb.enable=false
