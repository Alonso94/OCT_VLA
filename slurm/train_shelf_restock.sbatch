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
PEFT_ARGS=(--peft.method_type=LORA --peft.r="${LORA_R:-16}" --peft.lora_alpha="${LORA_ALPHA:-32}")
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
  groot)
    # Named by a config field rather than passed as a path: GrootConfig
    # defaults base_model_path to nvidia/GR00T-N1.7-3B and the policy loads the
    # transformers-format weights itself. This is what X-VLA could not do --
    # its published checkpoint carries `model_type`, not LeRobot's `type`.
    DEFAULT_BASE_MODEL=""
    OBJECT_POLICY_TYPE="control_groot"
    # GR00T's own loss is already masked: its processor (GrootN17PackInputsStep)
    # builds `action_mask` from both `action_is_pad` and the valid action width,
    # and the head divides by `action_mask.sum()` (groot_n1_7.py). So neither
    # the 17.4% padded steps nor the 116 unused columns of its 132-wide action
    # enter the loss (tests/policies/test_layerwise_backbones.py).
    #
    # slim_groot, not stock groot: the same network, but its checkpoint leaves
    # out the frozen 6.1 GB backbone, which is rebuilt from the base on load.
    # A stock save was ~49 GB on disk per checkpoint with optimiser state.
    STOCK_POLICY_TYPE="slim_groot"
    PRECISION_ARGS=(--policy.use_amp=true)
    # No LoRA: GR00T freezes its LLM and vision tower by default and tunes only
    # the projector and diffusion head, and its weights arrive through
    # base_model_path rather than pretrained_path -- which is what LeRobot's
    # PEFT guard checks, so it refuses LoRA outright.
    PEFT_ARGS=()
    ;;
  vla_jepa)
    # The LeRobot-format conversion, not ginwind/VLA-JEPA -- that repo ships
    # bare .pt files and VLAJEPAPolicy._load_as_safetensor needs safetensors.
    # Pretrain rather than LIBERO: we want the generic checkpoint, not one
    # already specialised to another benchmark's tasks.
    DEFAULT_BASE_MODEL="lerobot/VLA-JEPA-Pretrain"
    OBJECT_POLICY_TYPE="control_vla_jepa"
    STOCK_POLICY_TYPE="vla_jepa"
    PRECISION_ARGS=(--policy.use_amp=true)
    ;;
  *) echo "Invalid BACKBONE: $BACKBONE (expected pi05, smolvla, groot or vla_jepa)" >&2; exit 2 ;;
esac

# The object arm (oct_vla.policies.conditioning): kv, kv_adaln or kv_tokens.
# Refused rather than ignored if unknown or given to an rgb run -- an ignored
# arm variable is how the old `semantic` arm trained the entity model under
# another name.
for gone in OBJECT_ADALN OBJECT_INCONTEXT OBJECT_TOKEN_MODE OBJECT_INJECTION_MODE \
            OBJECT_ENTITY_POSITIONAL OBJECT_REPRESENTATION; do
  if [ -n "${!gone:-}" ]; then echo "$gone is gone since the cleanup: set CONDITIONING" >&2; exit 2; fi
done
CONDITIONING="${CONDITIONING:-}"
if [ "$TRAIN_VARIANT" = object ]; then
  CONDITIONING="${CONDITIONING:-kv}"
  case "$CONDITIONING" in kv|kv_adaln|kv_tokens) ;; *) echo "Invalid CONDITIONING=$CONDITIONING" >&2; exit 2 ;; esac
  # The KV hooks hold batch context only during forward; checkpoint
  # recomputation would replay them without it (see control_pi05's config).
  [ "$BACKBONE" = pi05 ] && PRECISION_ARGS=(--policy.dtype=bfloat16 --policy.gradient_checkpointing=false)
elif [ -n "$CONDITIONING" ]; then
  echo "CONDITIONING=$CONDITIONING is meaningless for TRAIN_VARIANT=rgb" >&2; exit 2
fi

TRAIN_SEED="${TRAIN_SEED:-1000}"

# The dataset supplies the action encoding, the state width and the control
# space, so it has to be the one this cell means -- not merely one of the same
# variant. TRAIN_DATASET names it; the default keeps the old convention.
TRAIN_DATASET="${TRAIN_DATASET:-three_object_${TRAIN_VARIANT}}"
DATASET="$OCTVLA_DATASET_ROOT/$TRAIN_DATASET"
[ -d "$DATASET" ] || { echo "No dataset at $DATASET" >&2; exit 2; }

# eval_split is a derived quantity, not a hyperparameter: it is whatever value
# makes LeRobot's positional hold-out select exactly the reserved validation
# seeds. Read it from the dataset rather than restating it, so the two can
# never drift apart. See scripts/build_shelf_restock_splits.py.
SPLIT_MANIFEST="$DATASET/split_manifest.json"
[ -f "$SPLIT_MANIFEST" ] || { echo "No split_manifest.json in $DATASET" >&2; exit 2; }
EVAL_SPLIT=$("$OCTVLA_POLICY_PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['eval_split'])" "$SPLIT_MANIFEST")

BASE_MODEL="${OCTVLA_BASE_MODEL:-$DEFAULT_BASE_MODEL}"

# Stage two of the ControlVLA recipe, and its budget control. STAGE1_CHECKPOINT
# is a full-weights stage-1 checkpoint that both the object arms (control_*) and
# rgb_cont (the stock type, continued) start from:
#   pi05, smolvla  the *merged* directory scripts/merge_stage1_adapter.py writes:
#                  stage 1 is a LoRA adapter, which no loader can put under a
#                  different policy class
#   groot          stage 1's own pretrained_model (slim: head only)
# Every such load is verified tensor by tensor (oct_vla.policies.stage_loading):
# pi0.5's loader returns an unloaded model on failure, printing a warning only.
STAGE="pretrain"
if [ -n "${STAGE1_CHECKPOINT:-}" ]; then
  [ -f "$STAGE1_CHECKPOINT/model.safetensors" ] || {
    echo "STAGE1_CHECKPOINT=$STAGE1_CHECKPOINT has no model.safetensors" >&2
    [ -f "$STAGE1_CHECKPOINT/adapter_config.json" ] && \
      echo "It is a LoRA adapter: merge it first with scripts/merge_stage1_adapter.py" >&2
    exit 2
  }
  [ -n "${TRAIN_TAG:-}" ] || { echo "A stage-2 run needs TRAIN_TAG to name it" >&2; exit 2; }
  BASE_MODEL="$STAGE1_CHECKPOINT"
  STAGE=$([ "$TRAIN_VARIANT" = object ] && echo finetune || echo continue)
fi

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
#
# A backbone with no DEFAULT_BASE_MODEL is a third case: GR00T carries its
# checkpoint in a config field (base_model_path) and loads the
# transformers-format weights itself, so there is nothing to hand to
# --policy.path or --policy.pretrained_path. Passing an empty one is not a
# no-op -- it resolves to the current directory.
if [ "$TRAIN_VARIANT" = "object" ]; then
  POLICY_ARGS=(
    --policy.type="$OBJECT_POLICY_TYPE"
    --policy.discover_packages_path=oct_vla.policies
    --policy.object_conditioning="$CONDITIONING"
  )
  [ -n "$BASE_MODEL" ] && POLICY_ARGS+=(--policy.pretrained_path="$BASE_MODEL")
elif [ -z "$BASE_MODEL" ]; then
  POLICY_ARGS=(--policy.type="$STOCK_POLICY_TYPE" --policy.discover_packages_path=oct_vla.policies)
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

# The entity capacity and training-split normalisation the dataset declares.
if [ "$TRAIN_VARIANT" = object ]; then
  ENTITY_ARGS_TEXT=$("$OCTVLA_POLICY_PYTHON" "$OCTVLA_REPO/scripts/entity_training_args.py" "$DATASET")
  mapfile -t ENTITY_ARGS <<< "$ENTITY_ARGS_TEXT"
  POLICY_ARGS+=("${ENTITY_ARGS[@]}")
fi

# The execution horizon is a measured convention (25; see
# docs/experiment_matrix.md (at tag stageA-2026-09-23)), and the backbones disagree on it -- pi0.5 and
# SmolVLA default to 50, GR00T to 40. Set it at training so the checkpoint
# carries it and evaluation does not have to remember to override it. Left
# unset, each backbone keeps its own default.
[ -n "${N_ACTION_STEPS:-}" ] && POLICY_ARGS+=(--policy.n_action_steps="$N_ACTION_STEPS")

# Name the cell, not the job. A job-id-only directory makes a sweep's runs
# mutually indistinguishable after the fact -- you cannot tell arm B seed 1001
# from arm C seed 1002 without going back to the Slurm accounting database,
# and aggregation has nothing to group by. Job id goes in run_metadata.json
# below instead, where it is still available but is not the identity.
RUN_NAME="${BACKBONE}_${CONDITIONING:-rgb}_s${TRAIN_SEED}"
# TRAIN_TAG distinguishes cells this name cannot -- a different dataset view,
# most of all, since the action encoding does not appear in the name at all.
[ -n "${TRAIN_TAG:-}" ] && RUN_NAME="${RUN_NAME}_${TRAIN_TAG}"
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
if [ "$TRAIN_DATASET" = "three_object_${TRAIN_VARIANT}" ]; then
  TRAIN_REPO_ID="local/oct-vla-shelf-restock-three_object-${TRAIN_VARIANT}"
else
  TRAIN_REPO_ID="local/oct-vla-shelf-restock-${TRAIN_DATASET#three_object_}"
fi

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
echo "arm        : ${CONDITIONING:-rgb}"
echo "seed       : $TRAIN_SEED"
echo "eval_split : $EVAL_SPLIT"
echo "output     : $OUTPUT"
echo "stage      : $STAGE ${STAGE1_CHECKPOINT:+from $STAGE1_CHECKPOINT}"
# The resolved policy flags, so a log shows which arm actually trained.
echo "policy args: ${POLICY_ARGS[*]}"
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
  "conditioning": "${CONDITIONING:-rgb}",
  "stage": "$STAGE",
  "stage1_checkpoint": "${STAGE1_CHECKPOINT:-}",
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

# INIT_CHECK=1: instead of training, build this exact run and prove stage 2
# starts at stage 1 and that gradient reaches the object branch
# (scripts/check_stage2_init.py). Minutes, not hours, and it writes no checkpoint.
ENTRY=(-m lerobot.scripts.lerobot_train)
if [ "${INIT_CHECK:-0}" = 1 ]; then
  [ -n "${STAGE1_CHECKPOINT:-}" ] || { echo "INIT_CHECK needs STAGE1_CHECKPOINT" >&2; exit 2; }
  ENTRY=("$OCTVLA_REPO/scripts/check_stage2_init.py")
fi
srun "$OCTVLA_POLICY_PYTHON" -u "${ENTRY[@]}" \
  --dataset.repo_id="$TRAIN_REPO_ID" \
  --dataset.root="$DATASET" \
  --dataset.eval_split="$EVAL_SPLIT" \
  --rename_map="$RENAME_MAP" \
  "${POLICY_ARGS[@]}" \
  "${PRECISION_ARGS[@]}" \
  --policy.push_to_hub=false \
  "${PEFT_ARGS[@]}" \
  --steps="${TRAIN_STEPS:-20000}" \
  --batch_size="${BATCH_SIZE:-4}" \
  --num_workers="${NUM_WORKERS:-8}" \
  --eval_steps="${EVAL_STEPS:-50}" \
  --max_eval_samples="${MAX_EVAL_SAMPLES:-256}" \
  --save_freq="${SAVE_FREQ:-${TRAIN_STEPS:-20000}}" \
  --log_freq=100 \
  --seed="$TRAIN_SEED" \
  --output_dir="$OUTPUT" \
  --job_name="$RUN_NAME" \
  --wandb.enable=false

# Storage: 1 TB of vault for everything. Only the final checkpoint is kept
# (save_freq defaults to the step budget; every evaluation reads `last`), and
# its optimiser state goes once the run has succeeded -- nothing here resumes.
# Only this run's own directory, and only after srun returned 0 (set -e).
rm -rf "$OUTPUT"/checkpoints/*/training_state
du -sh "$OUTPUT" 2>/dev/null | sed 's/^/kept       : /'
