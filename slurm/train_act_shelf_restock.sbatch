#!/usr/bin/env bash
# Train ACT on a shelf-restock dataset. Two roles, selected by the dataset:
#
#   RGB + joint deltas      -- a known-good architecture on the same data the
#                              VLA arms use, to establish that SOMETHING solves
#                              this task through this pipeline. Without it, a
#                              zero from a VLA is uninterpretable: it could be
#                              the policy, the data, or the task.
#
#   Privileged state only   -- object tokens as observation.environment_state,
#                              no cameras. An upper bound: if privileged state
#                              alone solves the task easily, the vision-based
#                              comparison has little headroom and the sweep is
#                              measuring noise (docs/experiment_sweep.md, §7.4).
#
# ACT rather than another VLA on purpose: it is small, trains in well under an
# hour, and is the standard RoboTwin/ALOHA baseline, so a failure here points at
# the data or the task rather than at an exotic policy.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00

set -euo pipefail

: "${OCTVLA_REPO:?Set OCT_VLA checkout path}"
: "${OCTVLA_POLICY_PYTHON:?Set LeRobot Python executable}"
: "${OCTVLA_DATASET_ROOT:?Set durable LeRobot dataset root}"
: "${OCTVLA_OUTPUT_ROOT:?Set training output root}"
: "${ACT_DATASET:?Set the dataset directory name, e.g. three_object_rgb}"

DATASET="$OCTVLA_DATASET_ROOT/$ACT_DATASET"
[ -d "$DATASET" ] || { echo "No dataset at $DATASET" >&2; exit 2; }
SPLIT_MANIFEST="$DATASET/split_manifest.json"
[ -f "$SPLIT_MANIFEST" ] || { echo "No split_manifest.json in $DATASET" >&2; exit 2; }
EVAL_SPLIT=$("$OCTVLA_POLICY_PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['eval_split'])" "$SPLIT_MANIFEST")

# Refused rather than ignored: this variable once selected a "semantic" arm that
# nothing read, so that arm trained the entity model under another name. The
# datasets are v2 (17 columns) with no variant code, and control_act embeds v2
# only, so there is no semantic path to select until a v3 export exists.
if [ "${OBJECT_SEMANTICS:-0}" != 0 ]; then
  echo "OBJECT_SEMANTICS is not wired: no v3 dataset or embedding exists" >&2
  exit 2
fi
ACT_POLICY_TYPE="${ACT_POLICY_TYPE:-act}"
POLICY_ARGS=(--policy.type="$ACT_POLICY_TYPE")
case "$ACT_POLICY_TYPE" in
  act)
    # Continue a trained rgb checkpoint: the budget control for stage two.
    # Every conditioned arm trains its 40k steps *on top of* stage one's 40k,
    # so without an rgb arm given the same extra steps, a gain could be extra
    # training rather than objects. This used to be ignored for plain act --
    # the variable was read only in the control_act branch -- so a
    # "continued" run would have silently retrained from scratch.
    if [ -n "${ACT_PRETRAINED:-}" ]; then
      [ -d "$ACT_PRETRAINED" ] || { echo "No checkpoint at $ACT_PRETRAINED" >&2; exit 2; }
      [ -n "${RUN_SUFFIX:-}" ] || { echo "A continued act run needs RUN_SUFFIX" >&2; exit 2; }
      POLICY_ARGS+=(--policy.pretrained_path="$ACT_PRETRAINED")
      ACT_STAGE=continue
    fi
    ;;
  control_act)
    ENTITY_ARGS_TEXT=$("$OCTVLA_POLICY_PYTHON" "$OCTVLA_REPO/scripts/entity_training_args.py" "$DATASET")
    mapfile -t ENTITY_ARGS <<< "$ENTITY_ARGS_TEXT"
    POLICY_ARGS+=(--policy.discover_packages_path=oct_vla.policies "${ENTITY_ARGS[@]}")
    ACT_CONDITIONING=object
    # Encoder-side branches beside the layerwise decoder one (LPWM,
    # arXiv:2603.04553): AdaLN-Zero on every block, or entity tokens in the
    # encoder sequence. Each is its own arm, so each isolates one addition.
    case "${OBJECT_ADALN:-0}${OBJECT_INCONTEXT:-0}" in
      00) ;;
      10) POLICY_ARGS+=(--policy.object_adaln=true); ACT_CONDITIONING=object_adaln ;;
      01) POLICY_ARGS+=(--policy.object_incontext=true); ACT_CONDITIONING=object_incontext ;;
      *) echo "Set at most one of OBJECT_ADALN / OBJECT_INCONTEXT, to 1" >&2; exit 2 ;;
    esac
    # Stage two of ControlVLA's recipe. The paper pre-trains a general policy
    # on DROID and only then adds object-centric conditioning; its "w/o
    # pretrain" ablation -- conditioning a from-scratch policy -- reports
    # severe jitter and large drops, hypothesised to be object features
    # offering a low-loss shortcut before stable visual features are learned.
    #
    # Training control_act from scratch *is* that ablation. Pointing this at a
    # trained rgb checkpoint makes it the method instead: zero-initialised K/V
    # means stage two begins as numerically exactly the stage-one policy.
    # Verified: all 160 ACT tensors carry over and the object branch stays zero.
    #
    # Left unset deliberately reproduces the ablation, which is worth one cell.
    if [ -n "${ACT_PRETRAINED:-}" ]; then
      [ -d "$ACT_PRETRAINED" ] || { echo "No checkpoint at $ACT_PRETRAINED" >&2; exit 2; }
      POLICY_ARGS+=(--policy.pretrained_path="$ACT_PRETRAINED")
      ACT_STAGE=finetune
    else
      ACT_STAGE=scratch
    fi
    ;;
  *) echo "Invalid ACT_POLICY_TYPE" >&2; exit 2 ;;
esac
RUN_NAME="${ACT_POLICY_TYPE}_${ACT_DATASET}_s${TRAIN_SEED:-1000}"
# A conditioned arm trained from scratch and one fine-tuned from an rgb
# checkpoint are different experiments and must not share an output directory.
[ "${ACT_STAGE:-}" = scratch ] && RUN_NAME="${RUN_NAME}_scratch"
[ -n "${RUN_SUFFIX:-}" ] && RUN_NAME="${RUN_NAME}_${RUN_SUFFIX}"
OUTPUT="$OCTVLA_OUTPUT_ROOT/$RUN_NAME"
if [ -e "$OUTPUT" ]; then
  echo "Run directory already exists: $OUTPUT" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUTPUT")"

export HF_HOME="${HF_HOME:?Set HF_HOME}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$SLURM_JOB_ID"
if [ -n "${OCTVLA_TORCHCODEC_LIBS:-}" ]; then
  export LD_LIBRARY_PATH="$OCTVLA_TORCHCODEC_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

echo "dataset    : $DATASET"
echo "run        : $RUN_NAME"
echo "eval_split : $EVAL_SPLIT"
# The resolved policy flags, so a log shows which arm actually trained.
echo "policy args: ${POLICY_ARGS[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$OCTVLA_OUTPUT_ROOT/run_metadata"
cat > "$OCTVLA_OUTPUT_ROOT/run_metadata/${RUN_NAME}.json" <<JSON
{
  "run_name": "$RUN_NAME",
  "backbone": "act",
  "conditioning": "${ACT_CONDITIONING:-rgb}",
  "stage": "${ACT_STAGE:-pretrain}",
  "pretrained_from": "${ACT_PRETRAINED:-}",
  "token_mode": "full",
  "seed": ${TRAIN_SEED:-1000},
  "steps": ${TRAIN_STEPS:-40000},
  "batch_size": ${BATCH_SIZE:-8},
  "dataset": "$DATASET",
  "job_id": "${SLURM_JOB_ID:-none}"
}
JSON

# No --rename_map: ACT has no pretrained camera names to match, so it takes the
# dataset's own keys. Chunk 50 rather than ACT's default 100 because the
# demonstrations are only 60-180 steps long.
srun "$OCTVLA_POLICY_PYTHON" -u -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="local/oct-vla-shelf-restock-${ACT_DATASET#three_object_}" \
  --dataset.root="$DATASET" \
  --dataset.eval_split="$EVAL_SPLIT" \
  "${POLICY_ARGS[@]}" \
  --policy.chunk_size="${CHUNK_SIZE:-50}" \
  --policy.n_action_steps="${N_ACTION_STEPS:-50}" \
  --policy.push_to_hub=false \
  --steps="${TRAIN_STEPS:-40000}" \
  --batch_size="${BATCH_SIZE:-8}" \
  --num_workers="${NUM_WORKERS:-8}" \
  --eval_steps="${EVAL_STEPS:-1000}" \
  --max_eval_samples="${MAX_EVAL_SAMPLES:-256}" \
  --save_freq="${SAVE_FREQ:-5000}" \
  --log_freq=200 \
  --seed="${TRAIN_SEED:-1000}" \
  --output_dir="$OUTPUT" \
  --job_name="$RUN_NAME" \
  --wandb.enable=false
