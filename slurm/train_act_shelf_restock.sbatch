#!/usr/bin/env bash
# Train ACT on a shelf-restock dataset view: stage 1 (`act`), the budget control
# (`act` continued from stage 1), or a conditioned arm (`control_act`, with
# CONDITIONING=kv|kv_adaln|kv_tokens, fine-tuned from stage 1 or from scratch).
# Arms are queued by slurm/submit_final_experiments.sh; see docs/method.md.
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
# nothing read, so that arm trained the entity model under another name.
if [ "${OBJECT_SEMANTICS:-0}" != 0 ]; then
  echo "OBJECT_SEMANTICS is not wired: there is no semantic entity yet" >&2
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
    # The arm (oct_vla.policies.conditioning): kv, kv_adaln or kv_tokens.
    # Refused rather than ignored if unknown -- an ignored arm variable is how
    # the old `semantic` arm trained the entity model under another name.
    CONDITIONING="${CONDITIONING:-kv}"
    case "$CONDITIONING" in kv|kv_adaln|kv_tokens) ;; *) echo "Invalid CONDITIONING=$CONDITIONING" >&2; exit 2 ;; esac
    if [ -n "${OBJECT_ADALN:-}${OBJECT_INCONTEXT:-}" ]; then
      echo "OBJECT_ADALN/OBJECT_INCONTEXT are gone since the cleanup: set CONDITIONING" >&2; exit 2
    fi
    POLICY_ARGS+=(--policy.object_conditioning="$CONDITIONING")
    ACT_CONDITIONING="$CONDITIONING"
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
# USE_VAE=0 trains ACT without its CVAE latent. On this corpus the latent
# collapses on every seed -- the KL term falls to 0.005 by 10k steps and 0.000 by
# 40k -- so the encoder carries nothing; the ablation asks whether removing it
# changes closed-loop behaviour or seed variance (ControlVLA blames posterior
# collapse for ACT's weakness in low-data settings).
case "${USE_VAE:-1}" in
  1) ;;
  0) POLICY_ARGS+=(--policy.use_vae=false); [ -n "${RUN_SUFFIX:-}" ] || RUN_SUFFIX=novae ;;
  *) echo "USE_VAE must be 0 or 1" >&2; exit 2 ;;
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
  "use_vae": ${USE_VAE:-1},
  "pretrained_from": "${ACT_PRETRAINED:-}",
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
  --save_freq="${SAVE_FREQ:-${TRAIN_STEPS:-40000}}" \
  --log_freq=200 \
  --seed="${TRAIN_SEED:-1000}" \
  --output_dir="$OUTPUT" \
  --job_name="$RUN_NAME" \
  --wandb.enable=false

# Storage: 1 TB of vault for everything. A 40k-step run used to keep eight
# checkpoints with optimiser state, 9.4 GB; every evaluation reads `last`, and
# stage 2 continues from `last/pretrained_model`. So one checkpoint, and its
# optimiser state removed once the run succeeded (set -e): ~0.4 GB a run.
rm -rf "$OUTPUT"/checkpoints/*/training_state
du -sh "$OUTPUT" 2>/dev/null | sed 's/^/kept       : /'
