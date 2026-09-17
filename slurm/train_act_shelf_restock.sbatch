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

RUN_NAME="act_${ACT_DATASET}_s${TRAIN_SEED:-1000}"
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$OCTVLA_OUTPUT_ROOT/run_metadata"
cat > "$OCTVLA_OUTPUT_ROOT/run_metadata/${RUN_NAME}.json" <<JSON
{
  "run_name": "$RUN_NAME",
  "backbone": "act",
  "conditioning": "${ACT_CONDITIONING:-rgb}",
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
  --policy.type=act \
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
