#!/usr/bin/env bash
# LoRA finetune pi0.5 on a shelf-restock dataset, in the LeRobot policy
# environment. Resources are supplied at submission, as for collection.
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

DATASET="$OCTVLA_DATASET_ROOT/three_object_${TRAIN_VARIANT}"
[ -d "$DATASET" ] || { echo "No dataset at $DATASET" >&2; exit 2; }

# eval_split is a derived quantity, not a hyperparameter: it is whatever value
# makes LeRobot's positional hold-out select exactly the reserved validation
# seeds. Read it from the dataset rather than restating it, so the two can
# never drift apart. See scripts/build_shelf_restock_splits.py.
SPLIT_MANIFEST="$DATASET/split_manifest.json"
[ -f "$SPLIT_MANIFEST" ] || { echo "No split_manifest.json in $DATASET" >&2; exit 2; }
EVAL_SPLIT=$("$OCTVLA_POLICY_PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['eval_split'])" "$SPLIT_MANIFEST")

OUTPUT="$OCTVLA_OUTPUT_ROOT/pi05_lora_three_object_${TRAIN_VARIANT}_${SLURM_JOB_ID}"
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
# torchcodec decodes ~9.5x faster than the PyAV fallback here; without this the
# dataloader, not the GPU, sets the training throughput. See ~/octvla/env.sh.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

echo "dataset    : $DATASET"
echo "variant    : $TRAIN_VARIANT"
echo "eval_split : $EVAL_SPLIT"
echo "output     : $OUTPUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# pi0.5 was pretrained with openpi's camera names, and make_policy hard-fails on
# any mismatch rather than guessing which of our views is the base view. Map our
# canonical names onto its expected ones; the same map must be used at
# evaluation time, or the policy silently receives a wrist view as its base.
RENAME_MAP='{"observation.images.head": "observation.images.base_0_rgb",
             "observation.images.left_wrist": "observation.images.left_wrist_0_rgb",
             "observation.images.right_wrist": "observation.images.right_wrist_0_rgb"}'

srun "$OCTVLA_POLICY_PYTHON" -u -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="local/oct-vla-shelf-restock-three_object-${TRAIN_VARIANT}" \
  --dataset.root="$DATASET" \
  --dataset.eval_split="$EVAL_SPLIT" \
  --rename_map="$RENAME_MAP" \
  --policy.path=lerobot/pi05_base \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
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
  --seed=1000 \
  --output_dir="$OUTPUT" \
  --job_name="pi05_lora_three_object_${TRAIN_VARIANT}" \
  --wandb.enable=false
