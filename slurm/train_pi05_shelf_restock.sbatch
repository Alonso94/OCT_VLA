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

# The variant selects the policy CLASS, not just the dataset. `object` swaps in
# the object-conditioned plugin, which reads the dataset's object-token columns
# through a zero-initialised residual; `rgb` is stock pi0.5 and ignores them.
# Getting this wrong is silent and costly: pointing the object run at stock
# pi0.5 trains a second RGB model on a dataset that merely *contains* tokens,
# and the comparison then reports "object conditioning does not help" when it
# was never enabled. The two arms must differ here, not only in --dataset.root.
#
# Stock pi0.5 loads via --policy.path (type comes from the checkpoint). The
# plugin cannot: its type is not in LeRobot's registry until its package is
# imported, so it needs --policy.type plus --policy.discover_packages_path,
# with the base weights supplied separately as --policy.pretrained_path.
if [ "$TRAIN_VARIANT" = "object" ]; then
  POLICY_ARGS=(
    --policy.type=control_pi05
    --policy.pretrained_path="${OCTVLA_BASE_MODEL:-lerobot/pi05_base}"
    --policy.discover_packages_path=oct_vla.policies
  )
else
  POLICY_ARGS=(--policy.path="${OCTVLA_BASE_MODEL:-lerobot/pi05_base}")
fi

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
# torchcodec decodes ~9.5x faster than the PyAV fallback; without it the
# dataloader, not the GPU, sets training throughput. Opt in here rather than
# globally: these libraries belong to the policy environment, and exposing them
# to the simulator environment stops SAPIEN importing at all.
if [ -n "${OCTVLA_TORCHCODEC_LIBS:-}" ]; then
  export LD_LIBRARY_PATH="$OCTVLA_TORCHCODEC_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "dataset    : $DATASET"
echo "variant    : $TRAIN_VARIANT"
echo "eval_split : $EVAL_SPLIT"
echo "output     : $OUTPUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# pi0.5 was pretrained with openpi's camera names, and make_policy hard-fails on
# any mismatch rather than guessing which of our views is the base view. Read the
# map from the package rather than restating it here: evaluation must use the
# same one, and inference would not complain about a wrong one.
RENAME_MAP=$(PYTHONPATH="$OCTVLA_REPO/src" "$OCTVLA_POLICY_PYTHON" -c \
  "import json; from oct_vla.data.policy_inputs import PI05_RENAME_MAP; print(json.dumps(PI05_RENAME_MAP))")

# The plugin lives in this repo, so the training process itself needs src on
# the path -- previously only the rename-map subshell above did.
export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

srun "$OCTVLA_POLICY_PYTHON" -u -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="local/oct-vla-shelf-restock-three_object-${TRAIN_VARIANT}" \
  --dataset.root="$DATASET" \
  --dataset.eval_split="$EVAL_SPLIT" \
  --rename_map="$RENAME_MAP" \
  "${POLICY_ARGS[@]}" \
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
