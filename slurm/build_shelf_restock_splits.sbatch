#!/usr/bin/env bash
# Export one LeRobot dataset from the canonical collection, with the split
# encoded in episode order (scripts/build_shelf_restock_splits.py).
#
# A batch job rather than a login-node command because encoding the three
# camera streams for a few hundred clips takes hours; the privileged variant
# drops the cameras and finishes in minutes, but shares this script so both
# variants of a comparison are built by identical code.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00

set -euo pipefail

: "${OCTVLA_REPO:?Set OCT_VLA checkout path}"
: "${OCTVLA_POLICY_PYTHON:?Set LeRobot Python executable}"
: "${OCTVLA_COLLECTION_ROOT:?Set canonical collection root}"
: "${OCTVLA_DATASET_ROOT:?Set durable LeRobot dataset root}"
: "${DATASET_NAME:?Set the dataset directory name, e.g. three_object_rgb_r75}"

PROFILE="${COLLECTION_PROFILE:-three_object}"
OUTPUT="$OCTVLA_DATASET_ROOT/$DATASET_NAME"
if [ -e "$OUTPUT" ]; then
  echo "Refusing to overwrite an existing dataset: $OUTPUT" >&2
  exit 2
fi

export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
if [ -n "${OCTVLA_TORCHCODEC_LIBS:-}" ]; then
  export LD_LIBRARY_PATH="$OCTVLA_TORCHCODEC_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

ARGS=(
  --canonical-root "$OCTVLA_COLLECTION_ROOT/$PROFILE"
  --output "$OUTPUT"
  --repo-id "local/oct-vla-shelf-restock-${DATASET_NAME#three_object_}"
  --profile "$PROFILE"
  --control-space "${CONTROL_SPACE:-joint_delta}"
  --state-encoding "${STATE_ENCODING:-position}"
  --gripper-encoding "${GRIPPER_ENCODING:-measured_aperture}"
)
# `if` rather than `[ ... ] && ...`: under `set -e` a false test is a failing
# command, so the short form would abort the job whenever a flag is off.
if [ "${OBJECT_TOKENS:-0}" = "1" ]; then ARGS+=(--object-tokens); fi
if [ "${PRIVILEGED:-0}" = "1" ]; then ARGS+=(--privileged); fi
# The entity-v2 export. It fits its normalization on the reserved *training*
# partition alone and records the fitting sources in the split manifest, so it
# has to be the split builder rather than export_lerobot_dataset.py, whose
# --entity-training-sources is named by hand and unchecked.
if [ "${ENTITY_TOKENS:-0}" = "1" ]; then ARGS+=(--entity-tokens); fi
if [ -n "${MAX_ENTITIES:-}" ]; then ARGS+=(--max-entities "$MAX_ENTITIES"); fi
if [ -n "${EPISODE_KIND:-}" ]; then ARGS+=(--episode-kind "$EPISODE_KIND"); fi
if [ -n "${MAX_RUNS:-}" ]; then ARGS+=(--max-runs "$MAX_RUNS"); fi

echo "building $OUTPUT"
printf '  %s\n' "${ARGS[@]}"
srun "$OCTVLA_POLICY_PYTHON" -u -B "$OCTVLA_REPO/scripts/build_shelf_restock_splits.py" "${ARGS[@]}"

# Share the camera streams with any sibling variant that already holds the same
# bytes. A variant changes only the parquet -- for the r75 corpus, 161 MB of
# each 168 MB dataset is the same three streams re-encoded from the same clips
# -- and the encoder is deterministic, so the files are byte-identical. Hard
# links make that one copy, invisibly to every reader.
if [ "${DEDUPE_VIDEO:-1}" = "1" ] && [ -d "$OUTPUT/videos" ]; then
  "$OCTVLA_POLICY_PYTHON" -u -B "$OCTVLA_REPO/scripts/dedupe_dataset_videos.py" \
    "$OCTVLA_DATASET_ROOT"/*/ || echo "dedupe skipped (non-fatal)"
fi
