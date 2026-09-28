#!/usr/bin/env bash
# Train the LeJEPA object-crop encoder (src/oct_vla/perception/lejepa) on the
# crops written by scripts/build_object_crops.py, then probe it. One GPU; the
# whole training split (~0.4 GB of uint8 crops) is held on the device.
#
#   sbatch --account=g107ea slurm/train_lejepa.sbatch
#   LEJEPA_SEED=1 LEJEPA_STEPS=40000 sbatch --account=g107ea slurm/train_lejepa.sbatch
#
# Run three seeds before reading anything into a probe number (CLAUDE.md).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00

set -euo pipefail

: "${OCTVLA_REPO:?Set OCT_VLA checkout path}"
: "${OCTVLA_POLICY_PYTHON:?Set LeRobot Python executable}"
: "${HPCVAULT:?Set HPCVAULT; crops and runs live under \$HPCVAULT/octvla-lejepa}"

CROPS="${LEJEPA_CROPS:-$HPCVAULT/octvla-lejepa/crops_v1}"
for split in train val; do
  [ -f "$CROPS/$split/meta.json" ] || {
    echo "No $split crops at $CROPS/$split; run scripts/build_object_crops.py --split $split" >&2
    exit 2
  }
done

SEED="${LEJEPA_SEED:-0}"
STEPS="${LEJEPA_STEPS:-20000}"
BATCH="${LEJEPA_BATCH:-256}"
LAMBDA="${LEJEPA_LAMBDA:-0.05}"
RUN_NAME="lejepa_b${BATCH}_l${LAMBDA}_st${STEPS}_s${SEED}${LEJEPA_SUFFIX:+_$LEJEPA_SUFFIX}"
OUTPUT="$HPCVAULT/octvla-lejepa/runs/$RUN_NAME"
if [ -e "$OUTPUT" ]; then
  echo "Run directory already exists: $OUTPUT" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUTPUT")"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

echo "crops      : $CROPS"
echo "run        : $RUN_NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$OCTVLA_POLICY_PYTHON" -u "$OCTVLA_REPO/scripts/train_lejepa.py" \
  --crops "$CROPS" \
  --out "$OUTPUT" \
  --steps "$STEPS" \
  --batch "$BATCH" \
  --lr "${LEJEPA_LR:-5e-4}" \
  --lambda-sigreg "$LAMBDA" \
  --target-views "${LEJEPA_TARGET_VIEWS:-3}" \
  --seed "$SEED" \
  --device cuda

# One checkpoint (~12 MB), no optimiser state; see docs/reproducibility.md.
du -sh "$OUTPUT" 2>/dev/null | sed 's/^/kept       : /'
