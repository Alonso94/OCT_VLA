#!/usr/bin/env bash
# BimanualPFN stage 3 data: for one run budget, the two-rule colour corpus (unified),
# its eeabs view, and the pose-window (PFN) and support-table (direct-demo) banks.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
set -euo pipefail
: "${OCTVLA_REPO:?}"; : "${OCTVLA_POLICY_PYTHON:?}"; : "${OCTVLA_DATASET_ROOT:?}"
: "${BPFN_BUDGET:?run budget (layout families)}"; : "${BPFN_COLLECT_ROOT:?}"; : "${BPFN_PFN_RUN:?}"
export PYTHONPATH="$OCTVLA_REPO/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
[ -n "${OCTVLA_TORCHCODEC_LIBS:-}" ] && export LD_LIBRARY_PATH="$OCTVLA_TORCHCODEC_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
C="$BPFN_COLLECT_ROOT/colour"
NAME="bpfn_colour_r${BPFN_BUDGET}"
U="$OCTVLA_DATASET_ROOT/${NAME}_unified"; V="$OCTVLA_DATASET_ROOT/${NAME}_eeabs"
cd "$OCTVLA_REPO"
if [ ! -d "$U" ]; then
  srun "$OCTVLA_POLICY_PYTHON" -u -B scripts/build_shelf_restock_splits.py \
    --canonical-root "restock_red=$C/restock_red/full_run/four_object" \
    --canonical-root "restock_blue=$C/restock_blue/full_run/four_object" \
    --output "$U" --repo-id "local/oct-vla-$NAME" --profile bpfn_colour \
    --control-space unified --gripper-encoding binary_command --entity-tokens \
    --max-entities 16 --episode-kind full_run --generic-instruction --max-runs "$BPFN_BUDGET"
fi
[ -d "$V" ] || srun "$OCTVLA_POLICY_PYTHON" -u scripts/project_dataset_view.py --unified "$U" \
  --output "$V" --control-space cartesian_absolute
SUP=(--supports "restock_red=$C/restock_red/full_run/four_object"
     --supports "restock_blue=$C/restock_blue/full_run/four_object")
[ -d "${V}_pw4" ] || srun "$OCTVLA_POLICY_PYTHON" -u scripts/bpfn/stage3_build_views.py --source "$V" \
  --kind pose_window --run "$BPFN_PFN_RUN" "${SUP[@]}" --bank 4 --device cuda
[ -d "${V}_st4" ] || srun "$OCTVLA_POLICY_PYTHON" -u scripts/bpfn/stage3_build_views.py --source "$V" \
  --kind support_table "${SUP[@]}" --bank 4
for v in "$V" "${V}_pw4" "${V}_st4"; do
  "$OCTVLA_POLICY_PYTHON" scripts/entity_training_args.py "$v" > /dev/null
done
echo "views ready for budget $BPFN_BUDGET"
