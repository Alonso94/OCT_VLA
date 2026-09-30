#!/usr/bin/env bash
# Re-run the four-object half of every ACT object-count rollout at 800 steps.
#
# Until 2026-09-30 the client never sent the step limit, so the server's
# default of 600 ended every four-object episode at 150 steps per object
# instead of 200 (docs/research_questions.md §3). The two-object half was
# unaffected. This re-scores each count cell's checkpoint on four objects
# alone, under <cell>-four.json, with everything else taken from the original
# report and the checkpoint's own train_config: checkpoint, dataset, execution
# horizon, identity pin. Nothing is guessed from a naming convention.
#
#   DRY_RUN=1 slurm/rerun_four_object.sh     # print the plan
#   PREFIXES="F CF" slurm/rerun_four_object.sh
#
# Skips pre-rename alias cells (entity, adaln, incontext, scratch: the same
# checkpoints re-scored under their current names), GR00T (submit_vla_experiments.sh
# ROLLOUTS=four), and any cell whose -four.json already exists.
set -euo pipefail
: "${OCTVLA_OUTPUT_ROOT:?source ~/octvla/env-leftmost.sh first}"
cd "$(dirname "$0")/.."
PREFIXES="${PREFIXES:-F CF AF D J X}"
EVAL="$OCTVLA_OUTPUT_ROOT/eval"
LOGS="$OCTVLA_OUTPUT_ROOT/slurm_logs"
VIDEO_STAGE="${VIDEO_STAGE:-$HPCVAULT/octvla-rollout-videos/final}"
SB=(--account="${SLURM_ACCOUNT:-g107ea}" --partition="${SLURM_PARTITION:-a40}" --gres=gpu:a40:1)

plan=$("$OCTVLA_POLICY_PYTHON" - "$EVAL" $PREFIXES <<'EOF'
import json, re, sys
from pathlib import Path
eval_dir, prefixes = Path(sys.argv[1]), set(sys.argv[2:])
aliases = {"entity", "adaln", "incontext", "scratch"}
cell = re.compile(r"^(?P<prefix>[A-Z]{1,3})-(?P<arm>[a-z_]+)-s(?P<seed>\d+)-count$")
for path in sorted(eval_dir.glob("*-count.json")):
    m = cell.match(path.stem)
    if not m or m["prefix"] not in prefixes or m["arm"] in aliases:
        continue
    tag = path.stem[: -len("count")] + "four"
    if (eval_dir / f"{tag}.json").exists():
        continue
    report = json.loads(path.read_text())
    ckpt = Path(report["checkpoint"])
    if not (ckpt / "model.safetensors").is_file():
        sys.exit(f"{path.name}: checkpoint {ckpt} is gone")
    root = Path(json.loads((ckpt / "train_config.json").read_text())["dataset"]["root"])
    if report.get("steps_per_object") != 200:
        sys.exit(f"{path.name}: steps_per_object {report.get('steps_per_object')}, expected 200")
    ids = ",".join(str(i) for i in report["model_ids_requested"])
    print(tag, ckpt, root.name, report["execution"]["n_action_steps"], ids)
EOF
)
[ -n "$plan" ] || { echo "Nothing to re-run."; exit 0; }
n=0
while read -r tag ckpt dataset exec ids; do
  cmd=(sbatch --parsable "${SB[@]}" --job-name="octvla-$tag" --time=04:00:00
       --output="$LOGS/$tag-%j.out" --export=ALL slurm/eval_shelf_restock.sbatch)
  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "$tag  dataset=$dataset exec=$exec ids=$ids  $ckpt"
  else
    job=$(EVAL_CHECKPOINT="$ckpt" EVAL_DATASET="$dataset" EVAL_SEEDS=800-819 \
      EVAL_PROFILES=four_object EVAL_STEPS_PER_OBJECT=200 EVAL_N_ACTION_STEPS="$exec" \
      EVAL_MODEL_IDS="$ids" EVAL_TAG="$tag" \
      EVAL_VIDEO_DIR="$VIDEO_STAGE/$tag" EVAL_VIDEO_LABEL="$tag" "${cmd[@]}")
    echo "$tag -> $job"
  fi
  n=$((n + 1))
done <<< "$plan"
echo; echo "$n rollouts."
