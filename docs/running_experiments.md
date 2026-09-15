# Running the experiments

End to end: collect demonstrations, build the split, finetune pi0.5 with LoRA on
both conditioning variants, then score the policies closed-loop in simulation.
Environment setup is in [setup_nhr_alex.md](setup_nhr_alex.md); collection
mechanics are in [slurm_collection.md](slurm_collection.md).

The comparison this supports is RGB-only versus object-conditioned pi0.5. Both
arms are trained from the *same* canonical clips, exported twice, so the only
difference between them is the presence of object tokens.

## 0. Environment

Every command below assumes a sourced environment file defining `OCTVLA_REPO`,
`OCTVLA_ROBOTWIN_ROOT`, `OCTVLA_ROBOTWIN_PYTHON`, `OCTVLA_POLICY_PYTHON`,
`OCTVLA_COLLECTION_ROOT`, `OCTVLA_DATASET_ROOT`, `OCTVLA_OUTPUT_ROOT`, `HF_HOME`,
`SLURM_ACCOUNT`, `SLURM_PARTITION`, and `LD_LIBRARY_PATH` for the torchcodec
libraries. Keep it outside the repository — it is machine-specific by nature.

## 1. Collect

One feasibility seed, reviewed, before any range (see
[slurm_collection.md](slurm_collection.md)). Then the reserved block for the
split being collected:

```bash
./slurm/submit_shelf_restock_collection.sh three_object a40 100-159   # train
./slurm/submit_shelf_restock_collection.sh three_object a40 200-219   # IID validation
```

Do not pass `--finalize` here. It chains the exporter with
`--dependency=afterok`, which requires *every* array task to succeed; a single
infrastructure `NODE_FAIL` then leaves the exporter permanently in
`DependencyNeverSatisfied`. Run step 2 explicitly instead.

Check yield before moving on — roughly half of seeds are discarded by design:

```bash
grep -l '"status": "ok"' $OCTVLA_COLLECTION_ROOT/three_object/seed_*/collection_report.json | wc -l
```

## 2. Build the split

Collection is split-agnostic: every seed of a profile lands under one canonical
root, and the split is recovered afterwards from each episode's seed.

```bash
$OCTVLA_POLICY_PYTHON scripts/build_shelf_restock_splits.py \
  --canonical-root $OCTVLA_COLLECTION_ROOT/three_object \
  --output $OCTVLA_DATASET_ROOT/three_object_rgb \
  --repo-id local/oct-vla-shelf-restock-three_object-rgb --profile three_object
# and again with --object-tokens into three_object_object
```

This exports train episodes first, then validation, each in ascending seed order,
and derives the `eval_split` fraction that makes LeRobot's **positional**
hold-out (`make_train_eval_datasets` keeps the last `ceil(n * eval_split)`
episodes per task — it does not read seeds) select exactly the reserved
validation block. The fraction is written to `split_manifest.json` beside the
dataset; the training job reads it from there rather than restating it, so the
two cannot drift. A boundary that does not fall on the train/validation seed
division is a hard error.

Confirm the reported boundary straddles the two blocks, e.g.
`episode 89 (seed 159) | episode 90 (seed 202)`.

## 3. Finetune

```bash
TRAIN_VARIANT=rgb BATCH_SIZE=16 TRAIN_STEPS=20000 \
sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION --gres=gpu:a40:1 \
       --time=24:00:00 --export=ALL slurm/train_pi05_shelf_restock.sbatch
# then the same with TRAIN_VARIANT=object
```

Train both variants together so everything except the conditioning is identical.

Notes that are easy to get wrong:

* **Camera names.** pi0.5 was pretrained with openpi's `base_0_rgb`,
  `left_wrist_0_rgb`, `right_wrist_0_rgb`; this dataset uses `head`,
  `left_wrist`, `right_wrist`. The job passes a `--rename_map`. LeRobot
  hard-fails on a mismatch rather than guessing, which is the good case — but
  **the same map must be used at evaluation**, where a wrong mapping would
  silently feed the policy a wrist view as its base view.
* **Memory.** Measured on one A40 (49 GB): batch 2 uses 10.3 GB, of which ~8.2 GB
  is weights, so each additional sample costs roughly 1 GB with gradient
  checkpointing on. Batch 16 fits comfortably.
* **LoRA is attached** when the log reports on the order of 1.3M learnable
  parameters against 4.1B total. A number near 4.1B means it is not.
* Validation loss appears as `eval_loss` at `--eval_steps` intervals; it is
  computed on the held-out reserved seeds, not a random tail.

Smoke-test any change to the training path with `TRAIN_STEPS=6` first. Every
failure mode found so far surfaced within six steps, and two of them —
`push_to_hub` demanding a Hub repo id, and the transformers pin silently
skipping the pretrained vision tower — would otherwise have cost a full run.

## 4. Evaluate closed-loop

Test is a simulation rollout, not a held-out dataset: the same policy is scored
on the two-, three- and four-object profiles, so a change can be attributed to
scene density rather than to the policy or its conditioning.

The simulator and the policy cannot share one interpreter (NumPy 1.26/Torch 2.4
against NumPy 2.x/Torch 2.11), so the simulator runs as a server in its own
environment and the policy drives it as a client. `slurm/eval_shelf_restock.sbatch`
starts both in one job, waits on the server's ready file rather than sleeping,
and kills the simulator on exit so it cannot outlive the job holding a GPU
context.

```bash
EVAL_CHECKPOINT=$OCTVLA_OUTPUT_ROOT/pi05_lora_three_object_rgb_<jobid>/checkpoints/last/pretrained_model \
EVAL_VARIANT=rgb EVAL_SEEDS=800-829 \
sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION --gres=gpu:a40:1 \
       --export=ALL slurm/eval_shelf_restock.sbatch
```

Evaluation seeds must come from outside every reserved block in
[dataset_protocol.md](dataset_protocol.md#reserved-seed-ranges), so that
closed-loop scenes were never trained on.

Results are written as JSON: per-episode outcome plus a per-profile summary of
success rate and mean transfers completed. Mean transfers is the partial-credit
signal — a policy that restocks two of three objects is distinguishable from one
that never grasps anything, which a binary success rate hides.

Run evaluation **after** training, never alongside it: SAPIEN and the policy
would contend for the same GPU.

### Bridge shape

`oct_vla.serve` is three modules. `protocol` is the only one both environments
import, so it is pure standard library — RGB travels as raw bytes with its shape
declared in a JSON header. `server` imports RoboTwin; `client` never does.

The server reuses `tasks.shelf_restock.collect`'s estimator, manager and success
check rather than reimplementing them: a policy has to be scored against the same
definition of "restocked" that labelled its training data.

Per-step execution uses `RoboTwinNativePort.ik()` — cuRobo IK servoing seeded
from the current joint configuration — not motion planning. A policy emits a
small end-effector increment per control step; planning a fresh trajectory each
step would be far too slow and wrong in kind, since the planner is free to reach
the target along any path.

> **Not yet validated against a live simulator.** The transport and client are
> covered by tests, but `ik()` and the server step loop have not run against
> SAPIEN. Exercise them with a scripted action sequence on a GPU node before
> treating any success rate as a measurement.
