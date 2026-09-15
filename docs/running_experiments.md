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

`TRAIN_VARIANT` selects the **policy class**, not only the dataset:

| Variant | Policy | Reads object tokens |
| --- | --- | --- |
| `rgb` | stock `pi05`, loaded with `--policy.path` | no |
| `object` | `control_pi05` (`src/oct_vla/policies/`), loaded with `--policy.type` + `--policy.pretrained_path` + `--policy.discover_packages_path` | yes |

The plugin needs the three-flag form because its type only enters LeRobot's
registry once its package is imported; `--policy.path` would take the type from
the base checkpoint and silently give a second stock pi0.5. That failure is
worth naming, because it does not look like a failure: the object arm would
train to a healthy loss on a dataset that merely *contains* token columns and
report that object conditioning does not help, having never been enabled.

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

### Verifying the conditioning is actually live

`ControlPI05Pytorch.object_injection` is zero-initialised, so the object path
contributes nothing at step 0 by construction — and contributes nothing ever if
the tokens fail to arrive, since the policy reads them with `batch.get(...)`
and a missing key is `None` rather than an error. Loss curves look identical
either way, so check the weights instead of the curve:

```bash
python -c "
from safetensors.torch import load_file
w = load_file('CHECKPOINT/pretrained_model/adapter_model.safetensors')
k = [x for x in w if 'object_injection.weight' in x][0]
print(k, 'max|w| =', w[k].float().abs().max().item())"
```

Non-zero means gradients reached the injection, which can only happen if real
tokens flowed through the object expert. Zero after training means the
conditioning never ran. Verified on a 6-step run: `9.36e-05`, with all 13
object-module tensors present in the checkpoint and its saved type recorded as
`control_pi05`.

One shape trap sits between training and evaluation. LeRobot's preprocessor
adds a batch dimension to the features declared on the policy config, and the
object tokens are not among them — they pass through untouched. Under training
the dataloader has already collated, so this is invisible; at evaluation a live
observation arrives as `observation.state (1,16)` alongside
`observation.object_tokens (8,15)`, and the object expert rejects the rank-2
tensor. `ControlPI05Policy` therefore batches them itself. A policy that
trains for a day and then dies on its first eval step is the failure this
avoids.

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

### Validating the bridge

```bash
$OCTVLA_ROBOTWIN_PYTHON scripts/validate_eval_bridge.py \
  --robotwin-root $OCTVLA_ROBOTWIN_ROOT --seed 1000 --profile three_object
```

Run this on a GPU node after any change to `ik()`, the step loop or the frame
conventions, and before trusting a success rate. It drives the real socket,
protocol and server with **scripted** actions rather than a policy — deliberately,
because a policy rollout cannot distinguish "the controller is wrong" from "the
policy is untrained". It checks that a zero-displacement action does not move the
arm, that a commanded translation is achieved in the right direction and
magnitude, that the gripper reaches an absolute target, and that the simulation
clock advances at the control rate.

Current result, all three profiles: 0.0 mm hold drift, 120.3 mm achieved against
120 mm commanded, 3.3 mm lateral, 1.132 s over 17 control steps (exactly
17 / 15 Hz).

Two defects it caught, both invisible to unit tests and neither of which would
have crashed an evaluation:

* cuRobo solves over the whole robot model and returns **nine** joint values for
  a Panda — seven arm joints plus two gripper fingers — where `set_arm_joints`
  wants the planner's seven. `ik()` therefore selects by joint *name*; taking the
  first seven would work here and break silently on any differently-ordered
  embodiment.
* `set_arm_joints` drives a position target **and a velocity target**. Commanding
  zero velocity asks the arm to arrive at rest, so within one 15 Hz step it
  decelerates and covers roughly a fifth of the commanded displacement. Because
  each step rebuilds its target from the freshly measured pose, that never
  accumulates into a visible lag -- it just rescales every action, and a policy
  replaying its own training actions would crawl. The server now commands the
  velocity that covers the gap in one step.

## LoRA adapter verification

Carried over from the RGB baseline work, because the same adapter machinery
backs both variants.

| Requirement | Status | Evidence |
| --- | --- | --- |
| LoRA parameter selection | Verified | 1,287,168 / 4,144,691,984 trainable (0.031%), all on `gemma_expert.*.self_attn.{q,v}_proj` — pi0.5's documented default targets. |
| Base behaviour preserved at init | Verified at the layer level | The same fixed input through one target `q_proj` before and after `wrap_with_peft` matches bit-exactly, and `lora_B` is all zero. Every touched layer is provably an identity wrapper at init, so the whole model is, by composition. |
| Two-episode overfit | Verified | Loss 0.451 → ~0.09 over 1500 steps. Needed the LR raised to `1e-4`; at pi0.5's default `2.5e-5` the loss oscillates flat and looks broken when it is merely slow. |
| Checkpoint round trip | Verified | A `control_pi05` checkpoint reloads through `PreTrainedConfig.from_pretrained` and runs single-observation inference (`select_action → (1,14)`). |

**A caveat worth keeping.** An earlier attempt to verify "preservation at init"
compared `base_policy.select_action(...)` against `wrapped.select_action(...)`
end to end and found a ~4.7% action difference, stable across dtypes (0.0453
bf16, 0.0446 fp32), which rules out numerical precision. That is **not**
evidence the adapter changes behaviour: `wrap_with_peft` mutates the policy in
place rather than returning an independent copy, so the two calls dispatch
through overlapping state, with two `torch.manual_seed()` resets around a
stochastic 10-step flow-matching sampler — not a controlled comparison. The
layer-level check isolates the actual question and is unambiguous. Recorded so
the number is not rediscovered and mistaken for a real discrepancy.
