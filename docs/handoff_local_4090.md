# Handoff: continuing on a local 4090

Written for whoever (or whatever) picks this up on the workstation. The cluster
is under a maintenance reservation covering every partition and nothing will
schedule, so the work moves local. `CLAUDE.md` holds the rules that must not be
rediscovered; this file holds the state and the next actions.

## What fits on a 4090, and what does not

| workload | peak VRAM | verdict |
| --- | ---: | --- |
| ACT training, 40 k steps | **1.69 GB** | fits with 14× headroom; ~40 min |
| ACT rollout (SAPIEN + policy) | small | fits; ~1.7 min/episode |
| GR00T training | 36 GB | **does not fit** |
| layerwise pi0.5 | — | **does not fit** — cannot use gradient checkpointing (see below) |

So the 4090 runs the **ACT arms**, which is the whole final matrix. The VLA arms
stay on the cluster. That is not a loss: ACT is the only backbone that has ever
completed the task, and four VLAs produced zero successes in 240 episodes.

Sequential on one GPU the full matrix is roughly **19 h** — an overnight run —
against ~3 h parallel on the cluster.

## What to copy

Only the dataset is needed for training:

```
$OCTVLA_DATASET_ROOT/three_object_rgb_eeabs        370 MB   the working ACT dataset
$OCTVLA_DATASET_ROOT/three_object_unified_entity    61 MB   entity tokens, no identity holdout
```

The 16 GB canonical collection is only needed to **re-export**. Skip it unless
the identity-holdout dataset has to be rebuilt locally — in which case copy
`$OCTVLA_COLLECTION_ROOT` too.

Rollouts additionally need the RoboTwin checkout and its assets
(`$OCTVLA_ROBOTWIN_ROOT`, ~32 GB of meshes) plus its own Python environment.
Training alone needs neither.

## Environment on the workstation

Two interpreters, as on the cluster, and they cannot be merged:

- **policy env** — torch + LeRobot 0.6.2 + peft 0.20. Runs training, export,
  diagnostics, and the eval *client*.
- **robotwin env** — NumPy 1.26 / Torch 2.4 + SAPIEN. Runs the simulator and
  the eval *server*.

Set the same `OCTVLA_*` variables `~/octvla/env-leftmost.sh` sets, pointing at
local paths. Everything downstream reads them.

## State: what is done

- **Final matrix is queued and ready.** `slurm/submit_final_experiments.sh`
  (3 arms × 3 seeds × 3 identity tiers) and
  `scripts/collect_final_results.py`. On a workstation the sbatch wrapper is not
  usable directly — read it for the exact env vars each cell needs and run them
  in sequence.
- **653 tests pass** under the policy interpreter.
- **Object conditioning** is the published ControlVLA (per-layer attention over
  the unpooled object set, zero-init K/V), with gradient flow to the entity
  encoder verified on GPU: at step 0 only V_z receives gradient; after six steps
  all ten encoder tensors have moved; the gradient matches float64 central
  differences.
- **Identity protocol**: six meshes recovered from recorded geometry; four train,
  two held out, one never collected. `--identity-holdout 2` on the split builder.
- **Entity token v3**: 32 columns, 16 geometric / 16 semantic.

## State: what is in flight

1. **The identity dataset has not been built.** `three_object_identity_entity`
   was queued (job 4289742) and never ran. Rebuild it locally:
   ```bash
   scripts/build_shelf_restock_splits.py --entity-tokens --identity-holdout 2 \
     --control-space unified --gripper-encoding binary_command --max-entities 16 \
     --canonical-root "$OCTVLA_COLLECTION_ROOT/three_object" \
     --output "$OCTVLA_DATASET_ROOT/three_object_identity_entity" \
     --repo-id local/oct-vla-shelf-restock-identity_entity
   ```
   Needs the 16 GB collection. Expect ~1 h, mostly video encoding.
   Then project the EE-absolute view with `scripts/project_dataset_view.py` and
   check `scripts/entity_training_args.py` accepts it — that script validates
   schema, capacity, `raw_on_disk` and the fitted statistics, and is the gate.

2. **`control_act` has never seen the real dataset.** Every check so far used
   synthetic tensors with `use_vae=False`; ACT's default is `use_vae=True` and
   that path is unexercised. Smoke-test before spending training time.

3. **The easy-task adapter is half built.** Two blockers are fixed and committed
   (`735ad79`): `recording.py` no longer imports the shelf oracle, and
   `single_span` lets a demonstration that reports no motion records be
   labelled. Still to write: a subclass of `place_container_plate` exposing
   `tracked_objects` and overriding `_take_picture` to capture canonical frames,
   plus a collection loop that calls `play_once` and **checks
   `task.plan_success`**. See `.claude/plans/starry-dancing-thimble.md`.

## Order of work

1. Rebuild the identity dataset, verify with `entity_training_args.py`.
2. Smoke-test `control_act` on it with `use_vae=True`.
3. **Run the `rgb` arm on 3 seeds first.** If ACT cannot learn this, that is the
   finding and the remaining six cells should not be spent.
4. If it learns, run `entity` and `semantic`, 3 seeds each.
5. Collect with `scripts/collect_final_results.py` and read the *ranges*, not
   the means.

The easy RoboTwin task (§3 above) is parallel work and does not block any of it.

## What not to do

Everything in `CLAUDE.md` under "Rules that were learned the expensive way", and
in particular: do not gate anything on validation loss, and do not report a
single-seed cell as a result. Both have already cost this project a retraction.
