---
license: mit
task_categories:
- robotics
tags:
- lerobot
- robotwin
- object-centric
- imitation-learning
size_categories:
- n<1K
---

# OCT-VLA shelf restock — unified, with entity tokens

Dual-arm Panda restocking a shelf in RoboTwin/SAPIEN. 327 episodes, recorded by
a scripted oracle, exported once in a **unified** form: the canonical
observation/action pair plus every alternative action encoding under an `alt.`
prefix, so a single corpus serves every control-space experiment.

## Why unified

LeRobot binds training to the columns literally named `observation.state` and
`action`. Storing one dataset per encoding meant eight near-identical copies
differing only in a few megabytes of parquet. Here the alternatives ride along
under `alt.`, which LeRobot's feature typing ignores, and
`scripts/project_dataset_view.py` promotes the pair a given run wants into a
view whose video files are hard links back to this copy.

| column | meaning |
| --- | --- |
| `observation.state` / `action` | absolute joint targets, 16-d (7 joints + gripper per arm) |
| `alt.joint_action_delta` | joint increments |
| `alt.eef_state` / `alt.eef_action_abs` | absolute end-effector pose, 16-d |
| `alt.eef_action_delta` | end-effector increments, 14-d |
| `alt.joint_state_velocity` | position + velocity state |
| `observation.entity_tokens` | 16 × 17 ground-truth entity tokens |
| `observation.entity_mask` | which slots are occupied |
| `observation.images.{head,left_wrist,right_wrist}` | three camera streams |

## Entity tokens

17 columns per entity: workcell position (3), the first two rotation-matrix
columns (6), size (3), gripper aperture (1), and a four-way type one-hot
(movable / left gripper / right gripper / support). Grippers and shelf decks are
entities in the same set as the objects.

Deliberately **no object IDs, no array-slot features, and no task-role labels**.
Roles are derived from the task context against track IDs, never stored on an
object, so a policy must find the target from the instruction and the scene
rather than read it off a field.

Values are **raw on disk**. Normalisation statistics are fitted on the training
partition alone and travel in the policy config, which is what
`meta/info.json → entity_tokens.normalization` records. Do not renormalise with
whole-dataset statistics: this repo contains both splits.

## Splits

`split_manifest.json` carries the exact episode ranges. Episodes are ordered
train-then-validation so LeRobot's positional `eval_split` reproduces the
reserved seeds exactly. Seed blocks are fixed in advance and a seed belongs to
exactly one split, so a scene can never migrate after the fact.

## Object identity

Every object is the asset `113_coffee-box`, but the corpus contains **six of its
seven mesh variants**, one per episode and never mixed within a scene
(81/72/69/63/24/18 episodes). An asset variant's size is a property of its mesh,
so identity inverts exactly from the recorded geometry — see
`src/oct_vla/data/entity_identity.py` in the source repository.

Note the consequence: in this corpus geometry *is* identity, so a semantic
channel derived from it carries no information the geometry lacks.

## Provenance and caveats

- Gripper actions are thresholded to `{0, 1}` at export (`gripper_encoding:
  binary_command`), not at execution.
- The oracle replays at 19/20 on held-out scenes; treat that as the ceiling.
- Trained policies on this corpus are **near zero** at the task. The best
  configuration scored 6/20 on one training seed and 0/20 on two others, so
  this is a dataset for studying representation, not a solved benchmark.

Full experimental record: `docs/project_status_2026_09_22.md` in the source
repository.
