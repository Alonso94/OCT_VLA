# Data and evaluation protocol

What the data is, how it is split, what a policy sees, and how a policy is
scored. The method that consumes the entity set is `docs/method.md`; how to run
each step is `docs/reproducibility.md`; results are `docs/research_questions.md`.
When this file and the code disagree, the code wins. Fix this file.

## 1. Task and profiles

Dual-arm Franka Panda, RoboTwin/SAPIEN. Move every object from the lower shelf
to the upper shelf. The target is always the **leftmost** remaining object on
the lower shelf (smallest workcell x, `_select_leftmost` in
`src/oct_vla/tasks/shelf_restock/manager.py`), and the instruction says so:

> Restock the leftmost object from the lower shelf to the upper shelf, then
> compact it toward the previous neighbor if one exists.

| profile | objects | task class | use |
| --- | ---: | --- | --- |
| `two_object` | 2 | `ShelfRestockTwoObjectTask` | count-shift evaluation |
| `three_object` | 3 | `ShelfRestockTask` | training data, main evaluation |
| `four_object` | 4 | `ShelfRestockFourObjectTask` | count-shift evaluation |

All three profiles share one geometry, `DEFAULT_SPEC` in
`src/oct_vla/tasks/shelf_restock/spec.py`. They differ only in object count, so
a count-shift result is attributable to count alone. The spawn span
(x ∈ [−0.49, −0.02], y ∈ [−0.28, −0.20], yaw ±0.4 rad) is sized for four
objects at `MIN_OBJECT_SEPARATION = 0.15` m. Do not give a profile its own spawn
region.

Every object is the asset `113_coffee-box`. It has seven meshes (`model_id`
0–6), and a scene uses one mesh for all its objects.

### What counts as success (evaluation)

All of these are computed in the server from the scene, using the same
`ShelfRestockManager` that labelled the data (`src/oct_vla/serve/server.py`,
`_report`):

| measure | definition |
| --- | --- |
| `success` | `is_restocked`: every object in the scene is inside `upper_shelf.contains()` (deck xy footprint, z in [top − 0.02, top + 0.15] m). Emptying the lower shelf is **not** enough; objects on the floor do not count. Compaction is not part of evaluation success. |
| `transfers_completed` | objects currently on the upper shelf, at episode end |
| `objects_lifted` | objects ever raised more than `LIFT_CLEARANCE = 0.08` m above the lower deck (accumulated over the episode) |
| `infeasible_steps` | steps where IK could not reach an arm's command; that arm holds. Always 0 in joint space |

An episode ends on success or at the step limit. Collection uses a stricter
per-transfer check (`success.py:check_placement`: on the upper shelf and, if a
neighbour exists, within `compaction_distance = 0.04` m of it).

## 2. Collection

The scripted oracle (`src/oct_vla/tasks/shelf_restock/oracle/`) plays one
continuous run per scene seed, and the recorder samples it at 15 Hz
(`DEFAULT_HZ`). `collect_episode` in `tasks/shelf_restock/collect.py` is
all-or-nothing: if any transfer fails to place its target, or any clip fails
validation, the whole seed is discarded.

**Runs and clips.** A **run** is one scene played to the end. By default it is
stored as its **atomic clips**, one per transfer (`episode_kind =
atomic_restock`), each keeping only its own frames; the first clip has no
previous neighbour, later ones do. A successful three-object run gives exactly
three clips. `EPISODE_KIND=full_run` stores the uncut run instead, under
`$OCTVLA_COLLECTION_ROOT/full_run/<profile>/`. `EPISODE_KIND=paired` writes
both from the **same** oracle execution into one seed directory (the full run
as `episode_<ssss>_full`), so an atomic and a full-run export differ only in
episode boundaries; the builder then selects one with `--episode-kind` and
refuses a mixed corpus without it. Re-collecting a seed does not reproduce its
trajectory (cuRobo plans stochastically), which is why this exists. Training
budgets count runs, not clips (`--max-runs`).

**Full runs are the training default from 2026-09-24.** On the paired corpus,
plain RGB ACT trained on full runs beats the same runs cut into atomic clips
on every training seed (research_questions.md §4.6).

**Identity partition.** Collection records each scene's mesh variants
(`metadata.model_ids`). The builder holds out a *declared* set
(`HOLDOUT_MODEL_IDS=0,5,6`, the evaluation's held-out and novel tiers) rather
than the rarest geometries (`IDENTITY_HOLDOUT=N`, kept for old corpora), since
a recollection can change which geometries are rarest.

**Count profiles.** `two_object` draws a three-object layout and drops one of
the later objects (`layout = "nested_in_3"`), so its first target is placed
exactly as in training; independently drawn, it reached 15 cm outside that
support. `four_object` is drawn directly and stays inside it.

**On disk (canonical).** Each clip is saved as
`$OCTVLA_COLLECTION_ROOT/<profile>/seed_<s>/episode_<ssss>_<i>/`. It holds
`episode.json` (samples, metadata, scene, context) and gzipped raw RGB for the
head, left-wrist and right-wrist cameras (`src/oct_vla/data/store.py`). Each
seed directory has a `collection_report.json`, with one entry per clip
(`"status": "ok"`) or one for a discarded seed (`"status": "discarded"`).
Count yield on `"ok"`.

Clip metadata includes `seed`, `hz`, `dt`, `task_profile`, `object_count`,
`episode_kind`, `transfer_index`, `transfers_in_run`, `target_track_id`,
`previous_neighbor_track_id` and `compacted`. `validate_episode`
(`data/episode.py`) rejects a clip that has fewer than 2 samples, non-increasing
or irregular timestamps, a non-finite action, or cameras of mixed size.

**Seeds.** One Slurm array task collects one seed; the array index *is* the
seed. A discarded seed is spent: never retry it (the failure is deterministic),
extend into the unused tail of the same block instead. Seeds are taken in
ascending order within a block, so a budget of N runs is always a subset of a
budget of N + 1. Keep reserved blocks below 1000, so an array index can be the
seed on any cluster.

### Reserved seed blocks

These blocks are the ones the code uses (`SPLIT_BLOCKS` in
`scripts/build_shelf_restock_splits.py`). A seed belongs to exactly one split,
and that is fixed before collection.

| profile | split | seeds |
| --- | --- | --- |
| `three_object` | train | 100–199, 350–399 |
| `three_object` | validation | 200–249 |
| `two_object` | validation | 400–449 |
| `four_object` | validation | 600–649 |

The builder ignores clips from any other seed and prints a note when it does.
Three more blocks were set aside earlier as offline test blocks: 250–349,
450–549 and 650–749. No code reads them. Leave them unused rather than reusing
them.

**Evaluation scenes are seeds 800–819.** They lie outside every block above, so
no rollout scene was ever trained or validated on. Training seeds
(`TRAIN_SEED=1000, 1001, 1002`) are a separate namespace: they seed the
optimiser, not a scene.

## 3. Splits

### Train/validation by seed, encoded in episode order

LeRobot's offline validation (`make_train_eval_datasets`) never reads seeds. It
holds out the **last** `ceil(n · eval_split)` episodes of each task, and this
corpus has one task string. So `build_shelf_restock_splits.py`:

1. writes train episodes first, then validation, each in ascending seed and
   clip order;
2. checks that every episode before the boundary is a train seed and every
   episode after it is not, and fails before encoding video if either is wrong;
3. derives `eval_split` as the midpoint of the interval that selects exactly
   `n_val` episodes;
4. writes `split_manifest.json` next to the dataset.

The manifest holds `eval_split`, the episode ranges and seeds per split,
`identity_holdout`, `train_identities`, and
`entity_normalization_training_sources`. Training jobs read `eval_split` from
this file. Never restate it on the command line.

`octvla_episode_manifest.json` maps each LeRobot episode index to its canonical
clip, seed, sample count, metadata and `episode.json` SHA-256. It is the
split-audit record.

### Mixed-rule corpus (effect-conditioned VLA)

`build_shelf_restock_splits.py --canonical-root leftmost=… --canonical-root
rightmost=… --generic-instruction` exports several rules' collections as one corpus
(`three_object_v4lr_*`):

- each clip's recorded rule must match its root's label;
- the split is by seed block, so a scene collected under both rules (18 of the v4
  scenes: 14 train, 4 validation) lands on the same side for both;
- every frame's task is `spec.GENERIC_INSTRUCTION`, which names no rule. The export is
  refused unless it holds exactly one task string. The rule stays in each episode's
  metadata; evaluation sends the same string (the view's `split_manifest.json →
  task_text`), never the scene's rule instruction.

### Identity holdout (shelf restock)

The seed split holds out *scenes*. `--identity-holdout N` also holds out whole
*objects*: the N rarest meshes are removed from both splits and recorded, not
exported. The current corpus uses `N = 2`.

| tier | `model_id`s | status |
| --- | --- | --- |
| seen | 1, 2, 3, 4 | in the training set |
| held-out | 0, 6 | collected, excluded from the dataset entirely |
| novel | 5 | in zero collected runs (the oracle cannot plan it) |

The tier table is defined twice and must stay identical: `TIER` in
`slurm/submit_final_experiments.sh` and `slurm/submit_vla_experiments.sh`, and
`SHELF_TIER_IDS` in `scripts/collect_final_results.py`.

**In this corpus, geometry is identity.** The meshes differ only in size, so a
mesh is recovered exactly from its recorded `size_xyz`
(`src/oct_vla/data/entity_identity.py`, 4-decimal match), and no scene mixes two
meshes. So "held-out identity" means "an object size never seen in training",
and a semantic channel derived from geometry adds nothing that geometry does
not already carry.

### Category holdout (`place_container_plate`)

This is the RoboTwin built-in task (`src/oct_vla/tasks/robotwin_builtin/`,
`scripts/collect_robotwin_builtin.py`, `scripts/export_robotwin_builtin.py`).
The container is either `002_bowl` or `021_cup`. `021_cup` is held out as a
whole category (`--holdout-category`). Validation is the last 20 % of the kept
episodes by seed (`--val-fraction 0.2`). There are no reserved seed blocks;
collection used seeds 0–249.

The built-in captures one frame every 17 control steps at dt = 1/250 s
(14.7 Hz), and the dataset declares 15 fps. The true rate is recorded as
`capture_save_freq` / `capture_hz` in `split_manifest.json`.

## 4. Unified export and control-space views

A single export carries every action encoding (`--control-space unified`,
`src/oct_vla/data/lerobot_export.py`). LeRobot treats only `observation*` and
`action*` columns as policy features, so the alternative encodings are stored
under `alt.`, which LeRobot ignores. Never name an extra column `action.*`:
LeRobot would treat it as a second ACTION feature.

`scripts/project_dataset_view.py` promotes one (state, action) pair to
`observation.state` / `action`, drops every `alt.*` column, and moves
`meta/stats.json` entries **with their column**. Normalisation is keyed by
column name, so leaving the old stats in place would scale increments by
absolute-joint statistics. It also rewrites `control_space` and
`state_encoding` in `meta/info.json`, copies both manifests, and **hard-links**
`videos/` and `images/`, so a view costs only its parquet. The views are
defined in `VIEWS` in `src/oct_vla/data/control_views.py`:

| view (`control_space`, `state_encoding`) | slug | `observation.state` | `action` |
| --- | --- | --- | --- |
| `joint`, `position` (canonical) | `abs` | 16: 7 joints + gripper per arm | next frame's measured joints, 16 |
| `joint_delta`, `position` | `delta` | 16 | `alt.joint_action_delta`: joint increment, gripper absolute |
| `joint_delta`, `position_velocity` | `delta_velo` | `alt.joint_state_velocity`, 32 | `alt.joint_action_delta` |
| `joint`, `position_velocity` | `abs_velo` | `alt.joint_state_velocity`, 32 | `action` |
| `cartesian`, `position` | `eedelta` | `alt.eef_state`, 16 | `alt.eef_action_delta`, 14 (canonical action, §5) |
| `cartesian_absolute`, `position` | `eeabs` | `alt.eef_state`, 16 | `alt.eef_action_abs`: next frame's EE pose, 16 |

**Column layouts.**
- Joint (16): `left_arm.j0…j6, left_arm.gripper`, then the same for the right arm.
- `position_velocity` (32): the 16 joint columns, then each again with a `.vel`
  suffix. The velocity is the backward difference and is 0 at frame 0, which is
  also what the eval client sends after a reset.
- EE pose (16): `{left,right}_eef.{x,y,z,qx,qy,qz,qw,gripper}`.
- On the last frame of a clip, the absolute actions repeat the current state.

View directories are named `<corpus>_<slug>`, for example
`three_object_identity_eeabs`. Training and evaluation derive the LeRobot repo
id from that name as `local/oct-vla-shelf-restock-${NAME#three_object_}`.

**Gripper: `binary_command`.** Every current export thresholds the gripper
*action* to {0, 1} at export, using the same constant as the bridge
(`GRIPPER_OPEN_THRESHOLD = 0.822522`, `src/oct_vla/core/gripper.py`). At
execution the command is decoded as `≥ 0.5 → open`. `observation.state` keeps
the measured aperture. `meta/info.json → gripper_encoding` records which
convention a dataset uses; if the field is absent, the dataset holds the raw
measurement. The builder's default is `measured_aperture`, so pass
`--gripper-encoding binary_command` explicitly.

**How the server executes each view** (`src/oct_vla/serve/server.py`):

| view | execution |
| --- | --- |
| absolute joint | commanded directly; cannot be infeasible |
| joint delta | increment added to the *measured* joints |
| absolute EE | IK every step; the quaternion is normalised; an infeasible target holds that arm |
| EE delta | integrated into a command reference, leashed to 5 cm (`COMMAND_REFERENCE_LEASH`) from the measured pose, then IK |

Evaluation reads `control_space`, `state_encoding` and `gripper_encoding` from
the dataset (`EVAL_DATASET`). That dataset must be the view the checkpoint was
trained on.

## 5. Canonical action, frames, timing

- **Frame.** Every pose the policy sees is in `WORKCELL_FRAME` (`src/oct_vla/core/frames.py`).
  For this task the workcell frame is the simulator world frame: `WORLD_TO_WORKCELL`
  in `tasks/shelf_restock/collect.py` is the identity, named so that there is one
  place to change it.
- **Conventions.** Metres, right-handed axes, quaternions **xyzw** (active,
  sign-canonicalised to w ≥ 0). `Transform(source="base", target="workcell")` is
  `T_workcell_base`. `a.compose(b)` applies b first.
- **Canonical action** (`src/oct_vla/core/action.py`, `ACTION_DIM = 14`): per
  arm, `[dx, dy, dz, drx, dry, drz, gripper]`, left arm then right. These are
  per-step displacements, not velocities, so never multiply them by dt.
  - `dp = p_next − p_t`
  - `dr = Log(R_next R_tᵀ)` (a spatial rotation vector, composed on the left)
  - gripper: an absolute target in [0, 1], where 0 is closed.
- **Helpers.** `action_between` and `apply_action` are the only conversion
  functions, used both for export and at execution. `Action.hold(state)` keeps
  both grippers where they are. A vector of all zeros instead commands both
  grippers **closed**.
- **Timing.**
  - Canonical samples carry simulator-clock timestamps (`tick · dt`).
  - LeRobot timestamps are `frame_index / fps`, with fps = 15.
  - One evaluation step is one 15 Hz control step, so 600 steps are 40 s of
    simulated time.

## 6. Entity token contract

The only object representation is the entity set, `oct-vla-entity-tokens-v2`
(`src/oct_vla/data/entity_tokens.py`, read by
`src/oct_vla/policies/conditioning/entity.py`).

| feature | shape | dtype |
| --- | --- | --- |
| `observation.entity_tokens` | `[N, 17]`, N = 16 | float32, raw |
| `observation.entity_mask` | `[N]` | float32 on disk, bool at inference |

| columns | content |
| --- | --- |
| 0–2 | position, workcell frame, metres |
| 3–8 | rotation, 6D (first two columns of the rotation matrix) |
| 9–11 | size, metres (0 for grippers) |
| 12 | gripper aperture (0 for non-grippers) |
| 13–16 | type one-hot: movable, left gripper, right gripper, support |

**Entities.** The rows are the objects, then the two grippers, then the
supports: the two shelf decks, from `shelf_support_entities(DEFAULT_SPEC)`.
The built-in task supplies its own supports. Movable objects and supports are
ordered by geometry alone, and padding rows are zero with mask 0. A
three-object scene fills 7 of 16 rows. The deck geometry is also recorded in
`meta/info.json → entity_tokens.supports`. At evaluation the server sends the
supports, and the client rebuilds the tokens with the same
`build_entity_tokens`.

**No leakage.** A token has no object id, no track id, no slot index and no
task role. The policy must find the target from the instruction and the scene.

**Normalisation.**
- Values on disk are raw (`raw_on_disk: true`).
- The builder fits position and size mean/std on the **training split only**
  (`EntityTokenNormalizer.fit`; size uses only entities that have a size). It
  stores them in `meta/info.json → entity_tokens.normalization` and lists the
  source clips in the split manifest.
- `scripts/entity_training_args.py` validates schema, width, capacity,
  `raw_on_disk` and the statistics, then emits `--policy.object_max_entities`
  and `--policy.object_entity_normalizer`. The statistics therefore live in the
  policy config and travel with the checkpoint. Never renormalise with
  whole-dataset statistics.

A v3 schema (32 columns) exists in code but is not exported. Training refuses
any dataset that is not v2.

### Object-effect rows (`<view>_effA`, `_effB`)

`scripts/add_effects.py` fills `observation.future_keyframe` with the **next object
effect** Z = (object, desired pose) in the goal-only 3 × 24 layout a goal-only prior
writes: row 0 the object at its current pose, row 2 its goal, both re-posed upright
with the token's yaw; gripper row masked, event empty. The labels
(`oct_vla.data.effects`) read object poses, EEF positions and the measured aperture
only; no phase label, planner or rule. Per transfer the timeline is grasp, lift,
place, release, let_go, own push, settled. Release is the start of the opening, not a
threshold: the aperture opens over 10–30 frames and a large object is held at
0.78–0.83, so a fixed threshold fires mid-carry.

Two contracts are fixed before any training:

| | Z switches to the next object | Goal |
|---|---|---|
| **A** (primary) | at let_go: placed, mover beyond 0.20 m (the keyframe oracle's own test) | the pose at let_go |
| **B** (ablation) | after the object's own push has settled | the pose at the end of its push |

An object's own push is compaction that starts before the next object is lifted; a
later neighbour's push that nudges it does not count. Stillness never ends a segment:
objects wait a median 62–69 frames between release and their push.

Online, `oracle_effect[:rule]` (`keyframes.ShelfEffectOracle`) serves contract A. It
plans the next placement beside the previous object's *compacted* pose while that
push is pending (`anticipate_compaction`). Without that, the goal served during
compaction sat a median 5.2 cm from where the object was later placed; with it,
0.7–0.9 cm (0.5–0.6 cm outside compaction). `add_effects.py` refuses an episode
unless the served rows and the hindsight labels agree: same object on ≥ 98 % of
frames, switches within 5 frames, row 0 identical, and each transfer's goal within
5 cm; the corpus is refused if the median over all transfers exceeds 2 cm. The goal
bounds catch systematic errors, not placement noise: labels are where the object came
to rest, the oracle serves where it plans it, and objects shift in the gripper. On the
v4 runs the per-transfer gap is a median 0.6 cm, 28 of 285 above 2 cm, max 4.3 cm,
correlated 0.50 with the in-gripper shift (`diagnostics/effect_goal_check_v4lr_A.json`).
Object identity and switch timing, the effect's discrete content, stay strict.

On the v4 leftmost (52) and rightmost (43) runs
(`diagnostics/interaction_audit.json`), contract A matches the oracle's target on
≥ 99.7 % of frames and its switch frame on every transfer.

## 7. Evaluation protocol

| setting | value |
| --- | --- |
| scene seeds | 800–819 (20 scenes) |
| profiles and step budget | `three_object`, 600 steps; `two_object` and `four_object`, 200 steps per object (400 / 800) |
| execution horizon | `n_action_steps = 25`, set at training and again at evaluation |
| checkpoint | `checkpoints/last` only (validation loss is never used to choose one) |
| training seeds | ≥ 3 per cell (1000, 1001, 1002) |

**Pinned identity tiers.**
- Each tier is a separate rollout job with `EVAL_MODEL_IDS` (seen, held-out, or
  novel). The count job uses seen identities only.
- The pin is sent **over the bridge with every reset**. It is not passed in the
  environment, because the server is forked before the job's variables are set.
- The server refuses a scene that spawns a mesh outside the pin, and records
  `model_ids` per episode.
- `collect_final_results.py` rejects any rollout file that lacks
  `model_ids_requested`, whose pin disagrees with the tier in its name, or
  whose episodes spawned outside the pin.

**Rollout naming.** Each rollout is written to
`$OCTVLA_OUTPUT_ROOT/eval/<prefix>-<arm>-s<train seed>-<seen|heldout|novel|count>.json`.
- ACT prefixes are one letter for the regime: `F` absolute EE, `J` absolute
  joint, `D` joint delta, `X` EE delta.
- VLA prefixes add the backbone first: `P` pi0.5, `S` SmolVLA, `G` GR00T, as in
  `PF`.
- Arm names use the current names (`rgb`, `rgb_cont`, `kv`, `kv_adaln`,
  `kv_tokens`, `scratch_kv`). Older files carry the old names, `entity`,
  `adaln`, `incontext` and `scratch`, and the collector maps them.

**Scoring and statistics** (`scripts/collect_final_results.py`).
- Per episode: success, ≥ 1 transfer, ≥ 1 lift, transfers.
- Per cell: counts over 20 episodes, and mean transfers. The spread shown is the
  **min–max across training seeds**, not a confidence interval.
- The collector prints a warning for any cell with fewer than two seeds and
  never presents one as a result. The protocol asks for three seeds.
- Each conditioned arm is compared with `rgb` and `rgb_cont` by **paired exact
  McNemar**. Pairs are matched on (training seed, tier, profile, scene seed);
  each stage-2 arm starts from the same seed's `rgb` checkpoint and faces the
  same scenes.
- Mechanism contrasts: kv vs kv_adaln, kv vs kv_tokens, kv_adaln vs kv_tokens.
- Many contrasts are run, so treat p < 0.05 as a lead, not a finding.
- Tiers are reported separately only if some arm's held-out or novel score
  moves the same way against its seen score on every training seed. Otherwise
  they are pooled, and the table says so. Object count is always reported.
- Read each score against the cell's own oracle ceiling, not against 100 %.
  The replay gates `scripts/replay_oracle_episode.py` and
  `scripts/replay_oracle_joints.py` measure that ceiling through the real bridge.

**Where the truth is.** Take numbers from `$OCTVLA_OUTPUT_ROOT/eval/*.json` and
from `$OCTVLA_OUTPUT_ROOT/diagnostics/*.json`, which holds collector outputs such
as `final_matrix_F.json`. Re-derive every number from these files. Do not copy
numbers from a document.

## 8. Datasets

`$OCTVLA_DATASET_ROOT` is `$HPCVAULT/octvla-datasets-leftmost` under
`~/octvla/env-leftmost.sh`. Current corpora:

| directory | kind | episodes | train / val | holdout |
| --- | --- | ---: | --- | --- |
| `three_object_unified_entity` | unified | 327 | 252 clips (84 runs) / 75 (25) | none |
| `three_object_identity_entity` | unified | 285 | 222 (74 runs) / 63 (21) | 42 clips (14 runs), `model_id` 0 and 6 |
| `three_object_identity_{abs,delta,eeabs,eedelta}` | views of the above | 285 | same | same |
| `place_container_plate_entity` | unified | 124 | 99 / 25 | 93 `021_cup` episodes |
| `place_container_plate_eeabs` | view of the above | 124 | same | same |

All current corpora are 15 fps with three 240×320 cameras, entity v2 at
capacity 16, and `binary_command` grippers. The identity corpus has
`eval_split = 0.2192982456140351`. Other directories under this root predate the
cleanup; do not train on them.

**Hub** (all private, owner `3liyounes`, uploaded with `scripts/publish_dataset.py`):

| repository | source |
| --- | --- |
| `3liyounes/oct-vla-shelf-restock-unified-entity` | `three_object_unified_entity` |
| `3liyounes/oct-vla-shelf-restock-identity-entity` | `three_object_identity_entity` |
| `3liyounes/oct-vla-place-container-plate-entity` | `place_container_plate_entity` |

`publish_dataset.py` generates the dataset card from `meta/info.json` and
`split_manifest.json`, uploads with `hf upload --private`, and refuses
`--public` unless `--i-understand-this-is-public` is also given. Publish only
unified exports: a view can be re-projected locally in minutes.
`scripts/hub_dataset.py` is the older path for immutable tagged releases. It
writes a `dataset_manifest.json` of SHA-256 hashes and can `download` a release.
The two `…-three_object-{rgb,object}` v1.0.0 repositories predate the entity
schema and the leftmost-target rule. Do not train on them.

## 9. BimanualPFN corpus (pose windows, semantic colour)

A fresh corpus under `$HPCVAULT/octvla-collection-bpfn`, never mixed with the v4 runs
(no fallback to the leftmost/rightmost roots). Collected by `slurm/submit_bpfn.sh`.

**Rules.** `restock_red` / `restock_blue` restock every object of one colour and leave
the others (distractors) where they are. The expert takes eligible objects leftmost
first; that is a tie-break, not the rule, so scoring accepts any eligible first pick.
Instructions name the colour for collection only; every policy and the PFN see
`GENERIC_INSTRUCTION`. Success (`serve/server.py`) = every eligible object on the upper
shelf **and** every distractor within 2 cm of its start; `post_success_steps` keeps an
episode running after first success so dwell and extra reaches can be scored.

**Colour.** The same `113_coffee-box` meshes, tinted at build time
(`robotwin_env._visual_material`): geometry and grasp feasibility are unchanged, so the
colour is the only difference. Configured per collection job:
- `OCTVLA_COLOURS=red,blue` (rule colour first), `OCTVLA_COLOUR_COUNTS=3,1` for a support
  with three eligible and one distractor, unset for a balanced scene;
- `OCTVLA_COLOUR_SWAP=1`: the same seed's layout with the two colours exchanged.

Colours are drawn from `random.Random("oct_vla/colour/v1/<seed>")`, never the layout
stream, so a seed's layout is identical with and without colour. The realised colour is
persisted per object (`ObjectState.colour`) through store, codec and server.

**Commands.** Full runs carry `episode.commands.npz` (`data/commands.py`): the applied
joint and gripper references of both arms per physics tick, including the held arm's
reference, with one frame tick per sample. Targets derived from them are labelled
`commanded_fk`; targets from measured poses (all legacy recordings) `next_measured_pose`.
Built-in RoboTwin tasks run RoboTwin's own planner and record `commands: none`, plus
their measured capture rate (`hz` = 250 / 17).

**Seed blocks** (disjoint from every v4 block and from evaluation scenes 800-819):

| Block | Use |
|---|---|
| 2000-2199 | colour source pool (executor training), both rules, plus swapped twins |
| 2200-2249 | colour development |
| 2250-2299 | support banks: colour supports (3 eligible + 1 distractor), count supports (3 all-eligible) |
| 2300-2399 | count query layouts (two- and four-object) |

All siblings of one layout family (a seed under both rules and its swapped twin) stay in
one split. Every attempted seed is used once; a discarded seed is not retried.

**Canary.** `submit_bpfn.sh stage1-canary` collects 2 seeds per rule, one swapped twin and one
3+1 support; `scripts/bpfn_canary_report.py` checks colour, rendering (contact sheet),
codec round trip, distractor stillness and command alignment, and writes `CANARY_PASSED`,
which the pilot stage requires.

**Stage 3 views (executors).** Per run budget N (4 < 8 < 16 layout families, nested by
ascending seed), `slurm/bpfn_views.sbatch` builds `bpfn_colour_rN_unified`, its eeabs
view, and two bank views under `$HPCVAULT/octvla-bpfn/datasets`:
- `_pw4`: per frame, four PFN windows (2H pose tokens each) from four supports drawn
  from *other* runs of the same rule (`scripts/bpfn/stage3_build_views.py --kind pose_window`);
- `_st4`: the same four supports as raw tables (both wrists over time, every object's
  start and end), the direct-demonstration baseline (`--kind support_table`).

Training takes one bank entry per sample, evaluation entry 0 with one recorded support per
rollout (`eval_shelf_restock_policy.py --pose-support`). The residual arm learns the
expert chunk minus the window in the action's normalised space (pose columns; grippers
absolute). Rollouts score each support against its own rule (`EVAL_RULE`, server
`--rule`) on resets 800-809 with one fixed palette (`OCTVLA_COLOURS=red,blue`).
