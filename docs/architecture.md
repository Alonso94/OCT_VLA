# Architecture and contracts

This document specifies the intended architecture. Commits 1–6 implement the
package boundary, geometry, frame-labelled poses/transforms, canonical EEF state,
14-D actions, external path configuration, `doctor` discovery, the canonical
`RobotObservation`/`RobotBackend` contract, a RoboTwin backend that converts
world-frame measurements to the workcell frame with checked dual-arm planning,
the canonical object-scene schema with a RoboTwin ground-truth estimator, and
the shelf-restocking task's pure specification, geometry, success check, and
repeat-until-empty manager, a RoboTwin scene that builds this task live
(spawned objects, no oracle), and the oracle's world-model registration (the
first oracle piece: cuRobo's planner is now genuinely aware of this task's
shelf and objects). Other interfaces below remain design contracts.

## Research scope

Compare RGB + language + proprioception against the same inputs augmented with
object-centric conditioning. Start with π0.5 LoRA and approximately 25 atomic
shelf-restocking episodes. A masks/visual-regions variant follows when ready.
Keep data selection, runtime, and evaluation shared across these variants.

Each atomic episode transfers one selected object to the upper shelf and, when
a previous neighbor exists, compacts the newly placed object toward it. A thin
task manager selects targets and repeats the skill until the lower shelf is
empty. The task is sequentially bimanual; no task-wide active-arm mask is used.

## Three independent interfaces

| Interface | Responsibility | Boundary |
| --- | --- | --- |
| `RobotBackend` | Reset, observe, execute a canonical step, stop, report health | Owns robot-specific control and frame conversion; no policy or training imports |
| `ObjectStateEstimator` | Produce an `ObjectScene` from explicitly supplied evidence | Runs before the policy; GT access belongs only to the simulation estimator |
| `Policy` | Consume canonical observations, task context, and optional objects; return action chunks | No simulator, detector, or task-planning implementation |

The runtime composes these interfaces. Training reads canonical datasets and
never imports RoboTwin. Evaluation uses the deployment runtime with a configured
backend. Real hardware support will implement the same contract only after the
simulation contract is stable; no pretend hardware driver is included.

`RobotObservation` contains acquisition time, head/left-wrist/right-wrist RGB,
both EEF poses, and gripper states. Camera names, shape/dtype, units, and optional
fields will be explicit typed contracts. Library dictionaries exist at adapters,
not as the internal schema.

`PrivilegedSceneState` is separate: exact object poses, segmentation IDs, contacts,
collision geometry, and planner metadata. Only the oracle, evaluator, diagnostics,
and explicitly selected GT estimator may consume it. It is not an implicit field
of policy observations.

`ObjectState` uses episode-local tracking IDs and a workcell `Pose` (reusing the
same type as EEF states, rather than separate center/orientation fields) plus
own-frame `size_xyz`, visibility, confidence, support surface, and optional
masks or visual features. `TaskContext` supplies instruction, target ID, and
optional previous-neighbor ID; `role_of(context, track_id)` derives TARGET,
PREVIOUS_NEIGHBOR, or OTHER on demand and is never stored on `ObjectState`
itself. Asset IDs and permanent semantic identities must not leak into
generalization experiments; RoboTwin's own `create_actor` gives every instance
of one asset the identical name, so track_id assignment is always the caller's
responsibility, never derived from a simulator-native identifier.

`ObjectStateEstimator.estimate(observation) -> ObjectScene` is the shared
output contract; `GroundTruthObjectStateEstimator` trusts an injected
`ObjectEvidenceSource` completely and only performs frame conversion, so it
has no RoboTwin dependency itself. `RoboTwinObjectEvidenceSource` is the one
piece that touches SAPIEN actors directly, reading measured poses (never a
segmentation ID or asset name) into that evidence contract. A future
`VisionObjectStateEstimator` produces the same `ObjectScene` from RGB instead.

## Shelf-restocking task specification

`tasks/shelf_restock/spec.py` defines the task purely: `ShelfRegion` (an
axis-aligned box in the workcell frame, one shelf level's deck), `ObjectVariation`
(uniform per-episode position/yaw/size sampling ranges), and `ShelfRestockSpec`
(both regions, the variation, a compaction-distance threshold, and the fixed
task instruction). None of this imports RoboTwin or SAPIEN; a scene builder
(not yet implemented) will instantiate a live scene from this spec. The default
geometry offsets the upper shelf in y from the lower shelf/source region
specifically because the old reference implementation's single shelf sat inside
the arm's straight-line base-to-source approach corridor and failed 0/545
collection attempts for exactly that reason; this offset is not yet confirmed
against a real planner and scene.

`tasks/shelf_restock/geometry.py` and `success.py` compute placement/compaction
success purely from an `ObjectScene` and the spec: horizontal (xy-only) distance
between target and previous neighbor, and shelf-region membership by position.
This checks geometry only; contacts, collisions, and grasp quality are
simulation-privileged diagnostics for the oracle/evaluator, not derivable from
the canonical object schema. `tasks/shelf_restock/manager.py`'s
`ShelfRestockManager` is the thin, stateful, deterministic loop from the
research plan: it selects the next lower-shelf target and reports the
previously-placed object as the neighbor, tracking placement order itself
since `ObjectScene` carries no per-object placement timestamp. It executes no
manipulation skill; the policy or oracle performs each restock and calls
`record_placement` back into it.

`tasks/shelf_restock/robotwin_env.py`'s `ShelfRestockTask` builds this scene
in RoboTwin: a static upper-shelf box plus randomly placed box objects on the
lower shelf/table, loaded as an external task entrypoint
(`module:ClassName`, resolved by `RoboTwinNativePort._load_task_class`
without copying anything into the RoboTwin checkout, reusing the mechanism
the old repo already proved out). `play_once` is intentionally
`NotImplementedError`; there is no oracle yet.

Live verification against real SAPIEN/cuRobo confirmed the object/GT-estimator
pipeline works against genuinely spawned objects, and surfaced one real bug
along the way: `ShelfRegion.contains()` originally checked whether a position
fell inside the deck's own thin physical slab, so no real object (whose
center sits above the deck by roughly half its height) ever registered as
"on" either shelf. Fixed by separating deck geometry from an "occupancy"
z-band above the deck surface.

It also surfaced an important scoping fact: cuRobo's collision world is built
once, in `CuroboPlanner.__init__` (called during `load_robot()`, before this
task's `load_actors()` ever runs), from a single hardcoded generic table
cuboid -- by default it has no knowledge of this shelf or any spawned object.

## Oracle: world model

`tasks/shelf_restock/oracle/world.py` closes that gap. `MotionGen.update_world`
can only ever replace obstacles up to the collision cache size fixed at
`MotionGenConfig` construction time -- a hard cuRobo limit under CUDA graphs,
which RoboTwin uses by default -- so that headroom must be reserved *before*
`CuroboPlanner.__init__` runs, before this task's objects even exist.
`install_world_patch` patches `MotionGenConfig.load_from_robot_config` in this
process only (nothing on disk, inside the RoboTwin checkout or otherwise, is
modified) to reserve capacity for the upper shelf plus every object this task
could spawn, and to inject the upper shelf's real geometry into the initial
world model. `ShelfRestockTask.setup_demo` installs this before calling
`_init_task_env_`. Once real objects exist, `register_objects` (called at the
end of `load_actors`) pushes their actual poses/sizes into both arms'
`MotionGen`/`MotionGen_batch` planners via `update_world` -- no placeholder or
parked dummy obstacles needed, unlike the old repo's abandoned v1 draft,
because `collision_cache={"obb": N}` reserves the headroom cleanly instead.

Live verification repeated the reachability probe above with the shelf now a
real obstacle. The lower-shelf-object reach still succeeded in the same 29
steps; the upper-shelf reach still failed cuRobo's global planner at the same
pose as before the patch. That the failure point didn't move is itself
informative: it means the earlier failure was never a shelf-collision issue
(cuRobo could not have avoided what it could not see, so a *different*
failure point after the patch would have been the sign of one) -- it is more
likely an IK/orientation difficulty, since the probe held a fixed orientation
from reset throughout, which a real oracle would not do. The pickup-side
success is now real evidence the shelf doesn't block that approach; grasp and
placement pose selection (candidate generation, not a fixed test orientation)
is the next oracle piece, not yet built.

## Action, frame, and timing contracts

The sole learned action is a 14-D Cartesian step:

```text
[left dx,dy,dz, drx,dry,drz, gripper,
 right dx,dy,dz, drx,dry,drz, gripper]
```

Translation is in meters and rotation vectors in radians, per policy step.
Gripper targets are absolute normalized commands: 0 closed, 1 open. Backend
calibration maps these to physical commands. Holding an arm means zero pose
increments and retaining its gripper target, not blindly zeroing all seven values.

Policy-facing EEF/object poses and deltas use a shelf-attached `WORKCELL_FRAME`.
Use right-handed coordinates, quaternions in `xyzw` order, and active rotations.
Let `R_t` map tool coordinates into workcell coordinates. Define:

```text
dp = p_next - p_t
dr = Log(R_next R_t^T)
p_next = p_t + dp
R_next = Exp(dr) R_t
```

The rotation increment is therefore expressed in the workcell frame and composed
on the left. `T_A_B` maps coordinates from frame B to frame A. Backend adapters
convert simulator world or robot base to/from workcell; the workcell origin and
axis alignment require explicit scene calibration. Commit 2 implements and tests
composition, quaternion sign equivalence, near-pi behavior, and transforms.

The 16 values needed to store two absolute EEF poses and grippers are an
observation representation, not a second learned action contract. Joint q/qdot
remain low-level execution or diagnostic data.

Target policy/data cadence is approximately 15 Hz. The backend owns faster inner
control, smooth execution, and measured feedback. Acquisition timestamps and
actual sampling intervals are mandatory; an FPS label cannot replace resampling.
The same action composition is used for data conversion and deployment.

## Processes and dependencies

Keep shared contracts compatible with Python 3.10 and 3.12. The simulator process
owns RoboTwin/SAPIEN/cuRobo; the policy process owns LeRobot/PyTorch/PEFT. Package
imports must not eagerly initialize either stack. Optional dependencies will be
introduced only when their integration is implemented and verified.

Keep RoboTwin external. A thin task lifecycle adapter can implement `load_actors`,
`play_once`, and `check_success`; scene geometry, success rules, and oracle logic
remain in small separate modules. Preserve required external modifications in
reproducibility records; a clean upstream revision is insufficient when local
changes affect behavior.

Prefer LeRobot/OpenPI-style action-chunk serving for policy inference. A separate
robot service should expose Reset, Observe, Step, Stop, Health, and an explicit
GetPrivilegedState operation. Avoid the old opaque `Call(pickle(...))` robot API.
Any unavoidable temporary compatibility transport must be documented and bounded.

External paths are configured locally. Precedence is CLI arguments over process
environment over an explicit env file over JSON configuration and derived defaults.
`doctor` checks dependency metadata, source resolution, directory presence, and GPU
driver visibility without installing anything. It does not verify runtime GPU use.
Existing environments are reused without modification; stale editable paths must
be handled explicitly at launch. Fresh-install documentation is separate from
normal execution.

## Expert generation and data

The oracle is a simulation-only state machine: select target, generate candidate
grasps, plan to pregrasp, approach locally, close, verify, attach, retreat,
transport, place, release, verify support, optionally compact, and retreat.
Use collision-aware global planning and short controlled Cartesian interaction
segments. Dense independent IK and manual axis corridors are not global planners.

Represent the held object during transport, along with neighboring objects and
the inactive arm. Preserve meaningful planner q/qdot. Freeze an
`ExpertEpisodePlan` with timestamps, joint positions/velocities, gripper commands,
phase labels, and scene/target metadata after generation and validation. Reset
and replay that exact plan for recording, without replanning. Generation success
and physical replay success are separate measurements; contact dynamics can
still cause replay divergence.

Store raw episodes separately from canonical episodes. Canonical data contains
timestamps, RGB, EEF/proprioception, object scene, task context, 14-D actions,
optional phases, and reproducibility/success metadata. Privileged debugging data
lives under `debug/privileged/*` and is excluded from training by default.

Validation checks monotonic timestamps and spacing, complete cameras, finite and
bounded actions, normalized quaternions, valid object IDs/masks and target IDs,
episode length, and success metadata. Export to LeRobot is an adapter over the
canonical schema. Old recorded command targets must not be silently relabeled
as subsequent measured poses when converting legacy data.

## Policy and scientific validation

Retain pretrained π0.5 and inject separately encoded object context through a
zero-initialized projection. Detector execution and task geometry stay outside
the neural policy. Verify LoRA parameter selection and checkpoint round trips.
Test RGB-only and conditioned forward paths, two-episode overfit, and preservation
of base behavior at initialization under identical inputs and inference noise.

Define object identity, appearance, geometry, source pose/orientation, neighbor
configuration, and shelf-location splits before collection. Explicitly evaluate
seen object/seen composition, seen object/unseen composition, unseen appearance/
seen geometry, unseen object/similar geometry, and unseen object/unseen composition.
Report appearance, geometry, and composition generalization separately.

## Incremental delivery and acceptance gates

1. Package skeleton and architecture.
2. Canonical geometry/actions with deterministic unit tests.
3. External path discovery, doctor, and fresh-install documentation.
4. RoboTwin backend with measured translation, rotation, gripper, and hold tests.
5. Object schema and GT estimator.
6. Atomic shelf environment, success rules, and split factors.
7. Grasp candidates, collision-aware global planning, and held-object handling.
8. Relational placement and compaction.
9. Immutable expert plan, exact replay, recording, and categorized diagnostics.
10. Timestamp resampling, validation, and LeRobot export.
11. π0.5 LoRA RGB baseline and two-episode overfit.
12. Object-conditioned π0.5 and LoRA support.
13. Shared runtime and evaluation metrics.
14. Reproducible 25-demo ablation configurations.

Verify basic robot actuation before implementing the shelf task. Test grasp,
transport, placement, and compaction independently, then full episodes. Before
training-data generation, target 100/100 reference-scene oracle successes or
document the precise shortfall; measure randomized-scene and replay success too.

Failures should identify seed, phase, and category, such as IK_NO_SOLUTION,
GLOBAL_PLAN_FAIL, PLANNED_COLLISION, EXECUTION_TRACKING_ERROR,
PHYSICAL_ROBOT_SHELF_CONTACT, GRASP_CONTACT_FAIL, GRASP_LIFT_FAIL,
HELD_OBJECT_COLLISION, PLACE_SUPPORT_FAIL, COMPACTION_FAIL, or REPLAY_DIVERGENCE.
Unit tests and videos alone are not evidence of physical task success.

Each step is reviewed before the next begins. No dependency installation,
dataset generation, training run, or physical validation is claimed by this
skeleton. Correct robot behavior and scientific validity take priority over
reuse of the old implementation.
