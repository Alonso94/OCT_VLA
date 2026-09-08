# Architecture and contracts

This document specifies the intended architecture. Commits 1–6 implement the
package boundary, geometry, frame-labelled poses/transforms, canonical EEF state,
14-D actions, external path configuration, `doctor` discovery, the canonical
`RobotObservation`/`RobotBackend` contract, a RoboTwin backend that converts
world-frame measurements to the workcell frame with checked dual-arm planning,
the canonical object-scene schema with a RoboTwin ground-truth estimator, and
the shelf-restocking task's pure specification, geometry, success check, and
repeat-until-empty manager, a RoboTwin scene that builds this task live
(spawned objects, no oracle), and the first four oracle pieces: world-model
registration (cuRobo's planner is now genuinely aware of this task's shelf
and objects), top-down grasp-candidate generation, side-aware arm selection,
and plan-once motion primitives. Other interfaces below remain design
contracts.

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
task instruction). None of this imports RoboTwin or SAPIEN; `robotwin_env.py`
instantiates a live scene from this spec. The geometry has since been
corrected twice against live measurement -- see the reachability findings and
the overhang problem below -- so treat the values in `spec.py` as measured
settings with a recorded rationale, not as free parameters.

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
failure point after the patch would have been the sign of one). The
pickup-side success is now real evidence the shelf doesn't block that
approach.

## Oracle: grasp candidates

`tasks/shelf_restock/oracle/grasps.py`'s `generate_top_down_grasps` is the
first candidate-generation piece (section 21 of the research plan: never
hard-code one grasp pose). It derives top-down grasp poses directly rather
than reading RoboTwin's own box "contact points": for a procedural
`create_box` actor those carry only orientation (translation is always
`(0, 0, 0)` regardless of object size, unlike a real scanned asset), so
reusing them buys nothing a direct derivation doesn't already give. Two
90-degree-apart candidates are generated per object (matching that RoboTwin's
own 4 box contact points reduce to the same 2 unique top-down orientations
plus 180-degree duplicates), filtered by whether the object's corresponding
horizontal dimension fits the gripper's opening. The canonical "pointing
straight down" quaternion is verified equal, up to the usual double-cover
sign, to RoboTwin's own `GRASP_DIRECTION_DIC["top_down"]` constant; composing
a yaw rotation about the workcell z-axis on its *left* preserves "pointing
down" while spinning the wrist, which is how per-object and per-candidate
yaw are both applied. This is pure and unit-tested; it has no SAPIEN
dependency, matching the generic/pure-first pattern used throughout (backend,
ground-truth estimator, world-model registration).

Live verification against a real spawned lower-shelf object: both candidates'
`closing_width` matched the object's real size along the expected axis, and
walking the arm to the pregrasp pose completed 40 incremental steps with zero
planning failures -- meaningfully stronger evidence than the earlier
fixed-orientation probe, since this used the actual candidate orientation
this task would use.

This also refined, rather than confirmed, the earlier IK/orientation
hypothesis for the upper-shelf failure: reaching the *same* upper-shelf point
with a proper top-down orientation (not the earlier arbitrary fixed one)
still failed with the identical `'Fail'` status. A correct orientation not
fixing it meant the earlier hypothesis was wrong. A follow-up position/arm
sweep tested both arms against a grid of positions and found `y=0.10` at any
height from 1.065-1.185 consistently failed or landed 0.12-0.27m from the
requested pose for *both* arms, while targets around `z=0.90-1.00, y=-0.05
to 0.00` -- including `x=0.0`, dead center between the two arm bases --
landed within 5-10mm in several individual trials. `DEFAULT_SPEC.upper_shelf`
now reflects that evidence: center `(0.0, -0.05, 0.92)`, `top_z=0.935`
(previously `(0.0, 0.10, 1.05)`, `top_z=1.065`), still comfortably clear of
`lower_shelf`'s `top_z=0.75`.

Those first sweeps measured *execution* (plan, then run the trajectory, then
compare the achieved pose), which conflates three things: whether a pose is
plannable, whether the plan is followed accurately, and whether the stepping
strategy leaves the arm in a good configuration for the next request. Asking
the planner directly instead -- `robot.left_plan_path(...)` with nothing
executed, so every query starts from the same joint configuration -- is a far
cleaner instrument, and it corrected several conclusions reached from the
execution sweeps:

- The corrected `upper_shelf` position is genuinely plannable: its deck top
  and a preplace pose above it both return `Success`, from the home
  configuration *and* from the configuration reached after 40 greedy
  incremental steps. The old position (`y=0.10`, `top_z=1.065`) returns
  `Fail` from both. So the `spec.py` geometry change is confirmed, not just
  "an improvement" -- the old geometry was genuinely unplannable and the new
  one is not.
- A 6x5 grid over `z=0.85-1.10` and `y=-0.30-0.00` at the shelf's x returned
  `Success` for all 40 queries, as did the whole lower-shelf grasp column.
  There is no reach *ceiling* in the region this task uses, and no collision
  with the registered shelf along the straight line between pregrasp and
  preplace poses -- both hypotheses raised from the execution sweeps were
  wrong.

`num_trajopt_seeds` was A/B tested on an identical 90-query grid (both arms,
three x, five y, three z), pre-installing the world patch with the seed count
under test so nothing else differed. **`seeds=1` and `seeds=4` produced
identical results: 78 Success / 12 Fail, the same 12 failures.** RoboTwin's
`num_trajopt_seeds=1` is therefore not a reliability problem for this task's
workspace, and the earlier speculation that it explained intermittent
failures is withdrawn; that change was implemented, measured, found to do
nothing, and reverted rather than kept on plausibility.

The 12 failures are not random -- they are a deterministic, symmetric
**cross-body reach limit**: the left arm cannot plan to `x=+0.15` at
`y >= -0.05` (any height tried), and the right arm cannot plan to `x=-0.15`
at the same `y`, while every same-side and centreline pose succeeds. This is
real and actionable: **arm selection must be side-aware**. The oracle cannot
assign an arbitrary arm to an arbitrary object or placement slot; each arm
must handle its own side of the workcell, and the shelf's usable x-span per
arm is bounded by this limit. Every diagnostic in this session that used the
left arm for a positive-x target was fighting this constraint.

## Oracle: arm selection

`tasks/shelf_restock/oracle/arms.py` encodes that finding. `select_arm`
applies a same-side rule (ties on the `x=0` centreline go left,
deterministically), which by construction never produces a pairing inside the
measured dead zone. `is_cross_body_limited` reports whether a given
arm/target pairing falls in it, so code that picks an arm for other reasons
fails with a kinematic explanation rather than an opaque planner `Fail`.

The constants record where failure was *observed*, not an interpolated
boundary: the true edge lies somewhere between the sampled points (in `y`
between -0.10, which planned, and -0.05, which did not; in `x` between 0.0
and 0.15). A unit test replays the full 90-pose grid and asserts the module
agrees with every recorded planner result, so this stays pinned to
measurement rather than to reasoning about it.

Arm choice deliberately does not appear in `TaskContext`: the canonical
action carries both arms every step and the policy learns which to move
(the stationary arm holds), so which arm the *expert* used is an oracle
implementation detail, not policy-facing state.

## Oracle: motion primitives

`tasks/shelf_restock/oracle/motion.py` provides `move_to` (plan once with
cuRobo, execute that trajectory's rows, then settle on the final row) and
`set_gripper` (ramp one gripper while both arms hold). The other arm holds at
its pre-motion joint positions with zero velocity for every tick, matching
the canonical backend's hold semantics. Both are written against the
`NativePort` protocol rather than SAPIEN, so they are unit-tested with a fake
port -- including the property that matters most, that planning happens
exactly once for a whole motion.

They deliberately do not route through `RobotBackend.step()`. That interface
takes bounded per-step Cartesian deltas because a *policy* emits them, and
re-plans every step; driving a long motion through it is the "dense waypoint
IK used as trajectory planning" anti-pattern, and it demonstrably drove the
arm into configurations whose next waypoint could not be planned.

### Four findings from live verification, all now fixed

Executing these primitives against the real scene surfaced problems that the
planning-only queries above could not have caught, because they only checked
plan *status*. Each is now fixed and the fix is validated by an end-to-end
grasp-and-lift (below).

1. **A cuRobo `Success` does not mean the trajectory is collision-free.**
   Executing a `Success` plan left the arm in sustained contact:
   `panda_link6 <-> upper_shelf` (impulse 0.52), `panda_link7 <-> upper_shelf`
   (0.26), `panda_rightfinger <-> table` (0.40), holding a ~0.147 rad
   steady-state joint error that did not decay over 2000 settle ticks --
   the arm was jammed, not converging. Collision avoidance in trajopt is a
   soft cost, exactly as the old repo's investigation documented. **Every
   reachability conclusion in this document that rests on plan status alone
   is therefore weaker than it reads.** Fixed by adding `Contact` and
   `NativePort.contacts()` (robot links identified from the articulations
   themselves, not by name matching) and a `check_contacts` gate that
   `move_to` applies after every motion; `allow_contact_with` carries the
   held object. `set_gripper` deliberately does not gate, since closing on
   an object *is* contact.
2. **The upper shelf overhung the object spawn zone.** The deck spanned
   `y in [-0.13, 0.03]` while objects spawned at `y in [-0.20, -0.10]`, so an
   object at `y=-0.124` sat beneath it and could not be grasped top-down.
   This hypothesis was raised earlier and wrongly dismissed when
   planning-only queries returned `Success` -- they did so precisely because
   collision is a soft cost. Fixed by separating them: deck
   `y in [-0.08, 0.04]`, spawn `y in [-0.30, -0.20]`, 0.12m clear, with a
   unit test asserting the invariant so it cannot silently regress.
3. **`grasps.py` ignored the TCP-to-grasp-point offset.** The pose the robot
   reports and accepts is 0.127m above the point between the finger pads, so
   commanding it at an object's centre drove the fingers ~0.13m below the
   object, into the table. `GRASP_TCP_OFFSET` now offsets every candidate
   back along its own approach axis. The constant is composed from three
   independently checkable sources (measured hand offset, `panda.urdf` joint
   origin, cuRobo's own collision spheres) and cross-checks against the
   measured finger-origin position to 0.5mm.
4. **Grasping at an object's centre buries the hand in tall objects.**
   `panda_hand`'s lowest collision sphere sits 0.1067m below the commanded
   pose, i.e. only 0.0203m above the pads, so any object taller than ~4cm is
   penetrated by the hand -- observed as `panda_hand <-> restock_object_0`
   on a 7.5cm object, while the spec allows 3-8cm. Fixed by grasping
   `grasp_depth` below the object's *top face* rather than at its centre,
   with `MAX_GRASP_DEPTH` derived from that geometry and enforced.

A fifth problem appeared once the contact gate was live: the hand is ~0.10m
from grasp axis to outer edge, but objects spawned as little as 0.08m apart,
so descending onto one shoved its neighbour (`panda_hand <->
restock_object_1`). `MIN_OBJECT_SEPARATION` is now 0.15m, and because
rejection sampling could not reliably fit three objects into the spawn strip
at that separation (it failed outright at seed 0), the x positions are now
*constructed* with guaranteed gaps rather than retried.

### End-to-end validation

With all of the above in place, a full open / pregrasp / descend / close /
lift sequence against the live scene: object lifted +0.104m, both fingers in
contact with it and nothing else, and **every free-space motion passed the
contact gate**. Arm selection routed the target correctly by side. A demo
video is written to `outputs/` (gitignored).

This is the first end-to-end physical manipulation this rebuild has
achieved. It is one seed and one object, not a success rate; the oracle
still needs place, compaction, retreat, and repetition before any success
statistics mean anything.

What remains genuinely unexplained is narrower than previously claimed: a
greedy stepping loop (recompute the full remaining delta, clamp it, re-plan
from scratch, execute, repeat) sometimes reaches a configuration from which
its next intermediate waypoint fails, even though both endpoints and every
straight-line waypoint plan fine from a fixed configuration. That is a
property of the stepping strategy, not of the geometry or the planner
config, and it is precisely the "dense waypoint IK used as trajectory
planning" anti-pattern the research plan warns against. A real oracle plans
once to a target and executes cuRobo's own interpolated trajectory; it
should not inherit this behaviour, and no fix belongs in the task spec or
planner configuration for it.

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
