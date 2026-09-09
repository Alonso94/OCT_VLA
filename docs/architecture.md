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
and objects), top-down grasp-candidate generation, role-based arm assignment,
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
in RoboTwin: a static upper-shelf box plus randomly placed `113_coffee-box`
mesh objects on the lower shelf/table, loaded as an external task entrypoint
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
could spawn. `ShelfRestockTask.setup_demo` installs this before calling
`_init_task_env_`. `register_objects` then pushes the real geometry into both
arms' `MotionGen`/`MotionGen_batch` planners via `update_world` -- no
placeholder or parked dummy obstacles needed, unlike the old repo's abandoned
v1 draft, because `collision_cache={"obb": N}` reserves the headroom cleanly
instead.

Capacity is *all* the patch reserves. It deliberately injects no geometry,
for the reason in the next section: the hook has no access to the planner
whose frame the world is expressed in, so anything it added would land in the
wrong place.

### cuRobo's world is in the robot's base frame, not the world frame

This was wrong here for several commits, and it invalidated the collision
safety those commits claimed. Each RoboTwin planner expresses its obstacles
relative to *its own robot base*, not in world coordinates. Three independent
confirmations in RoboTwin's `envs/robot/planner.py`: its stock table cuboid is
posed at `0.74 - robot_origion_pose.p[2]`; `plan_path` runs every target
through `_trans_from_world_to_base` before planning; and that function is
exactly `wRb.T @ (p - base_p)`.

Registering world-frame poses therefore placed every obstacle -- the shelf and
every object -- somewhere the arm was never going to be. The planner was
avoiding phantom geometry while the real shelf and real objects stayed
invisible to it, which is why plans kept reporting `Success` and then driving
through the shelf or sweeping a placed object off it. `planner_cuboids`
reproduces RoboTwin's own conversion and is applied per planner, since the two
arms have different bases.

It does not apply the planner's `frame_bias`, which `plan_path` adds to
targets after the same conversion. That value is `[0, 0, 0]` for franka-panda,
and guessing how a nonzero bias ought to apply to obstacles would be inventing
a convention rather than matching one.

A second bug sat underneath: `update_world` *replaces* the world rather than
merging into it, so registering shelf and objects silently deleted RoboTwin's
own table obstacle. The table is now re-supplied on every call, with the
dimensions `Base_Task.create_table_and_wall` actually uses
(`create_table(length=1.2, width=0.7, thickness=0.05)`, whose tabletop slab is
centred half a thickness below the actor origin).

Objects are re-registered from live actor poses before *every* plan, in
`RoboTwinNativePort.plan` rather than at each call site. A stale world is the
failure mode hardest to notice, because the planner still reports `Success`
while routing through an object it believes has not moved.

### One object the arm may ignore

Correct obstacles create a problem correct-but-invisible ones did not: an arm
cannot plan against the object it is deliberately engaging with. A grasp pose
overlaps the target's own bounding box, and a carried object sits exactly
where the hand already is. Either way cuRobo starts in collision and returns a
bare `Fail` -- seen first when placing a held object, then again when
descending onto a target.

`move_to(..., ignore_object=track_id)` drops that one object from the
planning world for the duration of the motion, via `NativePort.ignored_object`.
It is deliberately one object and deliberately explicit: the caller states
which object it is engaging, rather than the world guessing from proximity or
contact.

This is the narrow fix, not the complete one. The planner also stops
accounting for the *volume* of a carried object, so a carried object can still
clip scene geometry. cuRobo's `attach_objects_to_robot` is the fuller answer
-- it would move the object's geometry with the arm instead of deleting it --
and is not done here.

Note the two identifiers in play: the planning world is keyed by `track_id`
(`obj_0`), while contacts report SAPIEN body names (`restock_object_0`). A
carrying motion passes both, for different checks.

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

## Oracle: placement and compaction

Placing is grasping read backwards: the same held-object relationship, with
the object's *intended* pose substituted for its current one. So
`grasps.holding_tcp_pose` and `backed_off` are shared by both, and
`oracle/placement.py` only decides *where* the object goes.

The first object goes to the middle of the upper deck. A later one goes
beside the previously placed neighbour, deliberately leaving `PLACEMENT_GAP`
(0.06m) -- wider than the success threshold -- so that the compaction step
has something real to do; compaction then closes that to `COMPACTED_GAP`
(0.005m). Which side of the neighbour is chosen is whichever has more shelf
room, so a row builds outward instead of running off the deck.

The success criterion had to be fixed to make this expressible at all. It
compared *centre-to-centre* distance against `compaction_distance`, which is
unsatisfiable for every object size the spec allows: two boxes that are
physically touching are already `(width_a + width_b) / 2` apart. It now
measures the gap between *surfaces*, treating each object as a circle of
`horizontal_radius` (half its larger horizontal dimension) -- size-aware,
yaw-independent, and conservative, since for a non-square footprint the true
gap along the line of centres is never smaller than this.

### Compaction is a push, not a re-grasp

The compacting arm never opens. It descends beside the placed object with a
closed gripper, shoves it sideways, and lifts away. A re-grasp was built
first and did work end to end (gap 0.0099m), but a push has no grasp that can
miss, and a closed gripper is simply a rigid tool.

The approach is **top-down**, not from the side, and that was not the first
choice. A horizontal approach was implemented and rejected by the planner: the
pose the arm is commanded to sits `GRASP_TCP_OFFSET` (0.127m) back from the
finger pads along the approach axis, so making that axis horizontal throws the
wrist 0.15-0.25m sideways from the object. Pushing toward -x put it at
`x = 0.254` against the right arm's own base at `x = 0.4`; pushing toward +x
puts it beyond `x = -0.16`, past where that arm had already failed to plan.
Both ends of a horizontal axis are unreachable for this arm, so the push
reuses the only wrist orientation that plans reliably here, with the fingers
separated perpendicular to travel so both pads meet the object's face.

`push_axis_half_extent` exists because `horizontal_radius` is the wrong
measure for this. That function returns a yaw-independent circumscribing
radius -- correct and deliberately conservative for a neighbour-gap test, but
it *understates* the extent along any one axis for a rotated box. Sizing the
descent with it put the gripper 6mm inside a box yawed 0.33rad (0.027 assumed
against 0.0332 actual), and the closed gripper's own ~12mm half-thickness was
not counted at all.

The failure that exposed this is worth recording because it did not look like
a geometry bug. Every step reported success, the planner returned Success, and
the object simply did not move -- 3mm against 58mm commanded. The measurement
that settled it was commanded-versus-achieved pose: the arm sat 62mm *high* at
both contact and push, having jammed its descent against the box and ridden
over the top, catching it with a fingertip. With the clearance corrected the
same motion tracks to 2.6mm and the descent reports no contacts at all.

Both transfers then succeed, with the object moving 0.062m and a final
neighbour gap of 0.033m against a 0.04m threshold.

### The push is planned as a constrained path

An ordinary plan curves through the contact: the blade sweeps an arc and
shoves the object off its intended line. cuRobo can constrain the path
instead, via RoboTwin's `plan_path(constraint_pose=...)`, which becomes a
`PoseCostMetric(hold_partial_pose=True, hold_vec_weight=...)`. That vector has
six entries, rotation 0-2 and translation 3-5, 1.0 holding a component fixed
along the path; it is read in the *goal* frame, since cuRobo's
`project_distance` defaults to true and nothing here overrides it.

The obvious choice -- hold five of six and free only the travel axis -- is
what this wanted and cuRobo reproducibly refused to plan it. The probable
reason, not isolated directly, is that the held components are pinned to the
goal pose's values for the whole path while the arm arrives at its contact
pose a few millimetres off what was commanded (measured: commanded
`(0.0529, -0.0584, 1.1008)`, achieved `(0.0509, -0.0592, 1.1045)`), so the
constraint cannot be met from the first waypoint.

What plans is holding all three rotations plus the end-effector's local +X,
which under the push orientation is the vertical. That targets the defect that
actually matters: vertical deviation is what let the blade ride up over the
object in the 62mm stall, while a few millimetres of lateral bow does not hurt
a push -- and the planner needs that horizontal freedom to stay feasible.
Measured, the blade then holds its height to 0.2mm across the whole push, and
the neighbour gap comes out at 0.0035m against a 0.04m threshold, the best of
any variant tried.

Contact-rich motions are also executed slower, by commanding each planned row
for several ticks with its velocity scaled to match. The trajectory is run
open-loop, so a fast blade bounces the object rather than sliding it. The
descent beside the object needs no such help and runs at full speed. Push
slowdowns of 3, 5, 6, 7 and 10 were each measured live: 3 through 7 are
indistinguishable (gap 0.004-0.006m against a 0.04m threshold) and only 10 is
clearly worse, so the value is not a tuned parameter and should not be treated
as one.

### A complete episode

All three objects restocked, two compactions, the lower shelf emptied, and
`check_success` returning True -- its first exercise. Placement error 0.005m;
compaction gaps 0.0047m and 0.0037m against the 0.04m threshold.

What this run tested that the two-transfer runs could not is the row *chain*:
placements at -0.114, -0.005 and 0.043, compacting back to -0.071 and -0.027.
The row built and stayed inside both arms' reach across two links, which is
what the "start one step from centre and grow toward the compacting arm" rule
exists for, and the third placement is precisely the case the old roomier-side
heuristic would have dropped onto the first object.

It remains one run at one seed. cuRobo's trajopt is stochastic -- the identical
script at the identical seed has failed at different steps on consecutive runs
earlier in this work -- so this is an existence proof that a full episode is
possible, not a success rate.

One limitation this leaves: a flat blade pushing a *yawed* box does not push
it along the blade's normal, so the object slides in y as well -- 0.008m in
the measured run, and the horizontal freedom the constraint allows is part of
why. The gap threshold absorbs it, but it will grow with the object's yaw.

### Parking the idle arm

The idle arm is not in the planner's world (see the open gaps above), so
nothing but sequencing keeps the arms apart. Compaction makes a collision
certain rather than likely: it sends the right arm to exactly where the left
arm has just placed, and the left arm's retreat leaves it directly above that
spot. Observed as `right/panda_link6 <-> left/panda_link6` at 0.178 impulse,
with the two wrists 0.088m apart.

The oracle therefore parks the placing arm at its own measured rest pose
before the compacting arm moves in, and parks the compacting arm again
afterwards. The rest pose is read from the arm at episode start rather than
written down as a constant: it is known-reachable by construction, which a
hand-chosen park pose would not be.

### A live two-transfer run, and what it exposed

Transfer 0 succeeded end to end -- grasp, transport, place, release, retreat
-- with zero robot contacts remaining and the object on the upper shelf.
Transfer 1 failed at `preplace`, and the failure was reported as
`panda_rightfinger <-> panda_hand`.

That reading was wrong, and so was my first explanation of it (a missed
right-arm grasp -- disproved directly: an isolated right-arm grasp lifts the
object 0.104m with both fingers in contact and the gripper holding at 0.571,
nowhere near bottomed out). The truth is that **both arms load the same
Panda URDF**, so all fourteen link names are shared between them, and
`contacts()` was pooling both arms' link *names* into one set. It therefore
could not say which arm a link belonged to, and an arm-vs-arm collision was
indistinguishable from a gripper closing on itself.

Contacts are now keyed by each link entity's `per_scene_id` and carry the
owning `side`, with the counterpart qualified (`left/panda_hand`) whenever it
is also a robot link. That the original event was arm-vs-arm is an inference
from the shared naming, not a re-observation -- that exact run was not
repeated. The mechanism itself is confirmed directly, though: a later run
reported `right/panda_link6 <-> left/panda_link7` with both wrists 0.13m
apart, which the old code could not have expressed at all.

### Fixed arm roles

Arms are now assigned by **role**, not by which side the target is on: the
**left** arm grasps and places, the **right** arm only compacts. A fixed
split keeps each arm's job identical across episodes, which is what makes
demonstrations learnable. It does not by itself resolve the cross-arm
collision below: compaction sends the right arm to the very spot the left
arm has just placed into, and the left arm's retreat leaves it directly
above that spot. `select_arm` (same-side selection) is
gone, replaced by `arm_for(role)`; the measured cross-body data it was built
on is retained in `is_cross_body_limited`, which now guards real cases
rather than impossible ones.

The consequence to watch: the left arm must place across the whole deck, and
the upper shelf sits at `y = -0.02`, inside the cross-body band, so
placements at `x >= CROSS_BODY_X` (0.15) are out of its reach. `move_to`
raises with a kinematic explanation rather than letting it surface as an
opaque planner `Fail`. Grasping is unaffected -- lower-shelf objects sit
around `y = -0.25`, well below `CROSS_BODY_Y` -- so the left arm covers the
full spawn width there.

This bounds how many objects a row can hold before the placing arm runs out
of reach, which is a real limit on episode length, not a bug.

### The restocked object is a real asset

The objects are RoboTwin's `113_coffee-box` scanned mesh, not procedural
boxes. A box's uniform faces and exact symmetry would flatter the
object-centric conditioning this research is meant to evaluate, and the
policy should see the kind of object it would meet at deployment.

Three consequences follow, none of them cosmetic.

**Size stops being a task parameter.** `create_actor` overwrites its own
`scale` argument with the value inside `model_data<N>.json`, so an asset's
dimensions belong to the asset. `ObjectVariation`'s size sampling no longer
drives spawning; per-episode size variation is now which of the seven
variants is drawn. `assets.upright_size` reads `extents * scale` and rotates
it into the upright frame. Measured, the variants run 0.048-0.078m across
their narrower horizontal axis and 0.057-0.097m tall.

**The mesh frame is not the object frame.** These assets are authored y-up
and spawned upright by a fixed base quaternion (RoboTwin's own
`qpos=[0.5, 0.5, 0.5, 0.5]`, which maps model +y to world +z). That rotation
describes the mesh, not the object, so `TrackedActor.upright_rotation` is
divided back out before an `ObjectState` is reported. Leaving it in would
make the canonical pose asset-specific, and `grasps._object_yaw` -- which
reads the z component of the orientation's log, valid only for an upright,
yaw-only object -- would have extracted a meaningless number from a
120-degree tilt.

**Instance names must be assigned.** `create_actor` calls
`mesh.set_name(modelname)`, so all three objects would be `113_coffee-box`.
Contact reports and the oracle's `allow_contact_with` both key on that name,
so identical names would let a collision with *any* object be excused as
contact with the held one. The task renames each instance after building it.

Grasp candidates are now returned narrowest-closing-width first. The gripper
opens to 0.08m and a variant can be 0.078m across one axis and 0.022m across
the other; both "fit", so an unordered list let a caller take a grasp with
2mm of total clearance when a comfortable one existed.

### Head camera

The stock embodiment head-camera pose frames a tabletop and does not see the
upper shelf at all. `robotwin_env.HEAD_CAMERA_OVERRIDE` pulls it back and up
so both shelf levels are in frame -- a precondition for the task being
learnable from RGB, not a cosmetic choice. RoboTwin reads the static camera
list from `left_embodiment_config` only (`envs/camera/camera.py`), so only
that copy is overridden; the override raises if no `head_camera` entry is
found, since the alternative is recording an entire dataset that cannot see
the target shelf.

### Two open gaps in what the planner can see

Both were exposed by the same live two-transfer run, and both are the same
shape: `move_to` calls `port.plan(side, ...)`, a **single-arm** plan against
a world containing only the static geometry. Two things that can be hit are
absent from it.

**The other arm.** The idle arm is *held* at its joint positions but is not
in the planning world, so nothing stops one arm being routed through the
other. Compaction makes this unavoidable rather than incidental: it sends
the right arm to exactly the spot the left arm just placed into, while the
left arm's retreat leaves it directly above that spot. Observed as
`right/panda_link6 <-> left/panda_link7` (impulse 0.255) together with
`right/panda_hand <-> left/panda_leftfinger`, with both wrists at z ~1.2 and
only 0.13m apart in x. It needs either a park pose that clears the shared
workspace before the other arm works, or the idle arm's geometry added to
the planning world -- an open decision, not yet made.

**Where the objects are now.** `register_objects` does add every spawned
object to cuRobo's world -- but only once, at `load_actors`, from the poses
they were spawned at. Nothing re-registers them, so the moment the first
object is moved to the upper shelf the planner's model of the scene is
wrong: it still believes that object is sitting on the lower shelf where it
started, and believes the space it now occupies is empty.

The consequence is reproducible. In two separate runs the left arm's
cross-body reach for the second object swept the already-placed first object
clean off the upper shelf -- placed at `(0.012, -0.050, 0.977)` and found
afterwards at `(0.188, -0.395, 0.756)`, a shelf level down and most of the
way across the table. The second run shows the full cascade: the displaced
object then knocked `obj_1` about 16mm out of position, which made the grasp
pose (computed from the pre-move observation) stale, and the descent ended
with `left/panda_rightfinger <-> restock_object_1`.

Two things make this worse than a single failed transfer. The task had
already been scored a success for the object that was later swept away --
success is per transfer, so a later transfer silently undoing an earlier one
counts as two successes. And a stale world is *invisible*: the planner
reports `Success` for a trajectory through a space it wrongly believes to be
empty, which is the same soft-cost blind spot as before, only now fed with
stale data as well.

**Both of these are now fixed** -- and the deeper cause was worse than
staleness alone: the obstacles were also in the wrong *frame* (see "cuRobo's
world is in the robot's base frame" above), so the planner had never really
seen the shelf or the objects at all. With poses in the right frame and
refreshed before every plan, the same seed no longer sweeps the placed object
off the shelf: it stays at `(0.014, -0.044, 0.974)` through the following
transfer, where it previously ended at `(0.188, -0.395, 0.756)`. Placement
error also fell from ~0.036m to ~0.023m, and transfer 1 now carries its object
all the way to the shelf instead of failing at the grasp.

What still fails is transfer 1's `place`, as a planner `Fail` rather than a
collision. It is marginal, not systematic: transfer 0 places successfully at
`z=0.974` while transfer 1's object is 10mm shorter and places at `z=0.968`,
bringing the fingers nearer the deck that is only now a real obstacle. The
cause is not yet isolated, and guessing at it is what this document has
repeatedly had to retract.

These are not exotic. Both arms and every object converge on one narrow
shelf, so both gaps are reachable in ordinary operation.

### Run-to-run nondeterminism at a fixed seed

Two runs of the identical script at `seed=0` failed at different steps: one
at transfer 1's `descend` (`left/panda_rightfinger <-> restock_object_0`),
the other only later at `c:pregrasp` with the arm-vs-arm collision, having
completed the whole left-arm sequence. The scene is identical; cuRobo's
trajopt is seeded stochastically, so a fixed scene seed does not fix the
trajectory. Any reliability number therefore has to come from repeated runs
per seed, not one run per seed -- and a single passing run is not evidence
that a step is fixed.

### The placement error was a reach problem, now resolved

Earlier notes here speculated that objects "settle toward `y ~ -0.056`
regardless of where they were aimed", pointing at the deck's modelled centre
or a release-time nudge. That was wrong. Moving the upper deck forward to
`y = -0.06` -- done for reachability, not accuracy -- dropped placement error
from ~0.023-0.036m to **0.0004m**. The error was never settling or tracking:
it was the arm contorting to place inside its own cross-body-limited region,
and it disappears once the target is outside that region.

Two things follow. Aiming accuracy is not a limitation of this setup, so the
compaction step does not need to exist in order to correct placement (it
still does correct it, by re-observing before it re-grasps). And a
"discrepancy" that vanishes when an unrelated geometry constant changes was
never a discrepancy worth modelling -- it was a symptom of a reach limit.

Objects still come to rest ~3mm above `resting_z`, suggesting
`upper_shelf.top_z` is slightly under-stated. Unresolved, and now the largest
remaining placement discrepancy by an order of magnitude.

### A reach limit `is_cross_body_limited` does not cover

The right arm failed to plan to `x = -0.124` on the upper deck at
`y = -0.06`. Nothing flagged this in advance: `contains()` accepts the
position (the deck is 0.60m wide), and `is_cross_body_limited` no longer
fires anywhere on the deck at all, because moving it to `y = -0.06` put the
whole deck below `CROSS_BODY_Y`. The measured grid that produced those
constants only covered `x` in -0.15..0.15 at `y >= -0.05`, so it says nothing
about this pose.

So the deck's outer thirds are usable by the arm on their own side and by
nothing else. That is what makes *where a row starts* a reachability
decision rather than an aesthetic one: compaction pulls each new object back
toward its neighbour, so a row does not march across the deck -- it stays
clustered near its first object. Starting at the far edge parks the entire
row where the compacting arm cannot reach it. Rows now start one placement
step from the deck centre, away from the compacting arm, which keeps a short
row inside the overlap of both arms' reach.

The boundary of this limit is not measured. Treat 0.16m from the deck centre
as the tested-good extent, not as a known edge.

### Scenario geometry

The scene was retuned toward the reference scenario: objects spawn only on
the left half (`x` in -0.34..0.0), since the left arm both grasps and places
and its base is at `x = -0.4`; the deck is wider (0.40 -> 0.60m) to hold a
row; and every object in an episode is the same asset variant, so a row's
geometry does not change size mid-sequence for reasons the policy cannot see.

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
