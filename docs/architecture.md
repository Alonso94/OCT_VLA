# Architecture and contracts

This document specifies the intended architecture. Commit 1 implements only the
package boundary and its import test; the types and operations below are design
contracts, not available APIs.

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

`ObjectScene` uses episode-local tracking IDs, metric center/orientation/size,
visibility, confidence, support surface, and optional masks or visual features.
`TaskContext` supplies instruction, target ID, and optional previous-neighbor ID.
TARGET, PREVIOUS_NEIGHBOR, and OTHER describe roles independently of identity.
Asset IDs and permanent semantic identities must not leak into generalization
experiments. GT and future vision estimators produce the same object schema.

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
axis alignment require explicit scene calibration. Commit 2 will implement and
test composition, quaternion sign equivalence, near-pi behavior, and transforms.

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

External paths are configured locally. Planned precedence is CLI arguments over
environment variables over configuration defaults. `doctor` checks dependencies,
source resolution, assets, and GPU availability without installing anything.
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

1. Package skeleton and architecture (this step).
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
