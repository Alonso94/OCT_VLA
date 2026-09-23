"""Simulator-side server: runs shelf-restock episodes for a remote policy.

Lives in RoboTwin's Python 3.10 environment and is the only half of the bridge
that imports it. One client at a time, one episode at a time: SAPIEN holds GPU
render contexts that must not be shared across connections, and evaluation is
sequential anyway.

Scene construction, object-state estimation and success checking are taken from
`tasks.shelf_restock.collect`, not reimplemented. A policy has to be scored
against the same notion of "restocked" that produced its training labels; a
second implementation here could drift from the oracle's and silently move the
measured success rate.
"""

from __future__ import annotations

import math
import os
import socket
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field

from oct_vla.core.action import Action, action_between, apply_action
from oct_vla.core.frames import Pose
from oct_vla.core.gripper import GRIPPER_ENCODINGS, decode_gripper_command
from oct_vla.core.objects import ObjectScene, TaskContext
from oct_vla.core.observation import RobotObservation
from oct_vla.core.state import ArmJoints, ArmState, EEFState, JointState
from oct_vla.perception.ground_truth import GroundTruthObjectStateEstimator
from oct_vla.perception.robotwin.evidence import RoboTwinObjectEvidenceSource, TrackedActor
from oct_vla.robots.robotwin.backend import decode_pose, encode_pose
from oct_vla.robots.robotwin.native import UnreachablePose
from oct_vla.serve import protocol
from oct_vla.serve.codec import context_to_json, eef_to_json, joints_to_json, scene_to_json
from oct_vla.tasks.shelf_restock.collect import DEFAULT_HZ, WORLD_TO_WORKCELL
from oct_vla.tasks.shelf_restock.manager import (
    ShelfRestockManager,
    objects_lifted,
    objects_on_upper_shelf,
)
from oct_vla.tasks.shelf_restock.events import TransferEvents
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC, ShelfRestockSpec

# Measured stationary EEF poses in the demonstrations move by a few tenths of
# a micrometre due to physics. Sending those numerical remnants through IK is
# both needless and harmful: a stationary arm should hold its measured joints,
# not search for another configuration approximating the same Cartesian pose.
IK_TRANSLATION_DEADBAND = 1e-5  # metres
IK_ROTATION_DEADBAND = 1e-5  # radians

# Anti-windup bound on the integrated command reference. Retaining the intended
# endpoint stops actuator residual being discarded at every step, but an
# unbounded integrator also accumulates whatever the policy emits for an arm it
# is not really driving. That is not hypothetical: the oracle's recorded action
# for an idle arm is exactly zero, a learned policy can only approximate it, and
# ~1 mm/step of upward bias integrated over a 600-step episode floated the right
# arm from z~0.9 to z~1.6 -- far above the shelves -- which is how 8 of 9 probe
# episodes ended. Past this distance the arm is not tracking the reference at
# all, so continuing to integrate describes a motion nothing is performing.
COMMAND_REFERENCE_LEASH = 0.05  # metres

# Actions carry the *next measured* finger aperture, because they are formed
# between two observations. That is not the actuator command that produced the
# demonstration: while grasping, the measured aperture stalls around 0.6--0.8
# while the oracle keeps commanding fully closed. Sending the measured value
# back as a drive target releases the object -- which is why the first sweep
# recorded a mean of exactly 0.00 transfers across all 1080 episodes.
#
# So the aperture is decoded back to the binary actuator state with a steeply
# scaled sigmoid about the decision point. Two properties matter:
#
#   Saturated. An intermediate command is a weak grip, i.e. the same bug. At
#   this sharpness the output is within 3e-7 of 0 or 1 across the entire
#   observed range, so nothing in between is ever commanded.
#
#   Memoryless. An earlier hysteresis version returned the *previous* command
#   for apertures inside a dead band, which made the command a function of
#   history rather than of the observation -- two identical observations could
#   be driven differently, and the dead band was not empty: 0.34% of the
#   corpus (38 of 11064 samples) falls inside it.
#
# Both constants are measured, not chosen. The recorded apertures are dense
# through this region -- the widest empty interval anywhere in [0.78, 0.86] is
# 0.00107 wide -- so a centre picked by eye lands on real data and maps it to a
# half-open command. The centre is that interval's midpoint, and the sharpness
# is set so the nearest observed sample on either side still decodes to within
# 1e-8 of a binary command. Re-derive both if the embodiment or its gripper
# calibration changes; `tests/serve` asserts the saturation property directly.
GRIPPER_DECISION_CENTRE = 0.822522
GRIPPER_DECISION_SHARPNESS = 37528.0

#: Profile -> RoboTwin task class, identical to scripts/collect_shelf_restock.py.
#: Evaluating count shift means running the same policy against these three.
PROFILE_TASKS = {
    "two_object": "ShelfRestockTwoObjectTask",
    "three_object": "ShelfRestockTask",
    "four_object": "ShelfRestockFourObjectTask",
}

class EvalServerError(RuntimeError):
    """The episode cannot continue; the client is told and the socket stays up."""


#: Read by `ShelfRestockTask._load_objects`. Set here, per reset, rather than by
#: the launching job: the job used to export it *after* forking this server, so
#: the pin never arrived and every identity tier scored the same mixed scenes.
MODEL_IDS_ENV = "OCTVLA_MODEL_IDS"


@contextmanager
def _pinned_model_ids(model_ids: tuple[int, ...] | None):
    """Pin the mesh variants for one scene build, then restore the environment."""
    previous = os.environ.get(MODEL_IDS_ENV)
    if model_ids is None:
        os.environ.pop(MODEL_IDS_ENV, None)
    else:
        os.environ[MODEL_IDS_ENV] = ",".join(str(v) for v in model_ids)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(MODEL_IDS_ENV, None)
        else:
            os.environ[MODEL_IDS_ENV] = previous


def _spawned_model_ids(task, requested: tuple[int, ...] | None) -> dict[str, int]:
    """What the scene actually holds, checked against what was asked for."""
    spawned = dict(getattr(task, "spawned_model_ids", None) or {})
    if requested is None:
        return spawned
    if not spawned:
        raise EvalServerError(
            f"model_ids {list(requested)} were requested but the task reports no "
            "spawned variants, so the pin cannot be verified"
        )
    stray = {track: v for track, v in spawned.items() if v not in requested}
    if stray:
        raise EvalServerError(
            f"model_ids {list(requested)} were requested but the scene spawned {stray}"
        )
    return spawned


@dataclass
class _Episode:
    """Mutable per-episode state, replaced wholesale on every reset."""

    spec: ShelfRestockSpec
    manager: ShelfRestockManager
    estimator: GroundTruthObjectStateEstimator
    max_steps: int
    steps: int = 0
    #: Fractional simulation ticks carried between control steps. The sim runs
    #: at 250 Hz and control at 15, which is 16.67 ticks per step: rounding
    #: every step would accumulate a 2% timing error against the cadence the
    #: demonstrations were recorded at.
    tick_debt: float = 0.0
    #: Seconds of simulation since reset. Tracked here because the port exposes
    #: no clock -- during collection that job belongs to EpisodeRecorder, which
    #: closed-loop rollout does not use.
    sim_time: float = 0.0
    #: Most recent real context. Once the shelf is clear the manager has no next
    #: target, but the client still needs a valid TaskContext for the final
    #: observation, and TaskContext forbids an empty target_track_id -- so the
    #: last one is reported with phase "done" rather than inventing a blank.
    last_context: TaskContext | None = None
    #: Integrated Cartesian command reference. Policy actions are increments;
    #: retaining their intended endpoint prevents actuator residual from being
    #: discarded and compounded at every 15 Hz step.
    commanded_eef: EEFState | None = None
    #: Steps in which at least one arm's command was kinematically infeasible
    #: and was held instead. A diagnostic, not a termination condition.
    infeasible_steps: int = 0
    last_infeasible: str = ""
    #: Objects seen on the upper shelf, in the order they arrived. Evaluation
    #: has to maintain this itself: collection calls
    #: ShelfRestockManager.record_placement as the oracle finishes each
    #: transfer, and without the equivalent here `_placed_order` stays empty and
    #: `_previous_neighbor` silently falls through to the lexicographically
    #: smallest neighbour instead of the most recently placed one. That changes
    #: which object carries the "previous neighbour" role in the object tokens
    #: from the third transfer onward -- so the object-conditioned arms would be
    #: scored on a token layout they were never trained on.
    #: How to read the policy's gripper output. Defaults to the raw measurement
    #: so an older client, or a checkpoint trained on an older dataset, executes
    #: exactly as it did before.
    gripper_encoding: str = "measured_aperture"
    placed_seen: set[str] = field(default_factory=set)
    #: Every object raised clear of the lower shelf at any point this episode.
    #: Accumulated rather than sampled, so an object that is lifted and dropped
    #: still counts -- that is the distinction the measure exists to make.
    lifted_ever: set[str] = field(default_factory=set)
    #: "cartesian" (14-d canonical increments, executed through IK) or "joint"
    #: (absolute joint targets, commanded directly). Declared at reset rather
    #: than inferred, so a mismatch is an error instead of a misreading.
    control_space: str = "cartesian"
    #: Ground-truth transfer stages per object, for diagnosing where a transfer
    #: fails (tasks/shelf_restock/events.py). Reported on the final step only.
    events: TransferEvents | None = None
    #: The end-effector state of the latest snapshot, which `events` needs.
    last_eef: EEFState | None = None


def _leashed_reference(commanded: EEFState | None, measured: EEFState) -> EEFState:
    """The integrated reference, re-anchored per arm once it runs away.

    Anti-windup, applied independently to each arm: the arm the policy is
    driving keeps its residual correction, while an arm whose reference has
    drifted beyond what it is actually tracking is snapped back to where it
    really is. Without this the two cannot be told apart.
    """
    if commanded is None:
        return measured
    arms = []
    for reference, actual in (
        (commanded.left, measured.left),
        (commanded.right, measured.right),
    ):
        gap = sum(
            (a - b) ** 2
            for a, b in zip(reference.pose.position, actual.pose.position, strict=True)
        ) ** 0.5
        arms.append(actual if gap > COMMAND_REFERENCE_LEASH else reference)
    return EEFState(arms[0], arms[1])


CONTROL_SPACES = ("cartesian", "cartesian_absolute", "joint", "joint_delta")


def _split_joint_action(values: list[float]) -> dict[str, tuple[tuple[float, ...], float]]:
    """Unpack a joint action into per-arm (positions, gripper).

    Layout mirrors `JointState.to_vector`: each arm's joints followed by its
    gripper, left then right. The width is not fixed at 16 because it follows
    the embodiment -- but both arms must report the same count, so an odd
    total or a mismatch is a caller error worth failing on.
    """
    if len(values) % 2 != 0:
        raise EvalServerError(f"Joint action must have an even length; got {len(values)}")
    half = len(values) // 2
    if half < 2:
        raise EvalServerError(f"Joint action needs at least one joint per arm; got {len(values)}")
    return {
        "left": (tuple(values[: half - 1]), values[half - 1]),
        "right": (tuple(values[half : 2 * half - 1]), values[2 * half - 1]),
    }


def _decode_gripper_target(requested: float, encoding: str = "measured_aperture") -> float:
    """Map a measured-aperture label to a force-preserving actuator command.

    Written in the numerically stable branches rather than as one expression:
    the sharpness makes the exponent large enough on real inputs (|x| ~ 300 at
    a full grip) that the naive form overflows.
    """
    if encoding == "binary_command":
        # Already a command, not a measurement: the exporter wrote 0 or 1, so
        # the boundary belongs at the midpoint of the two classes rather than
        # at the top of an aperture range 0.0108 wide.
        return decode_gripper_command(requested, encoding)
    x = GRIPPER_DECISION_SHARPNESS * (requested - GRIPPER_DECISION_CENTRE)
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    positive = math.exp(x)
    return positive / (1.0 + positive)


class ShelfRestockEvalServer:
    """Serves closed-loop episodes over the bridge protocol."""

    def __init__(
        self, port_factory, *, hz: float = DEFAULT_HZ, spec: ShelfRestockSpec = DEFAULT_SPEC
    ):
        if hz <= 0:
            raise ValueError(f"hz must be positive; got {hz}")
        self._port_factory = port_factory
        self._hz = hz
        self._spec = spec
        self._port = None
        self._profile: str | None = None
        self._episode: _Episode | None = None

    # ---------------------------------------------------------------- scene

    def _observe(self) -> RobotObservation:
        reading = self._port.read()
        joints = JointState(
            ArmJoints(reading.left.joints, reading.left.gripper),
            ArmJoints(reading.right.joints, reading.right.gripper),
        )
        return RobotObservation(
            self._episode.sim_time if self._episode is not None else 0.0,
            EEFState(
                ArmState(
                    WORLD_TO_WORKCELL.apply_pose(decode_pose(reading.left.pose_wxyz)),
                    reading.left.gripper,
                ),
                ArmState(
                    WORLD_TO_WORKCELL.apply_pose(decode_pose(reading.right.pose_wxyz)),
                    reading.right.gripper,
                ),
            ),
            *reading.cameras,
            joints=joints,
        )

    def _scene(self, observation: RobotObservation) -> ObjectScene:
        return self._episode.estimator.estimate(observation)

    def _context(self, scene: ObjectScene) -> TaskContext:
        """The task context a policy is conditioned on right now.

        `TaskContext` requires a non-empty target, so a finished shelf cannot be
        reported as a blank context; the last real one is re-reported with phase
        "done". Object tokens are derived from this context, so handing the
        policy a target that is not in the scene would silently mislabel roles.
        """
        context = self._episode.manager.next_context(scene)
        if context is not None:
            self._episode.last_context = context
            return context
        if self._episode.last_context is None:
            raise EvalServerError(
                "Scene has no restockable object at reset; the seed produced an "
                "already-complete shelf"
            )
        return TaskContext(
            instruction=self._episode.last_context.instruction,
            target_track_id=self._episode.last_context.target_track_id,
            previous_neighbor_track_id=self._episode.last_context.previous_neighbor_track_id,
            phase="done",
        )

    def _snapshot(self) -> tuple[dict, tuple[protocol.Blob, ...], ObjectScene]:
        observation = self._observe()
        scene = self._scene(observation)
        self._episode.last_eef = observation.eef
        from oct_vla.data.entity_tokens import shelf_support_entities
        from oct_vla.serve.codec import supports_to_json

        header = {
            "supports": supports_to_json(shelf_support_entities(self._episode.spec)),
            "timestamp": observation.timestamp,
            "eef": eef_to_json(observation.eef),
            "joints": joints_to_json(observation.joints),
            "scene": scene_to_json(scene),
            "context": context_to_json(self._context(scene)),
        }
        frames = (
            ("head_camera", observation.head_rgb),
            ("left_wrist_camera", observation.left_wrist_rgb),
            ("right_wrist_camera", observation.right_wrist_rgb),
        )
        blobs = tuple(
            protocol.Blob(name, "uint8", (frame.height, frame.width, 3), frame.data)
            for name, frame in frames
        )
        return header, blobs, scene

    # ------------------------------------------------------------ operations

    def reset(
        self, seed: int, profile: str, max_steps: int, control_space: str = "cartesian",
        gripper_encoding: str = "measured_aperture",
        model_ids: tuple[int, ...] | None = None,
    ) -> tuple[dict, tuple[protocol.Blob, ...]]:
        if model_ids is not None and not model_ids:
            raise EvalServerError("model_ids, when given, must name at least one variant")
        if profile not in PROFILE_TASKS:
            raise EvalServerError(f"Unknown profile {profile!r}; have {sorted(PROFILE_TASKS)}")
        if control_space not in CONTROL_SPACES:
            raise EvalServerError(
                f"Unknown control_space {control_space!r}; have {sorted(CONTROL_SPACES)}"
            )
        if gripper_encoding not in GRIPPER_ENCODINGS:
            raise EvalServerError(
                f"Unknown gripper_encoding {gripper_encoding!r}; "
                f"have {sorted(GRIPPER_ENCODINGS)}"
            )
        # Rebuild the port only when the profile changes: each one is a distinct
        # RoboTwin task class, but reconstructing SAPIEN per episode is slow and
        # leaks render contexts.
        if self._port is None or profile != self._profile:
            if self._port is not None:
                self._port.close()
            self._port = self._port_factory(PROFILE_TASKS[profile])
            self._profile = profile

        with _pinned_model_ids(model_ids):
            self._port.reset(seed)
        spawned = _spawned_model_ids(self._port.task, model_ids)
        tracked = {
            track_id: TrackedActor(
                actor=entry.actor,
                size_xyz=entry.size_xyz,
                upright_rotation=entry.upright_rotation,
                center_offset=entry.center_offset,
            )
            for track_id, entry in self._port.task.tracked_objects.items()
        }
        self._episode = _Episode(
            spec=self._spec,
            manager=ShelfRestockManager(self._spec),
            estimator=GroundTruthObjectStateEstimator(
                RoboTwinObjectEvidenceSource(tracked), WORLD_TO_WORKCELL
            ),
            max_steps=max_steps,
            control_space=control_space,
            gripper_encoding=gripper_encoding,
            events=TransferEvents(self._spec),
        )
        header, blobs, _ = self._snapshot()
        header["model_ids"] = spawned
        # How the scene's x positions were drawn. The two-object profile's
        # layout changed so its first target matches three-object training
        # (robotwin_env.py); the evaluator records this so a result on the old
        # layout cannot be pooled with one on the new.
        header["layout"] = getattr(self._port.task, "layout", "independent")
        return header, blobs

    def _terminate(
        self, episode: _Episode, reason: str, detail: str
    ) -> tuple[dict, tuple[protocol.Blob, ...]]:
        """End the episode unsuccessfully, reporting why, without advancing sim."""
        header, blobs, scene = self._snapshot()
        header.update(
            success=False,
            done=True,
            reason=reason,
            detail=detail,
            transfers_completed=len(objects_on_upper_shelf(scene, episode.spec)),
            steps=episode.steps,
            events=episode.events.report() if episode.events is not None else {},
        )
        return header, blobs

    def _step_joint(
        self, episode: _Episode, action_vector: list[float]
    ) -> tuple[dict, tuple[protocol.Blob, ...]]:
        """Command joint targets directly, absolute or as an increment.

        The whole Cartesian apparatus is absent either way, and that is the
        point: no inverse kinematics, so no command can be infeasible, and no
        deadband, because holding still is just re-commanding the same
        configuration.

        `joint_delta` adds the increment to the *measured* configuration rather
        than to a retained reference. That keeps the well-conditioned learning
        target -- one oracle step is 0.40 sigma as an increment against 0.03 as
        an absolute position -- without reintroducing an integrator: each
        command is anchored to where the arm actually is, so error cannot
        accumulate the way it did with the Cartesian reference.

        The gripper is absolute in both spaces. It is a binary actuator state
        decoded through a threshold, not a position to integrate, and the
        action carries the next *measured* aperture, which stalls mid-close
        while grasping -- so sending it back untouched would release the object.
        """
        commands = []
        for side, (positions, gripper) in _split_joint_action(action_vector).items():
            measured = self._port.arm_joints(side)
            if len(positions) != len(measured):
                raise EvalServerError(
                    f"{side} arm has {len(measured)} joints but the action supplies "
                    f"{len(positions)}; the policy's action space does not match this robot"
                )
            if episode.control_space == "joint_delta":
                positions = tuple(
                    now + step for now, step in zip(measured, positions, strict=True)
                )
            commands.append((
                side, positions, _decode_gripper_target(gripper, episode.gripper_encoding),
            ))

        episode.tick_debt += 1.0 / (self._hz * self._port.dt)
        ticks, episode.tick_debt = divmod(episode.tick_debt, 1.0)
        ticks = int(ticks)
        for tick in range(ticks):
            remaining = (ticks - tick) * self._port.dt
            for side, goal, gripper in commands:
                measured = self._port.arm_joints(side)
                velocity = tuple(
                    (want - actual) / remaining
                    for want, actual in zip(goal, measured, strict=True)
                )
                self._port.command(side, goal, velocity, gripper)
            self._port.tick()
        episode.steps += 1
        episode.sim_time += ticks * self._port.dt
        return self._report(episode)

    def _step_eef_absolute(
        self, episode: _Episode, action_vector: list[float]
    ) -> tuple[dict, tuple[protocol.Blob, ...]]:
        """Solve IK straight to a commanded end-effector pose.

        The task-space counterpart of `_step_joint`'s absolute branch, and
        deliberately the simplest path in this server: no retained reference, no
        anti-windup leash and no deadband, because an absolute pose is a place to
        go rather than a displacement to accumulate. The incremental Cartesian
        path needs all three precisely because it integrates.

        The quaternion arrives from an L1-regression head with no unit-norm
        constraint, so it is normalised here. That is a real cost of encoding
        orientation as a free quaternion and it is done at the last possible
        moment, where it is visible, rather than folded into the exporter where
        it would silently flatter the encoding.
        """
        if len(action_vector) != 16:
            raise EvalServerError(
                f"cartesian_absolute expects a 16-element pose action "
                f"(position, quaternion, gripper, per arm); got {len(action_vector)}"
            )
        commands = []
        held: dict[str, bool] = {}
        for side, offset in (("left", 0), ("right", 8)):
            block = action_vector[offset : offset + 8]
            position = tuple(block[0:3])
            norm = sum(v * v for v in block[3:7]) ** 0.5
            if norm <= 1e-8:
                # A degenerate quaternion carries no orientation at all. Holding
                # is the same no-op an unreachable pose gets, rather than
                # inventing an identity rotation the policy never asked for.
                quaternion = None
            else:
                quaternion = tuple(v / norm for v in block[3:7])
            now = self._port.arm_joints(side)
            if quaternion is None:
                joints, infeasible = now, True
                episode.infeasible_steps += 1
                episode.last_infeasible = f"{side}: degenerate quaternion"
            else:
                pose = Pose(position=position, orientation=quaternion)
                world = encode_pose(WORLD_TO_WORKCELL.inverse().apply_pose(pose))
                try:
                    joints, infeasible = self._port.ik(side, world), False
                except UnreachablePose as failure:
                    joints, infeasible = now, True
                    episode.infeasible_steps += 1
                    episode.last_infeasible = str(failure)
            held[side] = infeasible
            commands.append((
                side, now, joints,
                _decode_gripper_target(block[7], episode.gripper_encoding),
            ))

        episode.tick_debt += 1.0 / (self._hz * self._port.dt)
        ticks, episode.tick_debt = divmod(episode.tick_debt, 1.0)
        ticks = int(ticks)
        for tick in range(ticks):
            remaining = (ticks - tick) * self._port.dt
            for side, _, goal, gripper in commands:
                measured = self._port.arm_joints(side)
                velocity = tuple(
                    (want - actual) / remaining
                    for want, actual in zip(goal, measured, strict=True)
                )
                self._port.command(side, goal, velocity, gripper)
            self._port.tick()
        episode.steps += 1
        episode.sim_time += ticks * self._port.dt
        return self._report(episode)

    def step(self, action_vector: list[float]) -> tuple[dict, tuple[protocol.Blob, ...]]:
        if self._episode is None:
            raise EvalServerError("step before reset")
        episode = self._episode

        if episode.control_space in ("joint", "joint_delta"):
            return self._step_joint(episode, action_vector)
        if episode.control_space == "cartesian_absolute":
            return self._step_eef_absolute(episode, action_vector)

        action = Action.from_vector(action_vector)
        current = self._observe().eef
        reference = _leashed_reference(episode.commanded_eef, current)
        target = apply_action(reference, action)
        correction = action_between(current, target)
        # Solve both arms before commanding either. A policy that asks for an
        # unreachable pose must leave the scene untouched, or the right arm's
        # failure would still have moved the left and the episode would end in
        # a state no action produced.
        commands = []
        held: dict[str, bool] = {}
        for side, delta, arm in (
            ("left", correction.left, target.left),
            ("right", correction.right, target.right),
        ):
            now = self._port.arm_joints(side)
            translation = sum(value * value for value in delta.translation) ** 0.5
            rotation = sum(value * value for value in delta.rotation) ** 0.5
            if translation <= IK_TRANSLATION_DEADBAND and rotation <= IK_ROTATION_DEADBAND:
                joints, infeasible = now, False
            else:
                world = encode_pose(WORLD_TO_WORKCELL.inverse().apply_pose(arm.pose))
                try:
                    joints, infeasible = self._port.ik(side, world), False
                except UnreachablePose as failure:
                    # An infeasible command is a no-op for that arm, not the end
                    # of the episode. Terminating gave a policy no chance to
                    # recover from one bad action and conflated "asked for
                    # somewhere the arm cannot go" with "failed the task": the
                    # first sweep scored 98.5% of episodes this way and could
                    # not tell the two apart. A real arm ignores a command it
                    # cannot execute, so this one holds its joints, keeps the
                    # reference where the arm actually is, and lets the episode
                    # run to success or the step limit. The rate is recorded
                    # instead, as a diagnostic.
                    joints, infeasible = now, True
                    episode.infeasible_steps += 1
                    episode.last_infeasible = str(failure)
            held[side] = infeasible
            commands.append((
                side, now, joints,
                _decode_gripper_target(arm.gripper, episode.gripper_encoding),
            ))
        # Do not integrate a reference the arm was never commanded towards, or
        # the unreachable target is re-requested every step for the rest of the
        # episode and the arm never recovers.
        episode.commanded_eef = EEFState(
            current.left if held["left"] else target.left,
            current.right if held["right"] else target.right,
        )

        episode.tick_debt += 1.0 / (self._hz * self._port.dt)
        ticks, episode.tick_debt = divmod(episode.tick_debt, 1.0)
        ticks = int(ticks)
        # Reissue the command at the native 250 Hz cadence, as collection does.
        # Besides refreshing drive targets, RoboTwin's command path recomputes
        # gravity/coriolis compensation; issuing it once and then ticking the
        # scene 16--17 times leaves stale passive forces and falls behind the
        # demonstrated motion. Keep the final arm and decoded gripper targets
        # active throughout the interval.
        for tick in range(ticks):
            remaining = (ticks - tick) * self._port.dt
            for side, _start, goal, gripper in commands:
                measured = self._port.arm_joints(side)
                # Finite-horizon feedback: if the drive falls behind early in
                # the interval, increase its velocity target enough to close
                # the residual by the final native tick instead of carrying a
                # systematic fraction of every Cartesian delta as lag.
                velocity = tuple(
                    (target - actual) / remaining
                    for target, actual in zip(goal, measured, strict=True)
                )
                self._port.command(side, goal, velocity, gripper)
            self._port.tick()
        episode.steps += 1
        episode.sim_time += ticks * self._port.dt

        return self._report(episode)

    def _report(self, episode: _Episode) -> tuple[dict, tuple[protocol.Blob, ...]]:
        """Score the current scene. Shared by both control spaces, so the two
        can never disagree about what counts as success."""
        header, blobs, scene = self._snapshot()
        episode.lifted_ever.update(objects_lifted(scene, episode.spec))
        if episode.events is not None:
            episode.events.update(episode.steps, scene, episode.last_eef)
        # Mirror collection's record_placement. Ordered by arrival, and sorted
        # within a step only so that two objects arriving in the same step are
        # recorded deterministically rather than in set order.
        for track_id in sorted(objects_on_upper_shelf(scene, episode.spec)):
            if track_id not in episode.placed_seen:
                episode.placed_seen.add(track_id)
                episode.manager.record_placement(track_id)
        # Restocked, not merely cleared: see ShelfRestockManager.is_restocked.
        done_task = episode.manager.is_restocked(scene)
        out_of_steps = episode.steps >= episode.max_steps
        header.update(
            success=done_task,
            done=done_task or out_of_steps,
            reason="success" if done_task else ("step_limit" if out_of_steps else ""),
            transfers_completed=len(objects_on_upper_shelf(scene, episode.spec)),
            steps=episode.steps,
            # How often the policy asked for somewhere the arm cannot go. Now a
            # diagnostic rather than a cause of death, this separates "cannot do
            # the task" from "cannot be executed at all" -- which the first
            # sweep's single unreachable_pose outcome could not. Always zero in
            # the joint control space, where no command can be infeasible.
            infeasible_steps=episode.infeasible_steps,
            detail=episode.last_infeasible,
            # Three levels of credit, coarsest first. `objects_lifted` counts
            # anything ever picked up, `transfers_completed` anything actually
            # restocked, `success` the whole task. A policy that grasps and
            # drops is far closer to working than one that never moves, and a
            # single binary hides that entirely.
            objects_lifted=len(episode.lifted_ever),
            objects_total=len(scene.objects),
        )
        if header["done"] and episode.events is not None:
            header["events"] = episode.events.report()
        return header, blobs

    def close(self) -> None:
        if self._port is not None:
            self._port.close()
            self._port = None
        self._episode = None

    # --------------------------------------------------------------- serving

    def handle(self, connection: socket.socket) -> None:
        """Serve one client until it says close or hangs up.

        A failed operation is reported over the wire and the loop continues:
        one unreachable seed should cost that episode, not the whole evaluation
        sweep and the minutes of SAPIEN startup behind it.
        """
        while True:
            try:
                message = protocol.recv(connection)
            except protocol.ProtocolError:
                return
            op = message.header.get("op")
            if op == "close":
                return
            try:
                if op == "reset":
                    requested = message.header.get("model_ids")
                    header, blobs = self.reset(
                        int(message.header["seed"]),
                        str(message.header["profile"]),
                        int(message.header.get("max_steps", 600)),
                        str(message.header.get("control_space", "cartesian")),
                        str(message.header.get("gripper_encoding", "measured_aperture")),
                        None if requested is None else tuple(int(v) for v in requested),
                    )
                elif op == "step":
                    header, blobs = self.step(list(message.header["action"]))
                else:
                    raise EvalServerError(f"Unknown op {op!r}")
            except Exception as error:  # noqa: BLE001 - reported to the client verbatim
                traceback.print_exc()
                protocol.send_error(connection, error)
                continue
            protocol.send(connection, header, blobs)

    def serve_forever(self, host: str, port: int, *, ready_file: str | None = None) -> None:
        """Accept clients until interrupted.

        `ready_file` is written once the socket is listening. A batch script
        starting the server and the policy in one job needs a definite signal
        that the port is open, and SAPIEN plus cuRobo take long enough to load
        that a fixed sleep is either wasteful or a race.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen(1)
            if ready_file is not None:
                from pathlib import Path

                Path(ready_file).write_text(f"{host}:{port}\n")
            print(f"shelf-restock eval server listening on {host}:{port}", flush=True)
            while True:
                connection, address = listener.accept()
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"client connected from {address}", flush=True)
                with connection:
                    self.handle(connection)
                print("client disconnected", flush=True)
