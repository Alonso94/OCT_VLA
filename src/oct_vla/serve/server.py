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
import socket
import traceback
from dataclasses import dataclass

from oct_vla.core.action import Action, action_between, apply_action
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
from oct_vla.tasks.shelf_restock.manager import ShelfRestockManager, objects_on_upper_shelf
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
    #: "cartesian" (14-d canonical increments, executed through IK) or "joint"
    #: (absolute joint targets, commanded directly). Declared at reset rather
    #: than inferred, so a mismatch is an error instead of a misreading.
    control_space: str = "cartesian"


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


CONTROL_SPACES = ("cartesian", "joint")


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


def _decode_gripper_target(requested: float) -> float:
    """Map a measured-aperture label to a force-preserving actuator command.

    Written in the numerically stable branches rather than as one expression:
    the sharpness makes the exponent large enough on real inputs (|x| ~ 300 at
    a full grip) that the naive form overflows.
    """
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
        header = {
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
        self, seed: int, profile: str, max_steps: int, control_space: str = "cartesian"
    ) -> tuple[dict, tuple[protocol.Blob, ...]]:
        if profile not in PROFILE_TASKS:
            raise EvalServerError(f"Unknown profile {profile!r}; have {sorted(PROFILE_TASKS)}")
        if control_space not in CONTROL_SPACES:
            raise EvalServerError(
                f"Unknown control_space {control_space!r}; have {sorted(CONTROL_SPACES)}"
            )
        # Rebuild the port only when the profile changes: each one is a distinct
        # RoboTwin task class, but reconstructing SAPIEN per episode is slow and
        # leaks render contexts.
        if self._port is None or profile != self._profile:
            if self._port is not None:
                self._port.close()
            self._port = self._port_factory(PROFILE_TASKS[profile])
            self._profile = profile

        self._port.reset(seed)
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
        )
        header, blobs, _ = self._snapshot()
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
        )
        return header, blobs

    def _step_joint(
        self, episode: _Episode, action_vector: list[float]
    ) -> tuple[dict, tuple[protocol.Blob, ...]]:
        """Command absolute joint targets directly.

        The whole Cartesian apparatus is absent here, and that is the point:
        no inverse kinematics, so no pose can be infeasible; no integrated
        reference, because an absolute target has nothing to accumulate; no
        deadband, because holding still is just re-commanding the same
        configuration. What remains is the gripper decode -- the action carries
        the next *measured* aperture, which stalls mid-close while grasping, so
        sending it back as a drive target would release the object.
        """
        commands = []
        for side, (positions, gripper) in _split_joint_action(action_vector).items():
            measured = self._port.arm_joints(side)
            if len(positions) != len(measured):
                raise EvalServerError(
                    f"{side} arm has {len(measured)} joints but the action supplies "
                    f"{len(positions)}; the policy's action space does not match this robot"
                )
            commands.append((side, positions, _decode_gripper_target(gripper)))

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

    def step(self, action_vector: list[float]) -> tuple[dict, tuple[protocol.Blob, ...]]:
        if self._episode is None:
            raise EvalServerError("step before reset")
        episode = self._episode

        if episode.control_space == "joint":
            return self._step_joint(episode, action_vector)

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
            commands.append((side, now, joints, _decode_gripper_target(arm.gripper)))
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
        )
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
                    header, blobs = self.reset(
                        int(message.header["seed"]),
                        str(message.header["profile"]),
                        int(message.header.get("max_steps", 600)),
                        str(message.header.get("control_space", "cartesian")),
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
