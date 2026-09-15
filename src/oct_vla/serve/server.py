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

import socket
import traceback
from dataclasses import dataclass

from oct_vla.core.action import Action, apply_action
from oct_vla.core.objects import ObjectScene, TaskContext
from oct_vla.core.observation import RobotObservation
from oct_vla.core.state import ArmState, EEFState
from oct_vla.perception.ground_truth import GroundTruthObjectStateEstimator
from oct_vla.perception.robotwin.evidence import RoboTwinObjectEvidenceSource, TrackedActor
from oct_vla.robots.robotwin.backend import decode_pose, encode_pose
from oct_vla.robots.robotwin.native import UnreachablePose
from oct_vla.serve import protocol
from oct_vla.serve.codec import context_to_json, eef_to_json, scene_to_json
from oct_vla.tasks.shelf_restock.collect import DEFAULT_HZ, WORLD_TO_WORKCELL
from oct_vla.tasks.shelf_restock.manager import ShelfRestockManager, objects_on_upper_shelf
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC, ShelfRestockSpec

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
        self, seed: int, profile: str, max_steps: int
    ) -> tuple[dict, tuple[protocol.Blob, ...]]:
        if profile not in PROFILE_TASKS:
            raise EvalServerError(f"Unknown profile {profile!r}; have {sorted(PROFILE_TASKS)}")
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

    def step(self, action_vector: list[float]) -> tuple[dict, tuple[protocol.Blob, ...]]:
        if self._episode is None:
            raise EvalServerError("step before reset")
        episode = self._episode

        current = self._observe().eef
        target = apply_action(current, Action.from_vector(action_vector))
        # Solve both arms before commanding either. A policy that asks for an
        # unreachable pose must leave the scene untouched, or the right arm's
        # failure would still have moved the left and the episode would end in
        # a state no action produced.
        commands = []
        try:
            for side, arm in (("left", target.left), ("right", target.right)):
                world = encode_pose(WORLD_TO_WORKCELL.inverse().apply_pose(arm.pose))
                joints = self._port.ik(side, world)
                # Velocity feedforward, not zero. RoboTwin's set_arm_joints drives
                # both a position and a velocity target, so commanding zero velocity
                # asks the arm to *arrive at rest* at the new pose: within one 15 Hz
                # control step it decelerates and covers only a fraction of the
                # commanded displacement. Because each step's target is rebuilt from
                # the freshly measured pose, that shortfall does not accumulate into
                # a lag -- it silently rescales every action, so a policy replaying
                # its training actions would crawl. Asking for the speed that
                # actually covers the gap in one step removes the built-in brake.
                now = self._port.arm_joints(side)
                rate = self._hz
                velocity = tuple(
                    (goal - start) * rate for goal, start in zip(joints, now, strict=True)
                )
                commands.append((side, joints, velocity, arm.gripper))
        except UnreachablePose as failure:
            # An outcome, not an error: score the episode as failed here and let
            # the evaluation move on to the next seed. Raising instead would
            # abort the whole run and discard every episode already scored --
            # and an undertrained policy commands unreachable poses routinely,
            # so the arms that most need measuring are the ones that never
            # produce a result.
            return self._terminate(episode, "unreachable_pose", str(failure))
        for side, joints, velocity, gripper in commands:
            self._port.command(side, joints, velocity, gripper)

        episode.tick_debt += 1.0 / (self._hz * self._port.dt)
        ticks, episode.tick_debt = divmod(episode.tick_debt, 1.0)
        for _ in range(int(ticks)):
            self._port.tick()
        episode.steps += 1
        episode.sim_time += int(ticks) * self._port.dt

        header, blobs, scene = self._snapshot()
        done_task = episode.manager.is_done(scene)
        out_of_steps = episode.steps >= episode.max_steps
        header.update(
            success=done_task,
            done=done_task or out_of_steps,
            reason="success" if done_task else ("step_limit" if out_of_steps else ""),
            transfers_completed=len(objects_on_upper_shelf(scene, episode.spec)),
            steps=episode.steps,
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
