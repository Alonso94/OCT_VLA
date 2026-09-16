"""Canonical boundary and checked trajectory execution for RoboTwin."""

from dataclasses import dataclass
from math import ceil, hypot, isfinite
from typing import Protocol

from oct_vla.core.action import Action, apply_action
from oct_vla.core.frames import WORKCELL_FRAME, Pose, Transform
from oct_vla.core.geometry import finite_values
from oct_vla.core.observation import RGBFrame, RobotObservation
from oct_vla.core.state import ArmJoints, ArmState, EEFState, JointState
from oct_vla.robots.base import Health, StepResult


@dataclass(frozen=True)
class ArmReading:
    pose_wxyz: tuple[float, ...]
    gripper: float
    joints: tuple[float, ...]


@dataclass(frozen=True)
class Reading:
    left: ArmReading
    right: ArmReading
    cameras: tuple[RGBFrame, RGBFrame, RGBFrame]


@dataclass(frozen=True)
class Trajectory:
    q: tuple[tuple[float, ...], ...]
    qdot: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.q or len(self.q) != len(self.qdot):
            raise ValueError("Planner trajectory must contain matching q/qdot rows")
        object.__setattr__(self, "q", tuple(finite_values(q, 7) for q in self.q))
        object.__setattr__(self, "qdot", tuple(finite_values(q, 7) for q in self.qdot))


@dataclass(frozen=True)
class Contact:
    """One contact involving a robot link. Privileged state: for the oracle,
    evaluator and diagnostics only, never for policy observations.

    `side` says which arm owns `link`. Both arms load the same Panda URDF, so
    every link name is shared between them ('panda_hand' and the rest); the
    side must therefore come from the articulation a link belongs to, never
    from its name. `other` is a scene body's name, or -- when the contact is
    with the robot itself -- the other link qualified by its own arm
    ('right/panda_hand'), so an arm-vs-arm collision cannot be misread as a
    self-collision. It was: an unqualified 'panda_rightfinger <-> panda_hand'
    from the right arm striking the parked left arm looked exactly like one
    gripper closing on itself (docs/architecture.md).
    """

    side: str
    link: str
    other: str
    impulse: float

    @property
    def qualified_link(self) -> str:
        return f"{self.side}/{self.link}"


class NativePort(Protocol):
    """Simulator-only seam; poses are world xyz+wxyz and joint rows have seven values."""

    dt: float
    #: track_id of the one object the moving arm may ignore as an obstacle,
    #: if any. An arm deliberately engaging an object -- descending onto it to
    #: grasp, or carrying it -- cannot plan against it: a grasp pose overlaps
    #: the target's own box, and a carried object sits exactly where the hand
    #: already is. Both make every plan start in collision.
    ignored_object: str | None

    def reset(self, seed: int) -> None: ...
    def read(self) -> Reading: ...
    def plan(
        self,
        side: str,
        pose_wxyz: tuple[float, ...],
        constraint: tuple[float, ...] | None = None,
    ) -> Trajectory:
        """`constraint` is cuRobo's own hold_vec_weight convention, not ours:
        6 values, indices 0-2 rotation and 3-5 translation, 1.0 holds that
        component fixed along the planned path and 0.0 leaves it free."""
        ...

    def command(
        self, side: str, q: tuple[float, ...], qdot: tuple[float, ...], gripper: float
    ) -> None: ...
    def tick(self) -> None: ...
    def hold(self) -> None: ...
    def close(self) -> None: ...

    def contacts(self) -> tuple[Contact, ...]:
        """Contacts currently involving a robot link.

        Needed because a cuRobo `Success` does not mean the trajectory was
        collision-free -- trajopt collision avoidance is a soft cost, and
        executing a "successful" plan has been observed to leave the arm
        jammed against scene geometry (docs/architecture.md).
        """
        ...


class BackendError(RuntimeError):
    pass


def decode_pose(values: tuple[float, ...]) -> Pose:
    x, y, z, w, qx, qy, qz = finite_values(values, 7)
    return Pose((x, y, z), (qx, qy, qz, w), "world")


def encode_pose(pose: Pose) -> tuple[float, ...]:
    if pose.frame != "world":
        raise ValueError("RoboTwin target must be expressed in world")
    x, y, z, w = pose.orientation
    return (*pose.position, w, x, y, z)


class RoboTwinBackend:
    def __init__(
        self,
        port: NativePort,
        world_to_workcell: Transform,
        *,
        max_translation: float = 0.05,
        max_rotation: float = 0.25,
        max_steps: int = 1000,
        settle_steps: int = 50,
    ) -> None:
        if world_to_workcell.source != "world" or world_to_workcell.target != WORKCELL_FRAME:
            raise ValueError("Expected T_workcell_world calibration")
        if not all(isfinite(v) and v > 0 for v in (port.dt, max_translation, max_rotation)):
            raise ValueError("Timestep and displacement limits must be finite and positive")
        if (
            type(max_steps) is not int
            or type(settle_steps) is not int
            or not 0 <= settle_steps < max_steps
        ):
            raise ValueError("Invalid execution step budget")
        self.port, self.transform = port, world_to_workcell
        self.max_translation, self.max_rotation = max_translation, max_rotation
        self.max_steps, self.settle_steps = max_steps, settle_steps
        self._ticks = 0
        self._ready = False

    def reset(self, seed: int) -> RobotObservation:
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        self._ready = False
        self.port.reset(seed)
        self._ticks = 0
        observation = self.observe()
        self._ready = True
        return observation

    def _observation(self, reading: Reading) -> RobotObservation:
        arms = [
            ArmState(self.transform.apply_pose(decode_pose(arm.pose_wxyz)), arm.gripper)
            for arm in (reading.left, reading.right)
        ]
        joints = JointState(
            ArmJoints(reading.left.joints, reading.left.gripper),
            ArmJoints(reading.right.joints, reading.right.gripper),
        )
        return RobotObservation(
            self._ticks * self.port.dt, EEFState(*arms), *reading.cameras, joints=joints
        )

    def observe(self) -> RobotObservation:
        return self._observation(self.port.read())

    def step(self, action: Action) -> StepResult:
        if not self._ready:
            raise BackendError("Backend stopped or not reset")
        for arm in (action.left, action.right):
            if (
                hypot(*arm.translation) > self.max_translation
                or hypot(*arm.rotation) > self.max_rotation
            ):
                raise ValueError("Action exceeds configured displacement bounds; no command sent")
        reading = self.port.read()
        current = self._observation(reading)
        target = apply_action(current.eef, action)
        paths = []
        try:
            # Plan both before sending any command, so failure cannot move one arm only.
            for side, delta, arm, measured in zip(
                ("left", "right"),
                (action.left, action.right),
                (target.left, target.right),
                (reading.left, reading.right),
                strict=True,
            ):
                if hypot(*delta.translation, *delta.rotation) == 0:
                    paths.append(Trajectory((measured.joints,), ((0.0,) * 7,)))
                else:
                    paths.append(
                        self.port.plan(
                            side, encode_pose(self.transform.inverse().apply_pose(arm.pose))
                        )
                    )
            count = max(ceil((1 / 15) / self.port.dt), *(len(path.q) for path in paths))
            count += self.settle_steps
            if count > self.max_steps:
                raise BackendError("EXECUTION_BUDGET_EXCEEDED")
            for index in range(count):
                for side, path, delta, measured in zip(
                    ("left", "right"),
                    paths,
                    (action.left, action.right),
                    (reading.left, reading.right),
                    strict=True,
                ):
                    q = path.q[min(index, len(path.q) - 1)]
                    qdot = path.qdot[index] if index < len(path.qdot) else (0.0,) * 7
                    fraction = min(1.0, (index + 1) / (count - self.settle_steps))
                    gripper = measured.gripper + fraction * (delta.gripper - measured.gripper)
                    self.port.command(side, q, qdot, gripper)
                self.port.tick()
                self._ticks += 1
            observation = self.observe()
        except Exception:
            self.stop()
            raise
        return StepResult(observation, action, observation.timestamp - current.timestamp)

    def stop(self) -> None:
        self._ready = False
        self.port.hold()

    def health(self) -> Health:
        return Health(self._ready, "ready" if self._ready else "stopped or not reset")

    def close(self) -> None:
        self._ready = False
        self.port.close()
