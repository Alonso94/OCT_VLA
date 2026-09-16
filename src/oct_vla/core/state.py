"""Canonical measured end-effector state (not a joint-action representation)."""

from dataclasses import dataclass
from math import isfinite

from .frames import WORKCELL_FRAME, Pose


def gripper_command(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("Gripper must be finite and within [0, 1]")
    return value


@dataclass(frozen=True)
class ArmState:
    pose: Pose
    gripper: float

    def __post_init__(self) -> None:
        if not isinstance(self.pose, Pose) or self.pose.frame != WORKCELL_FRAME:
            raise ValueError("Canonical EEF state requires a workcell Pose")
        object.__setattr__(self, "gripper", gripper_command(self.gripper))


@dataclass(frozen=True)
class EEFState:
    left: ArmState
    right: ArmState

    def __post_init__(self) -> None:
        if not isinstance(self.left, ArmState) or not isinstance(self.right, ArmState):
            raise ValueError("EEFState requires left and right ArmState values")


@dataclass(frozen=True)
class ArmJoints:
    """Measured arm joint positions plus the normalized gripper aperture.

    Deliberately separate from `ArmState`, which is the Cartesian abstraction
    and, as docs/canonical_action.md states, carries no joint targets. Joints
    are recorded alongside it rather than folded into it because they are
    embodiment-specific: the Cartesian state is comparable across robots, these
    are not.

    Recorded because the oracle *plans and executes in joint space* -- the
    Cartesian action is a derived re-encoding of it. Keeping the joints makes
    the original control signal available to a policy instead of only its
    round-trip through inverse kinematics.
    """

    positions: tuple[float, ...]
    gripper: float

    def __post_init__(self) -> None:
        positions = tuple(float(v) for v in self.positions)
        if not positions:
            raise ValueError("ArmJoints requires at least one joint position")
        if not all(isfinite(v) for v in positions):
            raise ValueError("Joint positions must be finite")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "gripper", gripper_command(self.gripper))


@dataclass(frozen=True)
class JointState:
    """Both arms' measured joint configuration at one instant."""

    left: ArmJoints
    right: ArmJoints

    def __post_init__(self) -> None:
        if not isinstance(self.left, ArmJoints) or not isinstance(self.right, ArmJoints):
            raise ValueError("JointState requires left and right ArmJoints values")
        if len(self.left.positions) != len(self.right.positions):
            raise ValueError(
                "Both arms must report the same number of joints; got "
                f"{len(self.left.positions)} and {len(self.right.positions)}"
            )

    def to_vector(self) -> tuple[float, ...]:
        """Flat left-then-right layout, each arm's joints followed by its gripper."""
        return (
            *self.left.positions,
            self.left.gripper,
            *self.right.positions,
            self.right.gripper,
        )
