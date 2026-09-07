"""Canonical measured end-effector state (not a joint-action representation)."""

from dataclasses import dataclass

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
