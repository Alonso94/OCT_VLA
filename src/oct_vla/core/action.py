"""The single learned action contract: left then right Cartesian increments."""

from collections.abc import Iterable
from dataclasses import dataclass

from .frames import Pose
from .geometry import Vector3, exp, finite_values, inverse, log, multiply, vector3
from .state import ArmState, EEFState, gripper_command

ACTION_DIM = 14


@dataclass(frozen=True)
class ArmAction:
    translation: Vector3
    rotation: Vector3
    gripper: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "translation", vector3(self.translation))
        object.__setattr__(self, "rotation", vector3(self.rotation))
        object.__setattr__(self, "gripper", gripper_command(self.gripper))


@dataclass(frozen=True)
class Action:
    """Workcell-frame meters/radians per step, absolute normalized grippers."""

    left: ArmAction
    right: ArmAction

    def __post_init__(self) -> None:
        if not isinstance(self.left, ArmAction) or not isinstance(self.right, ArmAction):
            raise ValueError("Action requires left and right ArmAction values")

    @classmethod
    def from_vector(cls, values: Iterable[float]) -> "Action":
        """Ingest a policy's raw output. The gripper channel saturates here.

        This is the boundary where a regressed number becomes a command, and a
        regressor has no reason to respect the range: under `binary_command`
        the targets are exactly {0, 1}, so an L1 head overshoots past both ends
        routinely. Letting that reach ArmAction's validator ends the episode
        with "Gripper must be finite and within [0, 1]" -- which is how the
        EE-delta arm scored nothing at all while the two absolute arms, whose
        gripper goes through a threshold instead, were unaffected.

        Saturating rather than rejecting, because that is what the hardware
        does: a gripper asked to close harder than fully closed is simply fully
        closed. Only the gripper is clamped; an out-of-range *pose* increment is
        a real request the IK solver should be allowed to refuse and record as
        infeasible.
        """
        values = finite_values(values, ACTION_DIM)
        return cls(
            *(
                ArmAction(
                    vector3(values[i : i + 3]),
                    vector3(values[i + 3 : i + 6]),
                    min(1.0, max(0.0, values[i + 6])),
                )
                for i in (0, 7)
            )
        )

    def to_vector(self) -> tuple[float, ...]:
        return (
            *self.left.translation,
            *self.left.rotation,
            self.left.gripper,
            *self.right.translation,
            *self.right.rotation,
            self.right.gripper,
        )

    @classmethod
    def hold(cls, state: EEFState) -> "Action":
        return cls(
            ArmAction((0, 0, 0), (0, 0, 0), state.left.gripper),
            ArmAction((0, 0, 0), (0, 0, 0), state.right.gripper),
        )


def apply_action(state: EEFState, action: Action) -> EEFState:
    """Compose without clipping; robot-specific bounds belong in the backend."""

    def apply(arm: ArmState, delta: ArmAction) -> ArmState:
        position = vector3(
            tuple(a + b for a, b in zip(arm.pose.position, delta.translation, strict=True))
        )
        orientation = multiply(exp(delta.rotation), arm.pose.orientation)
        return ArmState(Pose(position, orientation), delta.gripper)

    return EEFState(apply(state.left, action.left), apply(state.right, action.right))


def action_between(current: EEFState, following: EEFState) -> Action:
    """Compute dp and Log(R_next R_current^T), with next absolute grippers."""

    def between(start: ArmState, end: ArmState) -> ArmAction:
        translation = vector3(
            tuple(b - a for a, b in zip(start.pose.position, end.pose.position, strict=True))
        )
        rotation = log(multiply(end.pose.orientation, inverse(start.pose.orientation)))
        return ArmAction(translation, rotation, end.gripper)

    return Action(between(current.left, following.left), between(current.right, following.right))
