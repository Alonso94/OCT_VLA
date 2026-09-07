"""Frame-labelled rigid transforms: T_A_B maps coordinates B into A."""

from dataclasses import dataclass

from .geometry import Quaternion, Vector3, inverse, multiply, rotate, unit_quaternion, vector3

WORKCELL_FRAME = "workcell"


def _frame(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Frame name must be a nonempty string")


@dataclass(frozen=True)
class Pose:
    """Tool/object position and orientation expressed in the named frame."""

    position: Vector3
    orientation: Quaternion
    frame: str = WORKCELL_FRAME

    def __post_init__(self) -> None:
        _frame(self.frame)
        object.__setattr__(self, "position", vector3(self.position))
        object.__setattr__(self, "orientation", unit_quaternion(self.orientation))


@dataclass(frozen=True)
class Transform:
    """Rigid transform from source coordinates to target coordinates."""

    source: str
    target: str
    translation: Vector3
    rotation: Quaternion

    def __post_init__(self) -> None:
        _frame(self.source)
        _frame(self.target)
        object.__setattr__(self, "translation", vector3(self.translation))
        object.__setattr__(self, "rotation", unit_quaternion(self.rotation))

    def apply_vector(self, vector: Vector3) -> Vector3:
        """Change coordinates of a free vector (including dp or spatial dr)."""
        return rotate(self.rotation, vector)

    def apply_point(self, point: Vector3) -> Vector3:
        rotated = self.apply_vector(point)
        return vector3(tuple(a + b for a, b in zip(rotated, self.translation, strict=True)))

    def apply_pose(self, pose: Pose) -> Pose:
        if pose.frame != self.source:
            raise ValueError(f"Expected pose in {self.source}, got {pose.frame}")
        return Pose(
            self.apply_point(pose.position), multiply(self.rotation, pose.orientation), self.target
        )

    def inverse(self) -> "Transform":
        rotation = inverse(self.rotation)
        translation = rotate(rotation, vector3(tuple(-v for v in self.translation)))
        return Transform(self.target, self.source, translation, rotation)

    def compose(self, other: "Transform") -> "Transform":
        """Return self @ other: apply other first, then self."""
        if other.target != self.source:
            raise ValueError("Transform chain has mismatched frames")
        return Transform(
            other.source,
            self.target,
            self.apply_point(other.translation),
            multiply(self.rotation, other.rotation),
        )
