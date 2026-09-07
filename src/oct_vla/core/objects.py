"""Canonical object-centric scene: identity and geometry only, never task role.

Task roles (target, previous neighbor, other) are derived from a TaskContext
against an ObjectScene's track_ids, never stored on ObjectState itself. This
keeps compositional-generalization experiments honest: nothing here is a
permanent semantic label like an asset name, only an episode-local track_id.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .frames import WORKCELL_FRAME, Pose
from .geometry import finite_values


def _unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return value


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class ObjectState:
    """One tracked object: episode-local identity, workcell pose, own-frame size."""

    track_id: str
    pose: Pose
    size_xyz: tuple[float, float, float]
    visibility: float
    confidence: float
    support_surface: str | None = None
    mask: bytes | None = None
    embedding: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _nonempty(self.track_id, "track_id"))
        if not isinstance(self.pose, Pose) or self.pose.frame != WORKCELL_FRAME:
            raise ValueError("ObjectState requires a workcell Pose")
        size = finite_values(self.size_xyz, 3)
        if not all(v > 0 for v in size):
            raise ValueError("size_xyz must be finite and positive")
        object.__setattr__(self, "size_xyz", size)
        object.__setattr__(self, "visibility", _unit_interval(self.visibility, "visibility"))
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, "confidence"))
        if self.support_surface is not None:
            object.__setattr__(
                self, "support_surface", _nonempty(self.support_surface, "support_surface")
            )
        if self.embedding is not None:
            object.__setattr__(
                self, "embedding", finite_values(self.embedding, len(self.embedding))
            )


@dataclass(frozen=True)
class ObjectScene:
    """One estimator's read of the scene at a single instant."""

    timestamp: float
    objects: tuple[ObjectState, ...]

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("Timestamp must be finite and nonnegative")
        if not all(isinstance(o, ObjectState) for o in self.objects):
            raise ValueError("ObjectScene requires only ObjectState objects")
        track_ids = [o.track_id for o in self.objects]
        if len(set(track_ids)) != len(track_ids):
            raise ValueError("ObjectScene track_ids must be unique within the scene")

    def get(self, track_id: str) -> ObjectState | None:
        return next((o for o in self.objects if o.track_id == track_id), None)


@dataclass(frozen=True)
class TaskContext:
    """Task-side reference to objects, by track_id only; never an object's own field."""

    instruction: str
    target_track_id: str
    previous_neighbor_track_id: str | None = None
    phase: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruction", _nonempty(self.instruction, "instruction"))
        object.__setattr__(
            self, "target_track_id", _nonempty(self.target_track_id, "target_track_id")
        )
        if self.previous_neighbor_track_id is not None:
            object.__setattr__(
                self,
                "previous_neighbor_track_id",
                _nonempty(self.previous_neighbor_track_id, "previous_neighbor_track_id"),
            )
            if self.previous_neighbor_track_id == self.target_track_id:
                raise ValueError("previous_neighbor_track_id must differ from target_track_id")
        if self.phase is not None:
            object.__setattr__(self, "phase", _nonempty(self.phase, "phase"))


class ObjectRole(Enum):
    TARGET = "target"
    PREVIOUS_NEIGHBOR = "previous_neighbor"
    OTHER = "other"


def role_of(context: TaskContext, track_id: str) -> ObjectRole:
    """Role is computed from context each call; it is never stored on ObjectState."""
    if track_id == context.target_track_id:
        return ObjectRole.TARGET
    if track_id == context.previous_neighbor_track_id:
        return ObjectRole.PREVIOUS_NEIGHBOR
    return ObjectRole.OTHER
