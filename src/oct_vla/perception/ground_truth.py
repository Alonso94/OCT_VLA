"""Ground-truth object-state estimation from explicitly supplied evidence.

Trusts its evidence source completely -- this estimator is simulation-only
and must never be wired to a real-hardware backend. It performs no perception
of its own, only frame conversion and canonical-schema validation, so it is
reusable for any simulator whose privileged state can be expressed as
RawObjectEvidence (see perception/robotwin/evidence.py for the RoboTwin source).
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from oct_vla.core.frames import WORKCELL_FRAME, Pose, Transform
from oct_vla.core.objects import ObjectScene, ObjectState
from oct_vla.core.observation import RobotObservation


@dataclass(frozen=True)
class RawObjectEvidence:
    """One object's privileged state, in the evidence source's own frame."""

    track_id: str
    pose: Pose
    size_xyz: tuple[float, float, float]
    visibility: float = 1.0
    confidence: float = 1.0
    support_surface: str | None = None
    category: str | None = None


@runtime_checkable
class ObjectEvidenceSource(Protocol):
    """Simulation-only seam; implementations may read privileged simulator state."""

    def read(self) -> tuple[RawObjectEvidence, ...]: ...


class GroundTruthObjectStateEstimator:
    def __init__(self, source: ObjectEvidenceSource, source_to_workcell: Transform) -> None:
        if source_to_workcell.target != WORKCELL_FRAME:
            raise ValueError("Expected a transform into the workcell frame")
        self.source = source
        self.transform = source_to_workcell

    def estimate(self, observation: RobotObservation) -> ObjectScene:
        objects = tuple(
            ObjectState(
                track_id=evidence.track_id,
                pose=self.transform.apply_pose(evidence.pose),
                size_xyz=evidence.size_xyz,
                visibility=evidence.visibility,
                confidence=evidence.confidence,
                support_surface=evidence.support_surface,
                category=evidence.category,
            )
            for evidence in self.source.read()
        )
        return ObjectScene(observation.timestamp, objects)
