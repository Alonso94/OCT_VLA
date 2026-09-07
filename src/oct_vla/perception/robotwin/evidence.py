"""Reads RoboTwin/SAPIEN actor state as ground-truth evidence; simulation-only.

Track_id assignment and per-object size/support metadata come from the caller
(future task setup code), never from RoboTwin's own actor name: ``create_actor``
gives every instance of one asset the identical semantic model name, so it
cannot distinguish instances within an episode and must never be treated as
a track_id.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from oct_vla.perception.ground_truth import RawObjectEvidence
from oct_vla.robots.robotwin.backend import decode_pose


class SapienPose(Protocol):
    p: tuple[float, float, float]
    q: tuple[float, float, float, float]


class SapienActor(Protocol):
    """The one RoboTwin/SAPIEN capability this reader needs."""

    def get_pose(self) -> SapienPose: ...


@dataclass(frozen=True)
class TrackedActor:
    actor: SapienActor
    size_xyz: tuple[float, float, float]
    support_surface: str | None = None
    visibility: float = 1.0
    confidence: float = 1.0


class RoboTwinObjectEvidenceSource:
    def __init__(self, tracked: Mapping[str, TrackedActor]) -> None:
        self._tracked = dict(tracked)

    def read(self) -> tuple[RawObjectEvidence, ...]:
        evidence = []
        for track_id, entry in self._tracked.items():
            pose = entry.actor.get_pose()
            evidence.append(
                RawObjectEvidence(
                    track_id=track_id,
                    pose=decode_pose((*pose.p, *pose.q)),
                    size_xyz=entry.size_xyz,
                    visibility=entry.visibility,
                    confidence=entry.confidence,
                    support_surface=entry.support_surface,
                )
            )
        return tuple(evidence)
