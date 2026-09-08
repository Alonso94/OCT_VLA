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

from oct_vla.core.frames import Pose
from oct_vla.perception.ground_truth import RawObjectEvidence
from oct_vla.robots.robotwin.assets import centered_upright_pose
from oct_vla.robots.robotwin.backend import decode_pose


class SapienPose(Protocol):
    p: tuple[float, float, float]
    q: tuple[float, float, float, float]


class SapienActor(Protocol):
    """The one RoboTwin/SAPIEN capability this reader needs."""

    def get_pose(self) -> SapienPose: ...


@dataclass(frozen=True)
class TrackedActor:
    """One actor to read, plus what is needed to report it canonically.

    `upright_rotation` is the asset's own mesh-frame-to-upright rotation, and
    is divided out of the measured pose so the reported orientation describes
    the *object*, not the mesh the artist happened to author. RoboTwin's
    assets are y-up and are spawned with a fixed base quaternion; leaving
    that in would mean every downstream consumer of `ObjectState.pose` had to
    know which asset it came from, and code that reasonably assumes an
    upright object has a yaw-only orientation (grasp generation does) would
    read a meaningless yaw out of a 120-degree tilt.

    `center_offset` is where the object's bounding-box centre sits relative
    to the mesh's origin, in that same upright frame. RoboTwin's assets are
    authored with their origin at the object's base, and the canonical
    `ObjectState.pose` is defined as the object's centre, so this is added
    back on. Consumers rely on it: `holding_tcp_pose` derives an object's top
    face as `position + height / 2`, which is half an object too low if the
    position is really the base.

    `size_xyz` is likewise expected in that upright frame, not in mesh axes.
    """

    actor: SapienActor
    size_xyz: tuple[float, float, float]
    support_surface: str | None = None
    visibility: float = 1.0
    confidence: float = 1.0
    upright_rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


class RoboTwinObjectEvidenceSource:
    def __init__(self, tracked: Mapping[str, TrackedActor]) -> None:
        self._tracked = dict(tracked)

    def read(self) -> tuple[RawObjectEvidence, ...]:
        evidence = []
        for track_id, entry in self._tracked.items():
            pose = entry.actor.get_pose()
            measured = decode_pose((*pose.p, *pose.q))
            position, orientation = centered_upright_pose(
                measured.position,
                measured.orientation,
                entry.upright_rotation,
                entry.center_offset,
            )
            upright = Pose(position, orientation, measured.frame)
            evidence.append(
                RawObjectEvidence(
                    track_id=track_id,
                    pose=upright,
                    size_xyz=entry.size_xyz,
                    visibility=entry.visibility,
                    confidence=entry.confidence,
                    support_surface=entry.support_surface,
                )
            )
        return tuple(evidence)
