from dataclasses import dataclass

import pytest

from oct_vla.perception.robotwin.evidence import RoboTwinObjectEvidenceSource, TrackedActor


@dataclass(frozen=True)
class FakeSapienPose:
    p: tuple[float, float, float]
    q: tuple[float, float, float, float]  # wxyz, matching SAPIEN/transforms3d


class FakeActor:
    def __init__(self, pose: FakeSapienPose) -> None:
        self._pose = pose

    def get_pose(self) -> FakeSapienPose:
        return self._pose


def test_read_decodes_wxyz_pose_into_world_frame_pose():
    actor = FakeActor(FakeSapienPose((0.1, 0.2, 0.3), (1.0, 0.0, 0.0, 0.0)))
    source = RoboTwinObjectEvidenceSource(
        {"a": TrackedActor(actor=actor, size_xyz=(0.05, 0.05, 0.1))}
    )

    (evidence,) = source.read()

    assert evidence.track_id == "a"
    assert evidence.pose.frame == "world"
    assert evidence.pose.position == pytest.approx((0.1, 0.2, 0.3))
    assert evidence.pose.orientation == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert evidence.size_xyz == pytest.approx((0.05, 0.05, 0.1))


def test_read_preserves_metadata_and_defaults():
    actor = FakeActor(FakeSapienPose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
    source = RoboTwinObjectEvidenceSource(
        {
            "target": TrackedActor(
                actor=actor,
                size_xyz=(0.02, 0.02, 0.02),
                support_surface="upper_shelf",
                visibility=0.8,
                confidence=0.9,
            )
        }
    )

    (evidence,) = source.read()

    assert evidence.support_surface == "upper_shelf"
    assert evidence.visibility == pytest.approx(0.8)
    assert evidence.confidence == pytest.approx(0.9)


def test_read_reflects_current_pose_each_call_not_a_cached_snapshot():
    pose = FakeSapienPose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    actor = FakeActor(pose)
    source = RoboTwinObjectEvidenceSource(
        {"a": TrackedActor(actor=actor, size_xyz=(0.1, 0.1, 0.1))}
    )

    assert source.read()[0].pose.position == pytest.approx((0.0, 0.0, 0.0))
    actor._pose = FakeSapienPose((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    assert source.read()[0].pose.position == pytest.approx((1.0, 0.0, 0.0))


def test_read_returns_multiple_tracked_actors_independently():
    a = FakeActor(FakeSapienPose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
    b = FakeActor(FakeSapienPose((1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 0.0)))
    source = RoboTwinObjectEvidenceSource(
        {
            "a": TrackedActor(actor=a, size_xyz=(0.1, 0.1, 0.1)),
            "b": TrackedActor(actor=b, size_xyz=(0.2, 0.2, 0.2)),
        }
    )

    evidence = {e.track_id: e for e in source.read()}

    assert set(evidence) == {"a", "b"}
    assert evidence["b"].pose.position == pytest.approx((1.0, 1.0, 1.0))
