from math import pi

import pytest

from oct_vla.core.frames import WORKCELL_FRAME, Pose, Transform
from oct_vla.core.geometry import exp
from oct_vla.core.observation import RGBFrame, RobotObservation
from oct_vla.core.state import ArmState, EEFState
from oct_vla.perception.ground_truth import GroundTruthObjectStateEstimator, RawObjectEvidence

# A nontrivial transform (translation + 90 deg yaw), matching the RoboTwin
# backend tests: this must not pass by accident on an identity mapping.
WORLD_TO_WORKCELL = Transform("world", WORKCELL_FRAME, (0.0, 0.0, 0.74), exp((0.0, 0.0, pi / 2)))


class FakeSource:
    def __init__(self, evidence: tuple[RawObjectEvidence, ...]) -> None:
        self._evidence = evidence

    def read(self) -> tuple[RawObjectEvidence, ...]:
        return self._evidence


def observation(timestamp: float = 1.5) -> RobotObservation:
    arm = ArmState(Pose((0, 0, 0), (0, 0, 0, 1)), 0.5)
    frame = RGBFrame(1, 1, bytes(3))
    return RobotObservation(timestamp, EEFState(arm, arm), frame, frame, frame)


def world_evidence(track_id="obj", position=(0.4, 0.2, 0.3)) -> RawObjectEvidence:
    return RawObjectEvidence(
        track_id=track_id,
        pose=Pose(position, (0, 0, 0, 1), "world"),
        size_xyz=(0.05, 0.05, 0.1),
    )


def test_constructor_rejects_transform_not_targeting_workcell():
    with pytest.raises(ValueError):
        GroundTruthObjectStateEstimator(
            FakeSource(()), Transform("world", "not-workcell", (0, 0, 0), (0, 0, 0, 1))
        )


def test_estimate_transforms_pose_into_workcell_frame():
    estimator = GroundTruthObjectStateEstimator(FakeSource((world_evidence(),)), WORLD_TO_WORKCELL)

    scene = estimator.estimate(observation())

    assert len(scene.objects) == 1
    obj = scene.objects[0]
    assert obj.pose.frame == WORKCELL_FRAME
    # World (0.4, 0.2, 0.3) rotated +90deg about Z, then offset by (0, 0, 0.74).
    assert obj.pose.position == pytest.approx((-0.2, 0.4, 1.04), abs=1e-9)


def test_estimate_propagates_observation_timestamp():
    estimator = GroundTruthObjectStateEstimator(FakeSource(()), WORLD_TO_WORKCELL)
    scene = estimator.estimate(observation(timestamp=3.25))
    assert scene.timestamp == pytest.approx(3.25)


def test_estimate_preserves_all_evidence_fields():
    evidence = RawObjectEvidence(
        track_id="target",
        pose=Pose((0.0, 0.0, 0.0), (0, 0, 0, 1), "world"),
        size_xyz=(0.03, 0.04, 0.05),
        visibility=0.7,
        confidence=0.6,
        support_surface="lower_shelf",
    )
    estimator = GroundTruthObjectStateEstimator(FakeSource((evidence,)), WORLD_TO_WORKCELL)

    obj = estimator.estimate(observation()).objects[0]

    assert obj.track_id == "target"
    assert obj.size_xyz == pytest.approx((0.03, 0.04, 0.05))
    assert obj.visibility == pytest.approx(0.7)
    assert obj.confidence == pytest.approx(0.6)
    assert obj.support_surface == "lower_shelf"


def test_estimate_rejects_duplicate_track_ids_from_source():
    estimator = GroundTruthObjectStateEstimator(
        FakeSource((world_evidence("a"), world_evidence("a"))), WORLD_TO_WORKCELL
    )
    with pytest.raises(ValueError):
        estimator.estimate(observation())


def test_estimate_handles_empty_scene():
    estimator = GroundTruthObjectStateEstimator(FakeSource(()), WORLD_TO_WORKCELL)
    scene = estimator.estimate(observation())
    assert scene.objects == ()
