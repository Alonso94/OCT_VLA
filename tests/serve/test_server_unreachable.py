"""An unreachable commanded pose must end the episode, not the evaluation.

This is a regression test for a failure that cost a whole evaluation job: the
IK solver raised, the exception crossed the bridge as a protocol error, and the
client died on its first bad action -- discarding every episode scored before
it. Undertrained policies command unreachable poses constantly, so the arms the
sweep most needs to measure were exactly the ones that produced no results.
"""

from __future__ import annotations

from oct_vla.core.frames import Pose
from oct_vla.core.objects import ObjectScene, ObjectState
from oct_vla.core.observation import RGBFrame
from oct_vla.robots.robotwin.backend import ArmReading, Reading
from oct_vla.robots.robotwin.native import NativePortError, UnreachablePose
from oct_vla.serve.server import ShelfRestockEvalServer, _Episode
from oct_vla.tasks.shelf_restock.manager import ShelfRestockManager
from oct_vla.tasks.shelf_restock.spec import ObjectVariation, ShelfRegion, ShelfRestockSpec

IDENTITY_POSE = (0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0)


def spec() -> ShelfRestockSpec:
    return ShelfRestockSpec(
        lower_shelf=ShelfRegion("lower_shelf", (0.0, -0.15, 0.74), (0.2, 0.1, 0.01)),
        upper_shelf=ShelfRegion("upper_shelf", (0.0, 0.10, 1.05), (0.2, 0.1, 0.015)),
        object_variation=ObjectVariation(
            (-0.1, 0.1), (-0.2, -0.1), (-0.2, 0.2), ((0.03, 0.05),) * 3
        ),
    )


def frame() -> RGBFrame:
    return RGBFrame(2, 2, b"\x00" * 12)


class StubEstimator:
    """Stands in for the ground-truth estimator: one object, still on the
    lower shelf, so the manager always has a restockable target."""

    def estimate(self, observation):
        return ObjectScene(
            observation.timestamp,
            (ObjectState("a", Pose((0.0, -0.15, 0.74), (0, 0, 0, 1)), (0.04,) * 3, 1.0, 1.0),),
        )


class FakePort:
    """A port whose IK always fails, as it does for an off-distribution pose."""

    dt = 1.0 / 250.0

    def __init__(self, failure: Exception) -> None:
        self._failure = failure
        self.commands: list[tuple] = []
        self.ticks = 0

    def read(self) -> Reading:
        arm = ArmReading(IDENTITY_POSE, 0.04, (0.0,) * 7)
        return Reading(arm, arm, (frame(), frame(), frame()))

    def ik(self, side, pose_wxyz):
        raise self._failure

    def arm_joints(self, side):
        return (0.0,) * 7

    def command(self, *args):
        self.commands.append(args)

    def tick(self):
        self.ticks += 1


def server_with_episode(failure: Exception) -> tuple[ShelfRestockEvalServer, FakePort]:
    port = FakePort(failure)
    server = ShelfRestockEvalServer(lambda task: port, spec=spec())
    server._port = port
    server._episode = _Episode(
        spec=spec(),
        manager=ShelfRestockManager(spec()),
        estimator=StubEstimator(),
        max_steps=600,
    )
    return server, port


def test_unreachable_pose_ends_the_episode_instead_of_raising():
    server, _ = server_with_episode(UnreachablePose("left arm IK failed"))
    header, blobs = server.step([0.0] * 14)
    assert header["done"] is True
    assert header["success"] is False
    assert header["reason"] == "unreachable_pose"
    assert "IK failed" in header["detail"]
    assert blobs, "the terminal observation is still returned"


def test_unreachable_pose_leaves_the_scene_untouched():
    """No arm may be commanded and no tick taken: a half-applied action would
    end the episode in a state no policy output produced."""
    server, port = server_with_episode(UnreachablePose("left arm IK failed"))
    server.step([0.0] * 14)
    assert port.commands == []
    assert port.ticks == 0


def test_unreachable_pose_still_reports_partial_credit():
    server, _ = server_with_episode(UnreachablePose("left arm IK failed"))
    header, _ = server.step([0.0] * 14)
    # The single object is on the lower shelf, so no transfers are complete --
    # but the field must be present, since it is the sweep's partial credit.
    assert header["transfers_completed"] == 0
    assert header["steps"] == 0


def test_a_genuine_port_failure_still_propagates():
    """Only the unreachable-pose case is an outcome. A broken simulator must
    still stop the run rather than being scored as a failed episode."""
    server, _ = server_with_episode(NativePortError("planner process died"))
    try:
        server.step([0.0] * 14)
    except NativePortError as error:
        assert "planner process died" in str(error)
    else:
        raise AssertionError("a genuine NativePortError must not be swallowed")
