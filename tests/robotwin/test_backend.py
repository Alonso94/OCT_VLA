"""RoboTwinBackend contract tests against a fake NativePort.

These exercise the canonical wrapper's own logic (frame conversion, dual-plan-
before-commit, safety bounds, hold, stop/health latching) deterministically,
without SAPIEN/cuRobo. They do not substitute for the live actuator check
against the real simulator described in docs/setup.md and architecture.md's
acceptance gate 4 ("+x only, -y only, rotation only, gripper only, inactive
arm must have predictable measured effects").
"""

from math import pi

import pytest

from oct_vla.core.action import Action, ArmAction
from oct_vla.core.frames import WORKCELL_FRAME, Transform
from oct_vla.core.geometry import exp, inverse, multiply, rotate
from oct_vla.core.observation import RGBFrame
from oct_vla.robots.robotwin.backend import (
    ArmReading,
    BackendError,
    Reading,
    RoboTwinBackend,
    Trajectory,
)

# A nontrivial transform (translation + 90 deg yaw) so tests cannot pass by
# accident on an identity mapping between world and workcell frames.
WORLD_TO_WORKCELL = Transform("world", WORKCELL_FRAME, (0.0, 0.0, 0.74), exp((0.0, 0.0, pi / 2)))


def _frame() -> RGBFrame:
    return RGBFrame(1, 1, bytes(3))


class FakePort:
    """Treats a 7-vector 'q' as a world-frame pose, not a joint angle.

    plan() returns a single-row trajectory equal to the requested pose, and
    command() applies it directly; this makes the resulting measured pose an
    exact, checkable function of the requested target, isolating backend.py's
    own transform/composition logic from real robot kinematics.
    """

    dt = 1.0 / 250.0

    def __init__(self) -> None:
        self.reset_calls: list[int] = []
        self.plan_calls: list[tuple[str, tuple[float, ...]]] = []
        self.command_calls: list[tuple[str, tuple[float, ...], tuple[float, ...], float]] = []
        self.tick_count = 0
        self.held = False
        self.closed = False
        self._pose = {
            "left": (0.4, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0),
            "right": (0.4, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0),
        }
        self._gripper = {"left": 0.5, "right": 0.5}
        self.fail_plan_for: set[str] = set()

    def reset(self, seed: int) -> None:
        self.reset_calls.append(seed)

    def read(self) -> Reading:
        arms = {
            side: ArmReading(self._pose[side], self._gripper[side], self._pose[side])
            for side in ("left", "right")
        }
        return Reading(arms["left"], arms["right"], (_frame(), _frame(), _frame()))

    def plan(self, side: str, pose_wxyz: tuple[float, ...]) -> Trajectory:
        self.plan_calls.append((side, pose_wxyz))
        if side in self.fail_plan_for:
            raise RuntimeError(f"{side} plan failed")
        return Trajectory((tuple(pose_wxyz),), ((0.0,) * 7,))

    def command(self, side: str, q, qdot, gripper: float) -> None:
        self.command_calls.append((side, tuple(q), tuple(qdot), gripper))
        self._pose[side] = tuple(q)
        self._gripper[side] = gripper

    def tick(self) -> None:
        self.tick_count += 1

    def hold(self) -> None:
        self.held = True

    def close(self) -> None:
        self.closed = True


def make_backend(port: FakePort | None = None, **kwargs) -> tuple[RoboTwinBackend, FakePort]:
    port = port or FakePort()
    return RoboTwinBackend(port, WORLD_TO_WORKCELL, **kwargs), port


def hold_arm(gripper: float) -> ArmAction:
    return ArmAction((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), gripper)


def test_constructor_rejects_mismatched_transform_frames():
    with pytest.raises(ValueError):
        RoboTwinBackend(FakePort(), Transform("world", "not-workcell", (0, 0, 0), (0, 0, 0, 1)))


@pytest.mark.parametrize("field", ["max_translation", "max_rotation"])
def test_constructor_rejects_nonpositive_displacement_limits(field):
    with pytest.raises(ValueError):
        RoboTwinBackend(FakePort(), WORLD_TO_WORKCELL, **{field: 0.0})


def test_constructor_rejects_settle_steps_not_below_max_steps():
    with pytest.raises(ValueError):
        RoboTwinBackend(FakePort(), WORLD_TO_WORKCELL, max_steps=10, settle_steps=10)


def test_reset_rejects_invalid_seed():
    backend, _ = make_backend()
    with pytest.raises(ValueError):
        backend.reset(seed=-1)


def test_reset_transforms_world_pose_into_workcell_frame():
    backend, port = make_backend()
    obs = backend.reset(seed=3)
    assert port.reset_calls == [3]
    assert obs.eef.left.pose.frame == WORKCELL_FRAME
    # World (0.4, 0.2, 0.3) rotated +90deg about Z, then offset by (0, 0, 0.74).
    assert obs.eef.left.pose.position == pytest.approx((-0.2, 0.4, 1.04), abs=1e-9)


@pytest.mark.parametrize(
    "delta",
    [(0.02, 0.0, 0.0), (0.0, -0.02, 0.0), (0.0, 0.0, 0.01)],
    ids=["+x only", "-y only", "+z only"],
)
def test_translation_only_action_moves_left_arm_by_exact_workcell_delta(delta):
    backend, port = make_backend()
    before = backend.reset(seed=0)
    action = Action(
        ArmAction(delta, (0.0, 0.0, 0.0), before.eef.left.gripper),
        hold_arm(before.eef.right.gripper),
    )

    result = backend.step(action)

    moved = tuple(
        after - prior
        for after, prior in zip(
            result.observation.eef.left.pose.position, before.eef.left.pose.position, strict=True
        )
    )
    assert moved == pytest.approx(delta, abs=1e-9)
    assert result.observation.eef.right.pose.position == pytest.approx(
        before.eef.right.pose.position, abs=1e-9
    )
    assert port.tick_count > 0


def test_rotation_only_action_composes_on_the_left_in_workcell_frame():
    backend, _ = make_backend()
    before = backend.reset(seed=0)
    rotation = (0.0, 0.0, 0.1)
    action = Action(
        ArmAction((0.0, 0.0, 0.0), rotation, before.eef.left.gripper),
        hold_arm(before.eef.right.gripper),
    )

    result = backend.step(action)

    relative = multiply(
        result.observation.eef.left.pose.orientation, inverse(before.eef.left.pose.orientation)
    )
    expected = exp(rotation)
    for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        assert rotate(relative, axis) == pytest.approx(rotate(expected, axis), abs=1e-9)


def test_gripper_only_action_reaches_exact_target_without_moving():
    backend, _ = make_backend()
    before = backend.reset(seed=0)
    action = Action(
        ArmAction((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0),
        ArmAction((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0),
    )

    result = backend.step(action)

    assert result.observation.eef.left.gripper == pytest.approx(1.0)
    assert result.observation.eef.right.gripper == pytest.approx(0.0)
    assert result.observation.eef.left.pose.position == pytest.approx(
        before.eef.left.pose.position, abs=1e-9
    )
    assert result.observation.eef.right.pose.position == pytest.approx(
        before.eef.right.pose.position, abs=1e-9
    )


def test_inactive_arm_holds_via_measured_state_without_a_plan_call():
    backend, port = make_backend()
    before = backend.reset(seed=0)
    action = Action(
        ArmAction((0.02, 0.0, 0.0), (0.0, 0.0, 0.0), before.eef.left.gripper),
        hold_arm(before.eef.right.gripper),
    )

    result = backend.step(action)

    assert [side for side, _ in port.plan_calls] == ["left"]
    assert result.observation.eef.right.pose.position == pytest.approx(
        before.eef.right.pose.position, abs=1e-9
    )
    assert result.observation.eef.right.pose.orientation == pytest.approx(
        before.eef.right.pose.orientation, abs=1e-9
    )


@pytest.mark.parametrize(
    "bad",
    [
        ArmAction((0.2, 0.0, 0.0), (0.0, 0.0, 0.0), 0.5),
        ArmAction((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.5),
    ],
)
def test_action_exceeding_bounds_raises_and_sends_no_command(bad):
    backend, port = make_backend(max_translation=0.05, max_rotation=0.25)
    backend.reset(seed=0)
    action = Action(bad, hold_arm(0.5))

    with pytest.raises(ValueError):
        backend.step(action)

    assert port.plan_calls == []
    assert port.command_calls == []
    assert backend.health().ready is True


def test_plan_failure_stops_backend_until_next_reset():
    port = FakePort()
    port.fail_plan_for.add("left")
    backend, _ = make_backend(port)
    before = backend.reset(seed=0)
    action = Action(
        ArmAction((0.01, 0.0, 0.0), (0.0, 0.0, 0.0), before.eef.left.gripper),
        hold_arm(before.eef.right.gripper),
    )

    with pytest.raises(RuntimeError, match="left plan failed"):
        backend.step(action)

    assert backend.health().ready is False
    assert port.held is True
    with pytest.raises(BackendError):
        backend.step(action)

    backend.reset(seed=1)
    assert backend.health().ready is True


def test_execution_budget_exceeded_stops_backend():
    backend, _ = make_backend(max_steps=10, settle_steps=1)
    before = backend.reset(seed=0)
    action = Action(
        ArmAction((0.01, 0.0, 0.0), (0.0, 0.0, 0.0), before.eef.left.gripper),
        hold_arm(before.eef.right.gripper),
    )

    with pytest.raises(BackendError, match="EXECUTION_BUDGET_EXCEEDED"):
        backend.step(action)
    assert backend.health().ready is False


def test_stop_holds_port_and_latches_until_reset():
    backend, port = make_backend()
    backend.reset(seed=0)
    backend.stop()

    assert port.held is True
    assert backend.health().ready is False
    with pytest.raises(BackendError):
        backend.step(Action(hold_arm(0.5), hold_arm(0.5)))


def test_close_marks_not_ready_and_closes_port():
    backend, port = make_backend()
    backend.reset(seed=0)
    backend.close()

    assert port.closed is True
    assert backend.health().ready is False
