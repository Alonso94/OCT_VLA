from pathlib import Path

import pytest

from oct_vla.robots.robotwin.native import NativePortError, RoboTwinNativePort, _load_task_class


class DummyExternalTask:
    pass


def _plan_result(status: str, value: float = 0.0) -> dict:
    """A minimal cuRobo-shaped plan_path result; `value` marks which attempt
    produced it so tests can tell attempts apart by their returned Trajectory."""
    return {"status": status, "position": [[value] * 7], "velocity": [[value] * 7]}


class FakeRobot:
    """Stands in for `task.robot`: records every plan_path call (side, args,
    kwargs) and returns queued results in order, one per call."""

    def __init__(self, left_results=(), right_results=()) -> None:
        self.calls: list[tuple[str, list, dict]] = []
        self._left = list(left_results)
        self._right = list(right_results)

    def _pop(self, side: str, pose, kwargs) -> dict:
        self.calls.append((side, list(pose), dict(kwargs)))
        queue = self._left if side == "left" else self._right
        return queue.pop(0)

    def left_plan_path(self, pose, **kwargs) -> dict:
        return self._pop("left", pose, kwargs)

    def right_plan_path(self, pose, **kwargs) -> dict:
        return self._pop("right", pose, kwargs)


class FakeTask:
    def __init__(self, robot: FakeRobot) -> None:
        self.robot = robot
        self.refresh_calls: list[str | None] = []

    def refresh_planning_world(self, exclude: str | None = None) -> None:
        self.refresh_calls.append(exclude)


def _make_port(robot: FakeRobot, *, root: Path, plan_attempts: int = 4) -> RoboTwinNativePort:
    """Construct without reset(): set the private `_task` directly, the
    established pattern in this file for exercising post-reset behaviour
    without a real RoboTwin checkout."""
    port = RoboTwinNativePort(root, task_name="fake", plan_attempts=plan_attempts)
    port._task = FakeTask(robot)
    return port


POSE = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_plan_succeeding_first_try_calls_planner_once(tmp_path):
    robot = FakeRobot(left_results=[_plan_result("Success", 1.0)])
    port = _make_port(robot, root=tmp_path)

    trajectory = port.plan("left", POSE)

    assert len(robot.calls) == 1
    assert trajectory.q == ((1.0,) * 7,)


def test_plan_retries_after_failures_and_returns_the_successful_result(tmp_path):
    robot = FakeRobot(
        left_results=[_plan_result("Fail"), _plan_result("Fail"), _plan_result("Success", 3.0)]
    )
    port = _make_port(robot, root=tmp_path, plan_attempts=4)

    trajectory = port.plan("left", POSE)

    assert len(robot.calls) == 3
    assert trajectory.q == ((3.0,) * 7,)
    assert trajectory.qdot == ((3.0,) * 7,)


def test_plan_exhausting_all_attempts_raises_with_attempt_count_and_status(tmp_path):
    robot = FakeRobot(left_results=[_plan_result("Fail")] * 4)
    port = _make_port(robot, root=tmp_path, plan_attempts=4)

    with pytest.raises(NativePortError, match="4 attempts.*'Fail'"):
        port.plan("left", POSE)

    assert len(robot.calls) == 4


def test_plan_attempts_of_one_raises_after_a_single_call(tmp_path):
    robot = FakeRobot(left_results=[_plan_result("Fail")])
    port = _make_port(robot, root=tmp_path, plan_attempts=1)

    with pytest.raises(NativePortError, match="1 attempts"):
        port.plan("left", POSE)

    assert len(robot.calls) == 1


def test_plan_attempts_of_six_is_honoured(tmp_path):
    robot = FakeRobot(left_results=[_plan_result("Fail")] * 6)
    port = _make_port(robot, root=tmp_path, plan_attempts=6)

    with pytest.raises(NativePortError, match="6 attempts"):
        port.plan("left", POSE)

    assert len(robot.calls) == 6


@pytest.mark.parametrize("attempts", [0, -1])
def test_constructor_rejects_plan_attempts_below_one(tmp_path, attempts):
    with pytest.raises(ValueError):
        RoboTwinNativePort(tmp_path, task_name="fake", plan_attempts=attempts)


def test_plan_forwards_constraint_on_every_retry_attempt(tmp_path):
    constraint = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    robot = FakeRobot(
        left_results=[_plan_result("Fail"), _plan_result("Fail"), _plan_result("Success")]
    )
    port = _make_port(robot, root=tmp_path, plan_attempts=4)

    port.plan("left", POSE, constraint=constraint)

    assert len(robot.calls) == 3
    for _, _, kwargs in robot.calls:
        assert kwargs == {"constraint_pose": list(constraint)}


def test_plan_refreshes_planning_world_once_per_call_not_per_attempt(tmp_path):
    """Pins current behaviour: refresh happens before the retry loop starts,
    so a stale-world refresh is not repeated across attempts within one
    plan() call. Not a statement that this is ideal -- just what it does."""
    robot = FakeRobot(
        left_results=[_plan_result("Fail"), _plan_result("Fail"), _plan_result("Success")]
    )
    port = _make_port(robot, root=tmp_path, plan_attempts=4)

    port.plan("left", POSE)

    assert port._task.refresh_calls == [None]


def test_load_task_class_resolves_external_entrypoint_bypassing_envs():
    loaded = _load_task_class(f"{__name__}:DummyExternalTask")
    assert loaded is DummyExternalTask


def test_load_task_class_reports_missing_external_module():
    with pytest.raises(NativePortError):
        _load_task_class("oct_vla.does_not_exist:Whatever")


def test_load_task_class_reports_missing_external_class():
    with pytest.raises(NativePortError):
        _load_task_class(f"{__name__}:DoesNotExist")


def test_load_task_class_reports_missing_native_task():
    with pytest.raises(NativePortError):
        _load_task_class("no_such_robotwin_task")
