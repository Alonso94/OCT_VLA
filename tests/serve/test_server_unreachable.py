"""An unreachable commanded pose must end the episode, not the evaluation.

This is a regression test for a failure that cost a whole evaluation job: the
IK solver raised, the exception crossed the bridge as a protocol error, and the
client died on its first bad action -- discarding every episode scored before
it. Undertrained policies command unreachable poses constantly, so the arms the
sweep most needs to measure were exactly the ones that produced no results.
"""

from __future__ import annotations

import pytest

from oct_vla.core.frames import Pose
from oct_vla.core.objects import ObjectScene, ObjectState
from oct_vla.core.observation import RGBFrame
from oct_vla.robots.robotwin.backend import ArmReading, Reading
from oct_vla.robots.robotwin.native import NativePortError, UnreachablePose
from oct_vla.serve.server import ShelfRestockEvalServer, _decode_gripper_target, _Episode
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


MOVING_ACTION = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04] + [0.0] * 6 + [0.04]


def test_unreachable_pose_does_not_raise_or_end_the_run():
    """The original failure this guards: the IK error crossed the bridge as a
    protocol error and killed the client, discarding every episode already
    scored."""
    server, _ = server_with_episode(UnreachablePose("left arm IK failed"))
    header, blobs = server.step(MOVING_ACTION)
    assert header["done"] is False, "one infeasible command must not end the episode"
    assert blobs, "the observation is still returned"


def test_an_infeasible_command_holds_that_arm_and_is_counted():
    """A real arm ignores a command it cannot execute. Holding lets the policy
    recover, and the count separates 'cannot do the task' from 'asked for
    somewhere the arm cannot go' -- which a single terminal outcome could not."""
    server, port = server_with_episode(UnreachablePose("left arm IK failed"))
    header, _ = server.step(MOVING_ACTION)
    assert header["infeasible_steps"] == 1
    assert "IK failed" in header["detail"]
    # Held at the measured joints rather than driven anywhere.
    assert port.commands, "the arm is still commanded, to hold position"
    assert all(command[1] == (0.0,) * 7 for command in port.commands)


def test_the_episode_still_runs_to_the_step_limit():
    """Terminating on the first infeasible command gave a policy no chance to
    recover; 98.5% of the first sweep's episodes died that way."""
    server, _ = server_with_episode(UnreachablePose("left arm IK failed"))
    for _ in range(3):
        header, _ = server.step(MOVING_ACTION)
        assert header["done"] is False
    assert header["infeasible_steps"] == 3
    assert header["transfers_completed"] == 0


def test_a_genuine_port_failure_still_propagates():
    """Only the unreachable-pose case is an outcome. A broken simulator must
    still stop the run rather than being scored as a failed episode."""
    server, _ = server_with_episode(NativePortError("planner process died"))
    try:
        server.step(MOVING_ACTION)
    except NativePortError as error:
        assert "planner process died" in str(error)
    else:
        raise AssertionError("a genuine NativePortError must not be swallowed")


def test_near_zero_pose_delta_holds_without_calling_ik():
    """Physics noise in an inactive arm must not turn a hold into an IK query."""
    server, port = server_with_episode(AssertionError("IK must not be called"))
    action = [1e-7] * 6 + [0.8] + [-1e-7] * 6 + [0.2]

    header, _ = server.step(action)

    assert header["done"] is False
    assert len(port.commands) == port.ticks * 2
    assert all(command[1] == (0.0,) * 7 for command in port.commands)
    assert all(command[2] == (0.0,) * 7 for command in port.commands)
    assert port.ticks in (16, 17)


def test_measured_gripper_aperture_decodes_to_force_preserving_commands():
    """Contact aperture is an observation, not a weak motor target."""
    assert _decode_gripper_target(0.79) < 1e-6
    assert _decode_gripper_target(0.8333) > 1 - 1e-6
    assert _decode_gripper_target(0.0) == 0.0
    assert _decode_gripper_target(1.0) == 1.0


def test_the_gripper_command_never_lands_between_open_and_closed():
    """An intermediate command is a weak grip -- the bug that made every
    transfer fail. The sharpness is sized to the tightest gap in the recorded
    apertures, so the nearest real sample on either side still saturates."""
    from oct_vla.serve.server import GRIPPER_DECISION_CENTRE as C

    # The closest observed samples either side of the decision point.
    for aperture in (0.821989, 0.823055, 0.8095, 0.8250, 0.833317):
        command = _decode_gripper_target(aperture)
        assert min(command, 1.0 - command) < 1e-6, f"{aperture} decoded to {command}"
    # Only an input at the centre itself is undecided, and no sample sits there.
    assert _decode_gripper_target(C) == pytest.approx(0.5)


def test_the_gripper_decode_is_memoryless():
    """An earlier version returned the previous command inside a dead band,
    making the command a function of history rather than the observation --
    and the dead band was not empty: 0.34% of the corpus fell inside it."""
    import inspect

    from oct_vla.serve.server import _decode_gripper_target as decode

    assert list(inspect.signature(decode).parameters) == ["requested"]
    assert decode(0.82) == decode(0.82)


def test_the_gripper_decode_is_monotonic():
    """A wider aperture can never decode to a tighter grip."""
    values = [0.0, 0.3, 0.65, 0.80, 0.8224, 0.8231, 0.8333, 1.0]
    commands = [_decode_gripper_target(v) for v in values]
    assert commands == sorted(commands)


def arm(position, gripper=0.8):
    from oct_vla.core.state import ArmState
    return ArmState(Pose(position, (1.0, 0.0, 0.0, 0.0)), gripper)


def test_reference_keeps_residual_while_the_arm_is_tracking():
    """The point of integrating the reference: a small tracking shortfall must
    be carried forward, not silently dropped and re-accrued every step."""
    from oct_vla.core.state import EEFState
    from oct_vla.serve.server import _leashed_reference

    commanded = EEFState(arm((0.0, 0.0, 1.000)), arm((0.5, 0.0, 0.9)))
    measured = EEFState(arm((0.0, 0.0, 0.995)), arm((0.5, 0.0, 0.9)))
    out = _leashed_reference(commanded, measured)
    assert out.left.pose.position == commanded.left.pose.position


def test_a_runaway_reference_is_re_anchored_to_the_measured_pose():
    """Integrator windup: the policy emits ~1 mm/step for an arm it is not
    driving, and unbounded integration floated the right arm from z~0.9 to
    z~1.6 -- above the shelves -- ending 8 of 9 probe episodes."""
    from oct_vla.core.state import EEFState
    from oct_vla.serve.server import _leashed_reference

    commanded = EEFState(arm((0.0, 0.0, 0.9)), arm((0.5, 0.0, 1.600)))
    measured = EEFState(arm((0.0, 0.0, 0.9)), arm((0.5, 0.0, 0.900)))
    out = _leashed_reference(commanded, measured)
    assert out.right.pose.position == measured.right.pose.position


def test_each_arm_is_leashed_independently():
    """A driven arm must not lose its residual because the other arm drifted."""
    from oct_vla.core.state import EEFState
    from oct_vla.serve.server import _leashed_reference

    commanded = EEFState(arm((0.0, 0.0, 1.000)), arm((0.5, 0.0, 1.600)))
    measured = EEFState(arm((0.0, 0.0, 0.995)), arm((0.5, 0.0, 0.900)))
    out = _leashed_reference(commanded, measured)
    assert out.left.pose.position == commanded.left.pose.position
    assert out.right.pose.position == measured.right.pose.position


def test_the_first_step_uses_the_measured_pose():
    from oct_vla.core.state import EEFState
    from oct_vla.serve.server import _leashed_reference

    measured = EEFState(arm((0.0, 0.0, 0.9)), arm((0.5, 0.0, 0.9)))
    assert _leashed_reference(None, measured) is measured


# ------------------------------------------------------- joint control space


def joint_server() -> tuple[ShelfRestockEvalServer, FakePort]:
    server, port = server_with_episode(AssertionError("IK must not be called in joint space"))
    server._episode.control_space = "joint"
    return server, port


JOINT_ACTION = [0.1] * 7 + [0.0] + [0.2] * 7 + [1.0]


def test_joint_actions_never_touch_ik():
    """The point of the joint control space: with no solver in the loop, no
    command can be kinematically infeasible. FakePort raises if ik() is called."""
    server, port = joint_server()
    header, _ = server.step(JOINT_ACTION)
    assert header["infeasible_steps"] == 0
    assert port.commands, "the arms are still commanded"


def test_joint_targets_are_passed_through_unchanged():
    """Absolute targets, so there is nothing to integrate and nothing to drift."""
    server, port = joint_server()
    server.step(JOINT_ACTION)
    left = [c for c in port.commands if c[0] == "left"]
    right = [c for c in port.commands if c[0] == "right"]
    assert all(c[1] == (0.1,) * 7 for c in left)
    assert all(c[1] == (0.2,) * 7 for c in right)


def test_joint_gripper_still_decodes_to_a_force_preserving_command():
    """The action carries the next *measured* aperture in either control space,
    and it stalls mid-close while grasping -- so it needs the same decode."""
    server, port = joint_server()
    server.step(JOINT_ACTION)
    assert [c[3] for c in port.commands if c[0] == "left"][0] == 0.0
    assert [c[3] for c in port.commands if c[0] == "right"][0] == 1.0


def test_a_joint_action_of_the_wrong_width_is_refused():
    """A Cartesian policy pointed at a joint-space server would otherwise have
    its 14 numbers silently reinterpreted as joint targets."""
    from oct_vla.serve.server import EvalServerError

    server, _ = joint_server()
    try:
        server.step([0.0] * 14)
    except EvalServerError as error:
        assert "does not match this robot" in str(error) or "joints" in str(error)
    else:
        raise AssertionError("a mismatched action width must be refused")


def test_an_odd_length_joint_action_is_refused():
    from oct_vla.serve.server import EvalServerError

    server, _ = joint_server()
    try:
        server.step([0.0] * 15)
    except EvalServerError as error:
        assert "even length" in str(error)
    else:
        raise AssertionError("an odd-width joint action must be refused")


def test_an_unknown_control_space_is_refused_at_reset():
    from oct_vla.serve.server import CONTROL_SPACES

    assert "joint" in CONTROL_SPACES and "cartesian" in CONTROL_SPACES
