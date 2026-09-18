import json
import random
from dataclasses import FrozenInstanceError, asdict
from math import hypot, pi, sqrt

import pytest

from oct_vla.core.action import ACTION_DIM, Action, ArmAction, action_between, apply_action
from oct_vla.core.frames import Pose, Transform
from oct_vla.core.geometry import exp, log, rotate, unit_quaternion
from oct_vla.core.state import ArmState, EEFState


def state():
    return EEFState(
        ArmState(Pose((0.1, 0.2, 0.3), exp((pi / 2, 0, 0))), 0.3),
        ArmState(Pose((-0.1, 0, 0.5), exp((0, pi / 4, 0))), 0.8),
    )


def assert_pose(actual, expected):
    assert actual.frame == expected.frame
    assert actual.position == pytest.approx(expected.position, abs=1e-12)
    # Compare physical rotations without assuming a quaternion sign near pi.
    for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        assert rotate(actual.orientation, axis) == pytest.approx(
            rotate(expected.orientation, axis), abs=1e-12
        )


@pytest.mark.parametrize("angle", [0, 1e-14, 1e-8, 0.3, pi - 1e-10, pi, pi + 1e-10])
def test_exp_log_short_arc_and_quaternion_sign(angle):
    q = exp((0, angle, 0))
    negated = tuple(-v for v in q)
    assert log(q) == pytest.approx(log(negated), abs=1e-15)
    assert hypot(*log(q)) <= pi + 1e-15
    assert rotate(exp(log(q)), (1, 0, 0)) == pytest.approx(rotate(q, (1, 0, 0)), abs=1e-14)
    if angle <= pi:
        assert log(q)[1] == pytest.approx(angle, rel=1e-12, abs=1e-16)


def test_known_rotation_and_exact_pi_tie():
    assert rotate((0, 0, sqrt(0.5), sqrt(0.5)), (1, 0, 0)) == pytest.approx((0, 1, 0))
    assert log((-1, 0, 0, 0)) == pytest.approx((pi, 0, 0))
    assert unit_quaternion((0, 0, 0, 1 + 1e-8)) == (0, 0, 0, 1)


@pytest.mark.parametrize("q", [(0, 0, 0, 0), (0, 0, 0, 2), (0, 0, float("nan"), 1), (0, 1)])
def test_invalid_quaternions_rejected(q):
    with pytest.raises(ValueError):
        Pose((0, 0, 0), q)


@pytest.mark.parametrize(
    "values", [[0] * 13, [0] * 15, [[0] * 14], [float("nan")] * 14, [float("inf")] * 14]
)
def test_action_rejects_wrong_shape_and_nonfinite(values):
    with pytest.raises(ValueError):
        Action.from_vector(values)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_gripper_range(value):
    with pytest.raises(ValueError):
        ArmAction((0, 0, 0), (0, 0, 0), value)
    with pytest.raises(ValueError):
        ArmState(state().left.pose, value)


@pytest.mark.parametrize(
    ("emitted", "commanded"), [(-0.4, 0.0), (1.6, 1.0), (0.0, 0.0), (1.0, 1.0), (0.3, 0.3)]
)
def test_from_vector_saturates_gripper_but_arm_action_stays_strict(emitted, commanded):
    """A policy's overshoot becomes a saturated command, not a dead episode.

    Under `binary_command` the gripper targets are exactly {0, 1}, so an L1
    head overshoots both ends routinely. `from_vector` is the boundary where a
    regressed number becomes a command and is the only place that clamps;
    ArmAction itself still rejects, so an out-of-range value built internally
    is still a bug rather than a silent clip.
    """
    action = Action.from_vector([0.0] * 6 + [emitted] + [0.0] * 6 + [emitted])
    assert action.left.gripper == commanded
    assert action.right.gripper == commanded


def test_order_immutability_and_json_round_trip():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.8]
    action = Action.from_vector(values)
    assert len(action.to_vector()) == ACTION_DIM
    assert action.left.translation == (0.1, 0.2, 0.3)
    assert action.right.gripper == 0.8
    assert Action.from_vector(json.loads(json.dumps(action.to_vector()))) == action
    values[0] = 999
    assert action.left.translation[0] == 0.1
    with pytest.raises(FrozenInstanceError):
        action.left.gripper = 0
    pose = Pose(**json.loads(json.dumps(asdict(state().left.pose))))
    assert_pose(pose, state().left.pose)


def test_hold_preserves_grippers_and_pose():
    current = state()
    result = apply_action(current, Action.hold(current))
    assert_pose(result.left.pose, current.left.pose)
    assert_pose(result.right.pose, current.right.pose)
    assert (result.left.gripper, result.right.gripper) == (0.3, 0.8)


@pytest.mark.parametrize("translation", [(0.02, 0, 0), (0, -0.03, 0), (0, 0, 0)])
def test_translation_and_gripper_only(translation):
    current = state()
    delta = ArmAction(translation, (0, 0, 0), 0.1)
    result = apply_action(current, Action(Action.hold(current).left, delta))
    assert_pose(result.left.pose, current.left.pose)
    assert result.left.gripper == current.left.gripper
    assert result.right.pose.position == pytest.approx(
        tuple(a + b for a, b in zip(current.right.pose.position, translation, strict=True))
    )
    assert result.right.pose.orientation == pytest.approx(current.right.pose.orientation)
    assert result.right.gripper == 0.1


@pytest.mark.parametrize("position", [(0, 0), (0, 0, float("inf")), ((0, 0, 0),)])
def test_invalid_pose_positions(position):
    with pytest.raises(ValueError):
        Pose(position, (0, 0, 0, 1))


def test_empty_frame_rejected():
    with pytest.raises(ValueError):
        Pose((0, 0, 0), (0, 0, 0, 1), " ")


def test_spatial_left_multiplication_and_independent_arm():
    current = state()
    action = Action(ArmAction((0.02, 0, 0), (0, 0, pi / 2), 1), Action.hold(current).right)
    result = apply_action(current, action)
    assert result.left.pose.position == pytest.approx((0.12, 0.2, 0.3))
    # Rx(pi/2) maps y to z; a subsequent workcell Rz leaves z unchanged.
    # Reversing multiplication instead maps y to -x.
    assert rotate(result.left.pose.orientation, (0, 1, 0)) == pytest.approx((0, 0, 1))
    assert result.left.gripper == 1
    assert_pose(result.right.pose, current.right.pose)
    assert result.right.gripper == current.right.gripper


def test_random_pose_delta_round_trips():
    rng = random.Random(42)

    def random_state():
        def arm():
            return ArmState(
                Pose(
                    tuple(rng.uniform(-1, 1) for _ in range(3)),
                    exp(tuple(rng.uniform(-pi, pi) for _ in range(3))),
                ),
                rng.random(),
            )

        return EEFState(arm(), arm())

    for _ in range(100):
        current, following = random_state(), random_state()
        result = apply_action(current, action_between(current, following))
        for actual, expected in ((result.left, following.left), (result.right, following.right)):
            assert_pose(actual.pose, expected.pose)
            assert actual.gripper == expected.gripper


def test_transform_known_point_vector_inverse_and_chain():
    work_from_world = Transform("world", "workcell", (1, 2, 3), exp((0, 0, pi / 2)))
    assert work_from_world.apply_point((1, 0, 0)) == pytest.approx((1, 3, 3))
    assert work_from_world.apply_vector((1, 0, 0)) == pytest.approx((0, 1, 0))
    pose = Pose((0.1, 0.2, 0.3), exp((0.2, 0.3, 0.4)), "world")
    assert_pose(work_from_world.inverse().apply_pose(work_from_world.apply_pose(pose)), pose)
    world_from_base = Transform("base", "world", (-1, 0.4, 0.2), exp((0.4, 0, 0)))
    base_pose = Pose(pose.position, pose.orientation, "base")
    assert_pose(
        work_from_world.compose(world_from_base).apply_pose(base_pose),
        work_from_world.apply_pose(world_from_base.apply_pose(base_pose)),
    )
    with pytest.raises(ValueError):
        world_from_base.compose(work_from_world)
    with pytest.raises(ValueError):
        work_from_world.apply_pose(base_pose)
    with pytest.raises(ValueError):
        ArmState(base_pose, 0.5)


def test_delta_frame_conversion_commutes_with_pose_composition():
    transform = Transform("base", "workcell", (4, -2, 1), exp((0.3, -0.4, 0.2)))
    start = Pose((0.1, 0.2, 0.3), exp((0.8, 0.1, -0.3)), "base")
    end = Pose((-0.2, 0.5, 0.4), exp((-0.2, 0.5, 0.1)), "base")
    # Compute the same geometric delta in base and workcell coordinates.
    base_states = [
        EEFState(*(ArmState(Pose(p.position, p.orientation), 0.5) for _ in range(2)))
        for p in (start, end)
    ]
    delta = action_between(*base_states).left
    work_start, work_end = [transform.apply_pose(p) for p in (start, end)]
    current = EEFState(ArmState(work_start, 0.5), ArmState(work_start, 0.5))
    converted = ArmAction(
        transform.apply_vector(delta.translation), transform.apply_vector(delta.rotation), 0.5
    )
    result = apply_action(current, Action(converted, converted))
    assert_pose(result.left.pose, work_end)
