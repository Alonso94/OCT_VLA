from math import pi

import pytest

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.geometry import exp, rotate
from oct_vla.core.objects import ObjectState
from oct_vla.tasks.shelf_restock.oracle.grasps import (
    _HAND_REACH_BELOW_TCP,
    GRASP_TCP_OFFSET,
    MAX_GRASP_DEPTH,
    POINTING_DOWN,
    _object_yaw,
    _top_down_orientation,
    generate_top_down_grasps,
)


def obj(position=(0.1, 0.2, 0.75), size=(0.04, 0.05, 0.03), yaw=0.0) -> ObjectState:
    return ObjectState(
        "target", Pose(position, exp((0.0, 0.0, yaw))), size, visibility=1.0, confidence=1.0
    )


def test_pointing_down_approach_axis_is_straight_down():
    assert rotate(POINTING_DOWN, (1, 0, 0)) == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)


@pytest.mark.parametrize("yaw", [0.0, 0.3, pi / 2, -1.0, pi])
def test_top_down_orientation_keeps_approach_axis_down_for_any_yaw(yaw):
    orientation = _top_down_orientation(yaw)
    assert rotate(orientation, (1, 0, 0)) == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)


def test_top_down_orientation_spins_closing_axis_by_yaw_in_the_horizontal_plane():
    base = rotate(_top_down_orientation(0.0), (0, 1, 0))
    spun = rotate(_top_down_orientation(pi / 2), (0, 1, 0))
    # A 90 degree yaw about z should rotate the horizontal closing-axis
    # direction by 90 degrees too; z-component (still zero) is unaffected.
    assert spun == pytest.approx((-base[1], base[0], base[2]), abs=1e-9)


def test_object_yaw_recovers_a_pure_z_rotation():
    assert _object_yaw(obj(yaw=0.7)) == pytest.approx(0.7)
    assert _object_yaw(obj(yaw=0.0)) == pytest.approx(0.0)


def test_generate_top_down_grasps_returns_two_candidates_when_both_fit():
    candidates = generate_top_down_grasps(
        obj(size=(0.04, 0.05, 0.03)), gripper_max_width=0.08, standoff=0.1
    )
    assert len(candidates) == 2
    assert {c.closing_width for c in candidates} == {0.04, 0.05}


def test_generate_top_down_grasps_filters_out_the_dimension_that_does_not_fit():
    candidates = generate_top_down_grasps(
        obj(size=(0.10, 0.05, 0.03)), gripper_max_width=0.08, standoff=0.1
    )
    assert len(candidates) == 1
    assert candidates[0].closing_width == pytest.approx(0.05)
    assert candidates[0].wrist_yaw == pytest.approx(pi / 2)


def test_generate_top_down_grasps_returns_nothing_when_neither_dimension_fits():
    candidates = generate_top_down_grasps(
        obj(size=(0.10, 0.09, 0.03)), gripper_max_width=0.08, standoff=0.1
    )
    assert candidates == ()


def test_grasp_pose_is_offset_back_along_the_approach_axis_not_at_the_centre():
    """The commanded pose is not the point between the fingers: putting it at
    the object's centre buries the fingers GRASP_TCP_OFFSET below it."""
    height, depth = 0.03, 0.015
    (candidate,) = generate_top_down_grasps(
        obj(position=(0.1, 0.2, 0.75), size=(0.04, 0.09, height)),
        gripper_max_width=0.05,
        standoff=0.12,
        grasp_depth=depth,
    )
    pads_z = 0.75 + height / 2 - depth
    assert candidate.grasp_pose.position == pytest.approx((0.1, 0.2, pads_z + GRASP_TCP_OFFSET))
    assert candidate.pregrasp_pose.position == pytest.approx(
        (0.1, 0.2, pads_z + GRASP_TCP_OFFSET + 0.12)
    )
    assert candidate.grasp_pose.orientation == candidate.pregrasp_pose.orientation
    assert candidate.grasp_pose.frame == WORKCELL_FRAME


def test_finger_pads_land_just_below_the_object_top_face():
    """Reconstruct where the pads end up: the commanded pose advanced by
    GRASP_TCP_OFFSET along its own approach axis must sit grasp_depth below
    the object's top, which is what keeps the hand body clear of a tall
    object."""
    height, depth = 0.05, 0.015
    target = obj(position=(0.05, -0.25, 0.78), size=(0.04, 0.04, height), yaw=0.6)
    for candidate in generate_top_down_grasps(
        target, gripper_max_width=0.08, standoff=0.1, grasp_depth=depth
    ):
        approach = rotate(candidate.grasp_pose.orientation, (1, 0, 0))
        pads = tuple(
            p + GRASP_TCP_OFFSET * a
            for p, a in zip(candidate.grasp_pose.position, approach, strict=True)
        )
        top_z = target.pose.position[2] + height / 2
        assert pads[:2] == pytest.approx(target.pose.position[:2], abs=1e-9)
        assert pads[2] == pytest.approx(top_z - depth, abs=1e-9)


def test_hand_body_stays_clear_of_the_object_top_for_every_allowed_height():
    """The failure this guards: grasping a tall object at its centre put
    panda_hand inside it. Checked across the spec's full height range."""
    for height in (0.03, 0.05, 0.08, 0.12):
        target = obj(position=(0.0, -0.25, 0.80), size=(0.04, 0.04, height))
        (candidate, _) = generate_top_down_grasps(target, gripper_max_width=0.08, standoff=0.1)
        hand_lowest_z = candidate.grasp_pose.position[2] - _HAND_REACH_BELOW_TCP
        top_z = target.pose.position[2] + height / 2
        assert hand_lowest_z > top_z, f"hand enters a {height}m object"


def test_rejects_a_grasp_depth_that_would_bury_the_hand():
    with pytest.raises(ValueError, match="grasp_depth"):
        generate_top_down_grasps(
            obj(), gripper_max_width=0.08, standoff=0.1, grasp_depth=MAX_GRASP_DEPTH
        )
    with pytest.raises(ValueError, match="grasp_depth"):
        generate_top_down_grasps(obj(), gripper_max_width=0.08, standoff=0.1, grasp_depth=0.0)


def test_pregrasp_is_further_back_along_approach_than_grasp():
    (candidate, _) = generate_top_down_grasps(
        obj(size=(0.04, 0.04, 0.03)), gripper_max_width=0.08, standoff=0.09
    )
    # Top-down: "back along approach" is straight up.
    assert candidate.pregrasp_pose.position[2] - candidate.grasp_pose.position[2] == pytest.approx(
        0.09
    )


def test_wrist_yaw_accounts_for_the_object_s_own_yaw():
    candidates = generate_top_down_grasps(
        obj(size=(0.04, 0.04, 0.03), yaw=0.5), gripper_max_width=0.08, standoff=0.1
    )
    yaws = sorted(c.wrist_yaw for c in candidates)
    assert yaws == pytest.approx([0.5, 0.5 + pi / 2])


def test_candidate_orientation_still_points_straight_down_for_a_yawed_object():
    candidates = generate_top_down_grasps(
        obj(size=(0.04, 0.04, 0.03), yaw=1.1), gripper_max_width=0.08, standoff=0.1
    )
    for candidate in candidates:
        approach = rotate(candidate.grasp_pose.orientation, (1, 0, 0))
        assert approach == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)
