from math import pi

import pytest

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.geometry import exp, rotate
from oct_vla.core.objects import ObjectState
from oct_vla.tasks.shelf_restock.oracle.grasps import (
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


def test_grasp_and_pregrasp_pose_differ_only_by_standoff_along_z():
    (candidate,) = generate_top_down_grasps(
        obj(position=(0.1, 0.2, 0.75), size=(0.04, 0.09, 0.03)),
        gripper_max_width=0.05,
        standoff=0.12,
    )
    assert candidate.grasp_pose.position == pytest.approx((0.1, 0.2, 0.75))
    assert candidate.pregrasp_pose.position == pytest.approx((0.1, 0.2, 0.87))
    assert candidate.grasp_pose.orientation == candidate.pregrasp_pose.orientation
    assert candidate.grasp_pose.frame == WORKCELL_FRAME


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
