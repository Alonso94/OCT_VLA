import pytest

from oct_vla.core.frames import Pose
from oct_vla.core.objects import ObjectState
from oct_vla.tasks.shelf_restock.oracle.arms import (
    CROSS_BODY_X,
    CROSS_BODY_Y,
    arm_for,
    is_cross_body_limited,
)
from oct_vla.tasks.shelf_restock.oracle.placement import plan_placement
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC


def test_the_left_arm_moves_objects_and_the_right_arm_only_compacts():
    assert arm_for("grasp") == "left"
    assert arm_for("place") == "left"
    assert arm_for("compact") == "right"


def test_arm_for_rejects_an_unknown_role():
    with pytest.raises(ValueError, match="role must be one of"):
        arm_for("shove")


def test_the_grasping_arm_reaches_the_whole_lower_shelf():
    """Roles are fixed, so the left arm must cover every x an object can spawn
    at; lower-shelf y sits below CROSS_BODY_Y, which is what makes that hold."""
    low_x, high_x = DEFAULT_SPEC.object_variation.position_x_range
    low_y, high_y = DEFAULT_SPEC.object_variation.position_y_range
    for x in (low_x, 0.0, high_x):
        for y in (low_y, high_y):
            assert not is_cross_body_limited("left", (x, y, 0.78))


def test_placing_along_the_upper_shelf_is_no_longer_cross_body_limited():
    """The deck sat at y=-0.02 -- inside the cross-body band -- until the spec
    moved it to y=-0.06, specifically so left-arm placements land below
    CROSS_BODY_Y and stop being flagged, which is what lets the right arm
    reach what the left arm placed for compaction."""
    shelf_y = DEFAULT_SPEC.upper_shelf.center_xyz[1]
    assert shelf_y < CROSS_BODY_Y
    assert not is_cross_body_limited("left", (CROSS_BODY_X, shelf_y, 0.97))


def test_the_first_placement_is_within_the_placing_arm_reach():
    obj = ObjectState("t", Pose((0.0, -0.25, 0.78), (0, 0, 0, 1)), (0.04, 0.04, 0.05), 1.0, 1.0)
    placement = plan_placement(DEFAULT_SPEC, obj, 0.0, None, standoff=0.1)
    assert not is_cross_body_limited("left", placement.object_pose.position)


@pytest.mark.parametrize("z", [0.85, 0.95, 1.05])
@pytest.mark.parametrize("y", [CROSS_BODY_Y, 0.0])
def test_measured_cross_body_failures_are_flagged(y, z):
    # Exactly the 12 observed failures: left at x=+0.15, right at x=-0.15.
    assert is_cross_body_limited("left", (CROSS_BODY_X, y, z))
    assert is_cross_body_limited("right", (-CROSS_BODY_X, y, z))


@pytest.mark.parametrize("y", [-0.20, -0.15, -0.10])
def test_same_x_is_not_flagged_at_the_y_values_that_planned_successfully(y):
    assert not is_cross_body_limited("left", (CROSS_BODY_X, y, 0.95))
    assert not is_cross_body_limited("right", (-CROSS_BODY_X, y, 0.95))


def test_own_side_and_centreline_are_never_flagged():
    for y in (-0.20, -0.05, 0.0):
        assert not is_cross_body_limited("left", (-0.15, y, 0.95))
        assert not is_cross_body_limited("left", (0.0, y, 0.95))
        assert not is_cross_body_limited("right", (0.15, y, 0.95))
        assert not is_cross_body_limited("right", (0.0, y, 0.95))


def test_is_cross_body_limited_rejects_an_invalid_side():
    with pytest.raises(ValueError):
        is_cross_body_limited("middle", (0.0, 0.0, 0.9))


# The exact grid queried live against cuRobo (both arms; nothing executed, so
# every query started from the same joint configuration). Recorded failures
# were identical at num_trajopt_seeds 1 and 4. See docs/architecture.md (at tag stageA-2026-09-23).
OBSERVED_FAILURES = {
    ("left", 0.15, -0.05),
    ("left", 0.15, 0.0),
    ("right", -0.15, -0.05),
    ("right", -0.15, 0.0),
}


def test_agrees_with_every_observed_planner_result_on_the_measured_grid():
    checked = 0
    for side in ("left", "right"):
        for x in (-0.15, 0.0, 0.15):
            for y in (-0.20, -0.15, -0.10, -0.05, 0.0):
                for z in (0.85, 0.95, 1.05):
                    observed_fail = (side, x, y) in OBSERVED_FAILURES
                    assert is_cross_body_limited(side, (x, y, z)) is observed_fail, (
                        f"{side} at ({x}, {y}, {z})"
                    )
                    checked += 1
    assert checked == 90
    assert sum(1 for s, x, y in OBSERVED_FAILURES for _ in range(3)) == 12
