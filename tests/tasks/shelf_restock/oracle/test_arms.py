import pytest

from oct_vla.tasks.shelf_restock.oracle.arms import (
    CROSS_BODY_X,
    CROSS_BODY_Y,
    is_cross_body_limited,
    select_arm,
)


@pytest.mark.parametrize(
    "x,expected",
    [(-0.3, "left"), (-0.15, "left"), (-0.001, "left"), (0.001, "right"), (0.15, "right")],
)
def test_select_arm_picks_the_same_side(x, expected):
    assert select_arm((x, -0.1, 0.9)) == expected


def test_select_arm_breaks_centreline_ties_deterministically():
    assert select_arm((0.0, -0.1, 0.9)) == "left"
    assert select_arm((0.0, 0.0, 1.05)) == "left"


def test_select_arm_rejects_malformed_positions():
    with pytest.raises(ValueError):
        select_arm((0.0, 0.0))
    with pytest.raises(ValueError):
        select_arm((float("nan"), 0.0, 0.9))


def test_select_arm_never_produces_a_cross_body_limited_pairing():
    for x in (-0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3):
        for y in (-0.30, -0.20, -0.10, -0.05, 0.0):
            for z in (0.85, 0.95, 1.05):
                position = (x, y, z)
                assert not is_cross_body_limited(select_arm(position), position)


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
# were identical at num_trajopt_seeds 1 and 4. See docs/architecture.md.
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
