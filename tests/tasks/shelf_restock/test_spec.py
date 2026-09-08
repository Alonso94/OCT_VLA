import random

import pytest

from oct_vla.tasks.shelf_restock.spec import (
    DEFAULT_SPEC,
    ObjectVariation,
    ShelfRegion,
    ShelfRestockSpec,
)


def region() -> ShelfRegion:
    return ShelfRegion("lower_shelf", (0.0, -0.15, 0.74), (0.2, 0.1, 0.01))


def variation() -> ObjectVariation:
    return ObjectVariation(
        position_x_range=(-0.1, 0.1),
        position_y_range=(-0.2, -0.1),
        yaw_range=(-0.2, 0.2),
        size_xyz_range=((0.03, 0.05), (0.03, 0.05), (0.03, 0.05)),
    )


def test_shelf_region_top_z_is_center_plus_half_extent():
    assert region().top_z == pytest.approx(0.75)


@pytest.mark.parametrize(
    "position,expected",
    [
        ((0.0, -0.15, 0.75), True),  # exactly at the deck surface (top_z)
        ((0.19, -0.15, 0.80), True),  # in-footprint, resting above the deck
        ((0.0, -0.15, 0.899), True),  # just inside the occupancy volume
        ((0.21, -0.15, 0.75), False),  # outside the x footprint
        ((0.0, -0.15, 0.70), False),  # below the deck, not resting on it
        ((0.0, -0.15, 0.91), False),  # above the occupancy volume
    ],
)
def test_shelf_region_contains(position, expected):
    assert region().contains(position) is expected


def test_shelf_region_rejects_nonpositive_occupancy_height():
    with pytest.raises(ValueError):
        ShelfRegion("a", (0, 0, 0), (0.1, 0.1, 0.1), occupancy_height=0.0)


def test_shelf_region_rejects_negative_occupancy_tolerance():
    with pytest.raises(ValueError):
        ShelfRegion("a", (0, 0, 0), (0.1, 0.1, 0.1), occupancy_tolerance=-0.01)


def test_shelf_region_rejects_empty_name():
    with pytest.raises(ValueError):
        ShelfRegion("", (0, 0, 0), (0.1, 0.1, 0.1))


def test_shelf_region_rejects_nonpositive_half_extent():
    with pytest.raises(ValueError):
        ShelfRegion("a", (0, 0, 0), (0.1, 0.0, 0.1))


def test_object_variation_rejects_inverted_range():
    with pytest.raises(ValueError):
        ObjectVariation((0.1, -0.1), (0, 1), (0, 1), ((0.01, 0.02),) * 3)


def test_object_variation_rejects_nonpositive_size_lower_bound():
    with pytest.raises(ValueError):
        ObjectVariation((0, 1), (0, 1), (0, 1), ((0.0, 0.02), (0.01, 0.02), (0.01, 0.02)))


def test_sample_pose_is_deterministic_given_a_seeded_rng():
    v = variation()
    a = v.sample_pose(0.75, random.Random(0))
    b = v.sample_pose(0.75, random.Random(0))
    assert a.position == b.position
    assert a.orientation == b.orientation
    assert a.position[2] == pytest.approx(0.75)
    assert v.position_x_range[0] <= a.position[0] <= v.position_x_range[1]


def test_sample_size_stays_within_configured_ranges():
    v = variation()
    size = v.sample_size(random.Random(1))
    for value, (lo, hi) in zip(size, v.size_xyz_range, strict=True):
        assert lo <= value <= hi


def test_shelf_restock_spec_rejects_nonpositive_compaction_distance():
    with pytest.raises(ValueError):
        ShelfRestockSpec(region(), region(), variation(), compaction_distance=0.0)


def test_shelf_restock_spec_rejects_empty_instruction():
    with pytest.raises(ValueError):
        ShelfRestockSpec(region(), region(), variation(), instruction="  ")


def test_default_spec_lower_and_upper_shelf_do_not_overlap():
    lower, upper = DEFAULT_SPEC.lower_shelf, DEFAULT_SPEC.upper_shelf
    lower_top = lower.top_z
    upper_bottom = upper.center_xyz[2] - upper.half_extent_xyz[2]
    assert lower_top < upper_bottom


def test_upper_deck_does_not_overhang_the_object_spawn_zone():
    """A top-down grasp puts the wrist above the object, so an object beneath
    the deck cannot be grasped at all -- measured live, not hypothetical."""
    upper = DEFAULT_SPEC.upper_shelf
    deck_near_edge = upper.center_xyz[1] - upper.half_extent_xyz[1]
    _, spawn_far_y = DEFAULT_SPEC.object_variation.position_y_range
    assert spawn_far_y < deck_near_edge, (
        f"objects spawn up to y={spawn_far_y} but the deck starts at y={deck_near_edge}"
    )


def test_object_spawn_zone_lies_within_the_lower_shelf():
    lower = DEFAULT_SPEC.lower_shelf
    variation = DEFAULT_SPEC.object_variation
    for axis, (lo, hi) in enumerate((variation.position_x_range, variation.position_y_range)):
        region_lo = lower.center_xyz[axis] - lower.half_extent_xyz[axis]
        region_hi = lower.center_xyz[axis] + lower.half_extent_xyz[axis]
        assert region_lo <= lo and hi <= region_hi
