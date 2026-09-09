from math import sqrt

import pytest

from oct_vla.tasks.shelf_restock.oracle.world import planner_cuboids, shelf_cuboids
from oct_vla.tasks.shelf_restock.spec import ObjectVariation, ShelfRegion, ShelfRestockSpec

# All poses in this file follow world.py's convention: [x, y, z, qw, qx, qy, qz].
IDENTITY_ORIGIN = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
_HALF_SQRT2 = sqrt(2.0) / 2.0
# +90 degree rotation about z, chosen so the rotated components land on axes.
Z_90_ORIGIN = (0.0, 0.0, 0.0, _HALF_SQRT2, 0.0, 0.0, _HALF_SQRT2)


def spec() -> ShelfRestockSpec:
    return ShelfRestockSpec(
        lower_shelf=ShelfRegion("lower_shelf", (0.0, -0.15, 0.74), (0.2, 0.1, 0.01)),
        upper_shelf=ShelfRegion("upper_shelf", (0.0, 0.10, 1.05), (0.2, 0.1, 0.015)),
        object_variation=ObjectVariation(
            (-0.1, 0.1), (-0.2, -0.1), (-0.2, 0.2), ((0.03, 0.05),) * 3
        ),
    )


def test_shelf_cuboids_only_describes_the_upper_shelf():
    cuboids = shelf_cuboids(spec())
    assert set(cuboids) == {"upper_shelf"}


def test_shelf_cuboids_dims_are_full_extents_not_half():
    cuboids = shelf_cuboids(spec())
    assert cuboids["upper_shelf"]["dims"] == pytest.approx([0.4, 0.2, 0.03])


def test_shelf_cuboids_pose_is_xyz_plus_identity_wxyz_quaternion():
    cuboids = shelf_cuboids(spec())
    assert cuboids["upper_shelf"]["pose"] == pytest.approx([0.0, 0.10, 1.05, 1.0, 0.0, 0.0, 0.0])


def test_planner_cuboids_translated_base_subtracts_translation():
    cuboids = {"box": {"dims": [0.1, 0.1, 0.1], "pose": [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0]}}
    origin = (1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
    converted = planner_cuboids(cuboids, origin)
    assert converted["box"]["pose"][:3] == pytest.approx([4.0, 3.0, 2.0])
    assert converted["box"]["pose"][3:] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_planner_cuboids_rotated_base_rotates_position_and_orientation():
    """A base rotated +90deg about z sees a +x point as being along its -y axis."""
    cuboids = {"box": {"dims": [0.1, 0.1, 0.1], "pose": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}}
    converted = planner_cuboids(cuboids, Z_90_ORIGIN)
    assert converted["box"]["pose"][:3] == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)
    # Orientation is wRb.T @ wRt = wRb.T here, i.e. a -90deg rotation about z.
    assert converted["box"]["pose"][3:] == pytest.approx(
        [_HALF_SQRT2, 0.0, 0.0, -_HALF_SQRT2], abs=1e-9
    )


def test_planner_cuboids_identity_base_leaves_cuboids_unchanged():
    cuboids = {"box": {"dims": [0.2, 0.3, 0.4], "pose": [1.0, -2.0, 0.5, 0.9, 0.1, 0.2, 0.3]}}
    # Normalise the arbitrary quaternion so it is a valid pose in the first place.
    quaternion = cuboids["box"]["pose"][3:]
    norm = sqrt(sum(value**2 for value in quaternion))
    normalised = [value / norm for value in quaternion]
    cuboids["box"]["pose"] = [1.0, -2.0, 0.5, *normalised]
    converted = planner_cuboids(cuboids, IDENTITY_ORIGIN)
    assert converted["box"]["pose"] == pytest.approx(cuboids["box"]["pose"])


def test_planner_cuboids_passes_dims_through_and_does_not_mutate_input():
    original_pose = [5.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0]
    cuboids = {"box": {"dims": [0.1, 0.2, 0.3], "pose": list(original_pose)}}
    converted = planner_cuboids(cuboids, (1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0))
    assert converted["box"]["dims"] == pytest.approx([0.1, 0.2, 0.3])
    assert cuboids["box"]["pose"] == pytest.approx(original_pose)


def test_planner_cuboids_converts_every_cuboid_and_preserves_keys():
    cuboids = {
        "box_a": {"dims": [0.1, 0.1, 0.1], "pose": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
        "box_b": {"dims": [0.2, 0.2, 0.2], "pose": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
    }
    converted = planner_cuboids(cuboids, IDENTITY_ORIGIN)
    assert set(converted) == {"box_a", "box_b"}
    assert converted["box_a"]["pose"][:3] == pytest.approx([1.0, 0.0, 0.0])
    assert converted["box_b"]["pose"][:3] == pytest.approx([0.0, 1.0, 0.0])


def test_planner_cuboids_zero_norm_base_quaternion_raises():
    cuboids = {"box": {"dims": [0.1, 0.1, 0.1], "pose": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}}
    with pytest.raises(ValueError):
        planner_cuboids(cuboids, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


def test_planner_cuboids_non_normalised_base_quaternion_is_normalised():
    cuboids = {"box": {"dims": [0.1, 0.1, 0.1], "pose": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}}
    scaled_origin = (1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0)  # same rotation as identity, un-normalised
    normalised_origin = (1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
    scaled = planner_cuboids(cuboids, scaled_origin)
    normalised = planner_cuboids(cuboids, normalised_origin)
    assert scaled["box"]["pose"] == pytest.approx(normalised["box"]["pose"])


def test_planner_cuboids_round_trip_cuboid_at_base_pose_lands_at_origin():
    origin = (1.0, -2.0, 0.5, _HALF_SQRT2, 0.0, 0.0, _HALF_SQRT2)
    cuboids = {"box": {"dims": [0.1, 0.1, 0.1], "pose": list(origin)}}
    converted = planner_cuboids(cuboids, origin)
    assert converted["box"]["pose"][:3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert converted["box"]["pose"][3:] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-9)
