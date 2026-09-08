import pytest

from oct_vla.tasks.shelf_restock.oracle.world import shelf_cuboids
from oct_vla.tasks.shelf_restock.spec import ObjectVariation, ShelfRegion, ShelfRestockSpec


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
