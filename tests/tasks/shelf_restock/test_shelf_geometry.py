import pytest

from oct_vla.core.frames import Pose
from oct_vla.tasks.shelf_restock.geometry import horizontal_distance, resting_region
from oct_vla.tasks.shelf_restock.spec import ObjectVariation, ShelfRegion, ShelfRestockSpec


def spec() -> ShelfRestockSpec:
    return ShelfRestockSpec(
        lower_shelf=ShelfRegion("lower_shelf", (0.0, -0.15, 0.74), (0.2, 0.1, 0.01)),
        upper_shelf=ShelfRegion("upper_shelf", (0.0, 0.10, 1.05), (0.2, 0.1, 0.015)),
        object_variation=ObjectVariation(
            (-0.1, 0.1), (-0.2, -0.1), (-0.2, 0.2), ((0.03, 0.05),) * 3
        ),
    )


def test_horizontal_distance_ignores_height():
    a = Pose((0.0, 0.0, 0.5), (0, 0, 0, 1))
    b = Pose((0.03, 0.04, 5.0), (0, 0, 0, 1))
    assert horizontal_distance(a, b) == pytest.approx(0.05)


def test_horizontal_distance_is_symmetric_and_zero_for_same_point():
    a = Pose((1.0, 2.0, 0.0), (0, 0, 0, 1))
    b = Pose((1.0, 2.0, 9.0), (0, 0, 0, 1))
    assert horizontal_distance(a, b) == pytest.approx(0.0)
    assert horizontal_distance(a, b) == pytest.approx(horizontal_distance(b, a))


def test_resting_region_identifies_lower_and_upper_shelf():
    s = spec()
    assert resting_region((0.0, -0.15, 0.74), s) == "lower_shelf"
    assert resting_region((0.0, 0.10, 1.05), s) == "upper_shelf"


def test_resting_region_returns_none_when_on_neither_shelf():
    assert resting_region((5.0, 5.0, 5.0), spec()) is None
