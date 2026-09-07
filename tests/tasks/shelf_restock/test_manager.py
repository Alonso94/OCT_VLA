import pytest

from oct_vla.core.frames import Pose
from oct_vla.core.objects import ObjectScene, ObjectState
from oct_vla.tasks.shelf_restock.manager import (
    ShelfRestockManager,
    objects_on_lower_shelf,
    objects_on_upper_shelf,
)
from oct_vla.tasks.shelf_restock.spec import ObjectVariation, ShelfRegion, ShelfRestockSpec


def spec() -> ShelfRestockSpec:
    return ShelfRestockSpec(
        lower_shelf=ShelfRegion("lower_shelf", (0.0, -0.15, 0.74), (0.2, 0.1, 0.01)),
        upper_shelf=ShelfRegion("upper_shelf", (0.0, 0.10, 1.05), (0.2, 0.1, 0.015)),
        object_variation=ObjectVariation(
            (-0.1, 0.1), (-0.2, -0.1), (-0.2, 0.2), ((0.03, 0.05),) * 3
        ),
    )


def obj(track_id, position) -> ObjectState:
    return ObjectState(track_id, Pose(position, (0, 0, 0, 1)), (0.04, 0.04, 0.04), 1.0, 1.0)


def lower_scene(*track_ids) -> ObjectScene:
    return ObjectScene(0.0, tuple(obj(t, (0.0, -0.15, 0.74)) for t in track_ids))


def test_objects_on_lower_and_upper_shelf_partition_by_position():
    scene = ObjectScene(
        0.0, (obj("a", (0.0, -0.15, 0.74)), obj("b", (0.0, 0.10, 1.05)), obj("c", (9.0, 9.0, 9.0)))
    )
    s = spec()
    assert objects_on_lower_shelf(scene, s) == ("a",)
    assert objects_on_upper_shelf(scene, s) == ("b",)


def test_is_done_when_lower_shelf_is_empty():
    manager = ShelfRestockManager(spec())
    assert manager.is_done(lower_scene()) is True
    assert manager.is_done(lower_scene("a")) is False


def test_next_context_returns_none_when_lower_shelf_empty():
    manager = ShelfRestockManager(spec())
    assert manager.next_context(lower_scene()) is None


def test_next_context_selects_deterministically_and_has_no_previous_neighbor_first():
    manager = ShelfRestockManager(spec())
    context = manager.next_context(lower_scene("b", "a", "c"))
    assert context.target_track_id == "a"
    assert context.previous_neighbor_track_id is None
    assert context.instruction == spec().instruction


def test_record_placement_becomes_next_call_s_previous_neighbor():
    manager = ShelfRestockManager(spec())
    scene = lower_scene("a", "b")

    first = manager.next_context(scene)
    assert first.target_track_id == "a"
    manager.record_placement("a")

    remaining = lower_scene("b")
    second = manager.next_context(remaining)
    assert second.target_track_id == "b"
    assert second.previous_neighbor_track_id == "a"


def test_custom_selector_is_used():
    manager = ShelfRestockManager(spec(), selector=lambda remaining, scene: max(remaining))
    context = manager.next_context(lower_scene("a", "b"))
    assert context.target_track_id == "b"


def test_selector_returning_unknown_track_id_raises():
    manager = ShelfRestockManager(spec(), selector=lambda remaining, scene: "not-remaining")
    with pytest.raises(ValueError):
        manager.next_context(lower_scene("a"))
