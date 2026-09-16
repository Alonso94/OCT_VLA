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

    # "a" is now up top, where a real scene would still report it: the
    # neighbour must be observed on the upper shelf, not merely remembered.
    remaining = ObjectScene(0.0, (obj("a", (0.0, 0.10, 1.05)), obj("b", (0.0, -0.15, 0.74))))
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


def test_a_placed_object_knocked_back_down_is_not_its_own_previous_neighbour():
    """It stays in placement history but returns to the lower-shelf
    candidates, so without the scene check it could be selected as the target
    while still being reported as the neighbour -- which TaskContext rejects."""
    manager = ShelfRestockManager(spec())
    manager.record_placement("obj_0")
    context = manager.next_context(lower_scene("obj_0"))

    assert context.target_track_id == "obj_0"
    assert context.previous_neighbor_track_id is None


def test_the_neighbour_is_the_most_recent_placement_still_on_the_upper_shelf():
    manager = ShelfRestockManager(spec())
    manager.record_placement("obj_0")
    manager.record_placement("obj_1")
    scene = ObjectScene(
        0.0,
        (
            obj("obj_0", (0.0, 0.10, 1.05)),
            obj("obj_1", (0.0, -0.15, 0.74)),
            obj("obj_2", (0.0, -0.15, 0.74)),
        ),
    )
    context = manager.next_context(scene)

    # obj_1 was placed most recently but is back down, so obj_0 is the live one.
    assert context.previous_neighbor_track_id == "obj_0"


def test_an_upper_shelf_object_this_manager_never_placed_is_still_a_neighbour():
    """Atomic demonstrations start from a scene that already holds a restocked
    object, so the neighbour cannot come from placement history alone."""
    manager = ShelfRestockManager(spec())
    scene = ObjectScene(0.0, (obj("obj_0", (0.0, -0.15, 0.74)), obj("obj_1", (0.0, 0.10, 1.05))))
    context = manager.next_context(scene)

    assert context.target_track_id == "obj_0"
    assert context.previous_neighbor_track_id == "obj_1"


def test_placement_order_wins_over_a_bare_upper_shelf_object():
    """When the manager has placed something itself, that is the neighbour the
    row was built against; an unrelated object up there must not displace it."""
    manager = ShelfRestockManager(spec())
    manager.record_placement("obj_2")
    scene = ObjectScene(
        0.0,
        (
            obj("obj_0", (0.0, -0.15, 0.74)),
            obj("obj_1", (0.0, 0.10, 1.05)),
            obj("obj_2", (0.05, 0.10, 1.05)),
        ),
    )
    assert manager.next_context(scene).previous_neighbor_track_id == "obj_2"


def test_sweeping_objects_off_the_shelf_is_not_a_success():
    """`is_done` only asks whether the lower shelf is clear, which a policy can
    satisfy by knocking everything onto the floor -- one did, and scored a
    success with zero transfers. Success must require the objects to be
    somewhere specific."""
    s = spec()
    floor = ObjectScene(0.0, (obj("a", (9.0, 9.0, 0.0)), obj("b", (9.0, 9.1, 0.0))))
    manager = ShelfRestockManager(s)
    assert manager.is_done(floor) is True, "the lower shelf is indeed clear"
    assert manager.is_restocked(floor) is False, "but nothing was restocked"


def test_all_objects_on_the_upper_shelf_is_a_success():
    s = spec()
    done = ObjectScene(0.0, (obj("a", (0.0, 0.10, 1.05)), obj("b", (0.05, 0.10, 1.05))))
    assert ShelfRestockManager(s).is_restocked(done) is True


def test_a_partial_transfer_is_not_a_success():
    s = spec()
    partial = ObjectScene(0.0, (obj("a", (0.0, 0.10, 1.05)), obj("b", (0.0, -0.15, 0.74))))
    assert ShelfRestockManager(s).is_restocked(partial) is False


def test_an_empty_scene_is_not_a_success():
    """Vacuous truth would make a scene that lost every object look solved."""
    assert ShelfRestockManager(spec()).is_restocked(ObjectScene(0.0, ())) is False
