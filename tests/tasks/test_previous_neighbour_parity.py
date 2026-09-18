"""Collection records each placement as the oracle finishes a transfer;
evaluation has to do the same. Without it `_placed_order` stays empty and
`_previous_neighbor` falls through to the lexicographically smallest neighbour
instead of the most recently placed one -- a different role token, from the
third transfer onward, for exactly the object-conditioned arms whose comparison
is the point of the project."""

from __future__ import annotations

from oct_vla.tasks.shelf_restock.manager import ShelfRestockManager


def test_placement_order_decides_the_previous_neighbour(monkeypatch):
    """Two objects on the upper shelf: the answer must be the later one, not
    the alphabetically smaller one."""
    import oct_vla.tasks.shelf_restock.manager as m

    monkeypatch.setattr(m, "objects_on_upper_shelf", lambda scene, spec: ("obj_a", "obj_c"))
    manager = ShelfRestockManager.__new__(ShelfRestockManager)
    manager._placed_order = []
    manager.spec = None

    # No history -- the fallback, which is what evaluation used to do always.
    assert manager._previous_neighbor("obj_b", scene=None) == "obj_a"

    # With history, the most recently placed wins.
    manager._placed_order = ["obj_a", "obj_c"]
    assert manager._previous_neighbor("obj_b", scene=None) == "obj_c"


def test_a_target_is_never_its_own_previous_neighbour(monkeypatch):
    import oct_vla.tasks.shelf_restock.manager as m

    monkeypatch.setattr(m, "objects_on_upper_shelf", lambda scene, spec: ("obj_a", "obj_b"))
    manager = ShelfRestockManager.__new__(ShelfRestockManager)
    manager._placed_order = ["obj_b"]
    manager.spec = None
    assert manager._previous_neighbor("obj_b", scene=None) == "obj_a"


def test_an_object_knocked_back_down_stops_being_a_neighbour(monkeypatch):
    """It stays in placement history but is no longer on the shelf, so it
    cannot be compacted against."""
    import oct_vla.tasks.shelf_restock.manager as m

    monkeypatch.setattr(m, "objects_on_upper_shelf", lambda scene, spec: ("obj_a",))
    manager = ShelfRestockManager.__new__(ShelfRestockManager)
    manager._placed_order = ["obj_a", "obj_c"]  # obj_c fell back down
    manager.spec = None
    assert manager._previous_neighbor("obj_b", scene=None) == "obj_a"


# ------------------------------------------------------------ target selection


def scene_with(positions: dict[str, float]):
    """A scene whose objects differ only in their workcell x."""
    from types import SimpleNamespace as NS

    return NS(objects=tuple(
        NS(track_id=t, pose=NS(position=(x, 0.0, 0.8))) for t, x in positions.items()
    ))


def test_the_target_is_the_leftmost_object_not_the_lowest_track_id():
    """A track id is absent from the images and dropped by role stripping, so
    selecting on it paired near-identical observations with actions toward
    different objects. Position is visible in every observation variant."""
    from oct_vla.tasks.shelf_restock.manager import _select_leftmost

    scene = scene_with({"obj_0": -0.25, "obj_1": -0.43, "obj_2": -0.08})
    assert _select_leftmost(("obj_0", "obj_1", "obj_2"), scene) == "obj_1"


def test_selection_follows_position_as_objects_are_removed():
    """Once the leftmost is restocked the next-leftmost becomes the target, so
    the rule stays resolvable from the observation for every transfer."""
    from oct_vla.tasks.shelf_restock.manager import _select_leftmost

    scene = scene_with({"obj_0": -0.25, "obj_1": -0.43, "obj_2": -0.08})
    assert _select_leftmost(("obj_0", "obj_2"), scene) == "obj_0"
    assert _select_leftmost(("obj_2",), scene) == "obj_2"


def test_a_tie_is_broken_deterministically():
    """The rule has to be total. The spawner does not place two objects at the
    same x to the float, but a selector that could return either would make the
    demonstrations inconsistent with themselves."""
    from oct_vla.tasks.shelf_restock.manager import _select_leftmost

    scene = scene_with({"obj_1": -0.30, "obj_0": -0.30})
    assert _select_leftmost(("obj_1", "obj_0"), scene) == "obj_0"


def test_the_instruction_names_the_rule():
    """'The selected object' named nothing an observation could resolve, so a
    language-conditioned policy had no more to go on than a vision-only one."""
    from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC

    assert "leftmost" in DEFAULT_SPEC.instruction
