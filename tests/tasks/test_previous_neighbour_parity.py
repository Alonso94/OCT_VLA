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
