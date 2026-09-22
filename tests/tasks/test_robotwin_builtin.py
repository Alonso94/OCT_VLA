"""The adapter's declarations, checked without a simulator.

Everything here runs on CPU with no RoboTwin import: the bindings, the guards
against a mis-declared spec, and the support geometry. What needs the simulator
-- that `_take_picture` actually fires and that `play_once` leaves
`plan_success` true -- is checked by the collection probe.
"""

import pytest

from oct_vla.core.frames import WORKCELL_FRAME
from oct_vla.tasks.robotwin_builtin.adapter import (
    PLACE_CONTAINER_PLATE,
    BuiltinTaskSpec,
    ObjectBinding,
    spec_for,
)


class FakeTask:
    """Stands in for a RoboTwin task instance: objects as named attributes."""

    def __init__(self):
        self.container = object()
        self.plate = object()
        self.actor_name = "021_cup"
        self.container_id = 3
        self.plate_id = 0


def test_a_binding_resolves_the_asset_and_variant_the_task_drew():
    """Size must come from the mesh the episode actually spawned. RoboTwin
    picks the container's category and variant at random per episode, so a
    binding that hardcoded either would mislabel most of the corpus."""
    task = FakeTask()
    actor, name, variant = PLACE_CONTAINER_PLATE.objects[0].resolve(task)
    assert actor is task.container
    assert (name, variant) == ("021_cup", 3)

    # The plate's asset name is fixed, its variant read from the task.
    actor, name, variant = PLACE_CONTAINER_PLATE.objects[1].resolve(task)
    assert actor is task.plate
    assert (name, variant) == ("003_plate", 0)


def test_a_binding_naming_an_object_the_task_does_not_spawn_is_loud():
    binding = ObjectBinding("hammer", "hammer", modelname="020_hammer")
    with pytest.raises(AttributeError, match="does not spawn"):
        binding.resolve(FakeTask())


def test_a_target_outside_the_bindings_is_refused_at_declaration():
    """`validate_episode` requires every sample's target to be in the scene, so
    a spec naming an object it never records would produce a corpus where every
    episode fails validation -- after collection, not before it."""
    with pytest.raises(ValueError, match="not among"):
        BuiltinTaskSpec(
            task_name="t",
            instruction="i",
            target_track_id="nonexistent",
            objects=(ObjectBinding("plate", "plate", modelname="003_plate"),),
        )


def test_duplicate_track_ids_are_refused():
    with pytest.raises(ValueError, match="duplicate track_id"):
        BuiltinTaskSpec(
            task_name="t",
            instruction="i",
            target_track_id="a",
            objects=(
                ObjectBinding("container", "a", modelname="002_bowl"),
                ObjectBinding("plate", "a", modelname="003_plate"),
            ),
        )


def test_the_work_surface_is_one_support_entity_in_the_workcell_frame():
    """Our own task has two shelf decks; a RoboTwin tabletop has one table. The
    entity set must carry it either way, or the policy sees objects floating
    with nothing to rest on."""
    supports = PLACE_CONTAINER_PLATE.supports()
    assert len(supports) == 1
    assert supports[0].pose.frame == WORKCELL_FRAME
    assert all(value > 0 for value in supports[0].size_xyz)


def test_the_registry_names_the_task_and_refuses_an_unknown_one():
    assert spec_for("place_container_plate") is PLACE_CONTAINER_PLATE
    assert PLACE_CONTAINER_PLATE.target_track_id == "container"
    with pytest.raises(ValueError, match="no binding for"):
        spec_for("beat_block_hammer")
