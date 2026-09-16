"""Thin deterministic task manager: selects targets and repeats until the lower
shelf is empty. It executes no manipulation skill of its own -- the policy (or,
for demonstration collection, the oracle) performs each restock; this module
only decides which object is next and what the previous-neighbor context is.

ObjectScene carries no placement-order timestamp per object, so "the previous
neighbor" cannot be derived from a scene snapshot alone: the manager tracks
placement order itself, across calls, for one episode.
"""

from collections.abc import Callable

from oct_vla.core.objects import ObjectScene, TaskContext
from oct_vla.tasks.shelf_restock.spec import ShelfRestockSpec

Selector = Callable[[tuple[str, ...], ObjectScene], str]


def _select_lowest_track_id(remaining: tuple[str, ...], scene: ObjectScene) -> str:
    return min(remaining)


def objects_on_lower_shelf(scene: ObjectScene, spec: ShelfRestockSpec) -> tuple[str, ...]:
    return tuple(o.track_id for o in scene.objects if spec.lower_shelf.contains(o.pose.position))


def objects_on_upper_shelf(scene: ObjectScene, spec: ShelfRestockSpec) -> tuple[str, ...]:
    return tuple(o.track_id for o in scene.objects if spec.upper_shelf.contains(o.pose.position))


class ShelfRestockManager:
    def __init__(
        self, spec: ShelfRestockSpec, *, selector: Selector = _select_lowest_track_id
    ) -> None:
        self.spec = spec
        self._selector = selector
        self._placed_order: list[str] = []

    def is_done(self, scene: ObjectScene) -> bool:
        """Nothing remains to be restocked.

        True once the lower shelf is clear, however it was cleared. That suits
        the oracle, which only ever moves objects upward, but it is NOT a
        success test for a policy -- see `is_restocked`.
        """
        return not objects_on_lower_shelf(scene, self.spec)

    def is_restocked(self, scene: ObjectScene) -> bool:
        """Every object in the scene is on the upper shelf.

        The success criterion for closed-loop evaluation, and deliberately
        stricter than `is_done`. Emptying the lower shelf is not the task:
        a policy that sweeps the objects onto the floor also empties it, and
        scored a full success under the weaker test while having transferred
        nothing. Requiring the objects to be *somewhere specific* cannot be
        satisfied by destroying the scene.
        """
        return bool(scene.objects) and len(objects_on_upper_shelf(scene, self.spec)) == len(
            scene.objects
        )

    def next_context(self, scene: ObjectScene) -> TaskContext | None:
        remaining = objects_on_lower_shelf(scene, self.spec)
        if not remaining:
            return None
        target = self._selector(remaining, scene)
        if target not in remaining:
            raise ValueError("Selector must choose a track_id from the given remaining objects")
        previous = self._previous_neighbor(target, scene)
        return TaskContext(
            instruction=self.spec.instruction,
            target_track_id=target,
            previous_neighbor_track_id=previous,
        )

    def _previous_neighbor(self, target: str, scene: ObjectScene) -> str | None:
        """The most recently placed object still up there to compact against.

        Not simply the last entry in placement order. An object that was
        placed and then knocked back down is still in that history, but it is
        no longer a neighbour on the upper shelf, and because it is back among
        the lower-shelf candidates it can be selected as the target -- which
        made it its own previous neighbour and raised out of TaskContext
        mid-collection. Both conditions are checked here rather than trusting
        placement order, since the scene is the authority on where things are.
        """
        upper = objects_on_upper_shelf(scene, self.spec)
        placed = set(upper)
        for track_id in reversed(self._placed_order):
            if track_id != target and track_id in placed:
                return track_id
        # Nothing *this* manager placed is up there, but something may be
        # anyway: an atomic demonstration starts from a scene that already
        # contains a restocked neighbour, precisely so compaction can be
        # demonstrated in a single transfer. Trusting placement order alone
        # reported no neighbour for exactly those scenes, so the oracle placed
        # the target where the neighbour already stood and the plan failed.
        # The scene, not this object's memory, is the authority on what is on
        # the shelf.
        candidates = tuple(track_id for track_id in upper if track_id != target)
        return min(candidates) if candidates else None

    def record_placement(self, track_id: str) -> None:
        self._placed_order.append(track_id)
