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


def _select_leftmost(remaining: tuple[str, ...], scene: ObjectScene) -> str:
    """The remaining object with the smallest workcell x.

    The target has to be something the *observation* determines, and a track id
    is not. It is an internal simulator identifier: absent from the images, and
    dropped along with the role one-hot in the role-stripped token mode. So
    choosing by lowest track id -- the previous rule -- paired near-identical
    observations with actions toward different objects, and behaviour cloning's
    optimum on an ill-posed mapping like that is an average that reaches none of
    them. Only the object-conditioned arm could ever see the answer, which is
    also why it held the lowest training loss in every sweep.

    Position is visible in every observation variant: in the images, in the
    object tokens, and in the tokens even after role stripping. Objects are
    spawned roughly 0.2 m apart across a 0.45 m span, so the ordering is
    unambiguous at the scale of the spawn jitter.

    Ties fall back to the track id so the rule stays total and deterministic;
    with that spacing a tie means two objects at the same x to the floating
    point, which the spawner does not produce.
    """
    by_position = {o.track_id: o.pose.position[0] for o in scene.objects}
    return min(remaining, key=lambda track_id: (by_position[track_id], track_id))


def objects_on_lower_shelf(scene: ObjectScene, spec: ShelfRestockSpec) -> tuple[str, ...]:
    return tuple(o.track_id for o in scene.objects if spec.lower_shelf.contains(o.pose.position))


def objects_on_upper_shelf(scene: ObjectScene, spec: ShelfRestockSpec) -> tuple[str, ...]:
    return tuple(o.track_id for o in scene.objects if spec.upper_shelf.contains(o.pose.position))


#: How far above the lower deck an object must rise to count as picked up, in
#: metres. Comfortably beyond the shelf's own occupancy tolerance, so settling
#: or being nudged along the deck does not register as a lift.
LIFT_CLEARANCE = 0.08


def objects_lifted(scene: ObjectScene, spec: ShelfRestockSpec) -> tuple[str, ...]:
    """Objects currently raised clear of the lower shelf.

    The coarsest of the three progress measures, and the one that separates
    "never touched anything" from "grasped it and lost it". Without it every
    failing policy scores an identical zero, which is exactly what four sweeps
    have reported -- a policy that lifts an object and drops it is much closer
    to working than one that never moves the arm, and the aggregate could not
    tell them apart.

    Height alone, deliberately: an object swept onto the floor also leaves the
    lower shelf's xy footprint, so testing "not on the lower shelf" would score
    destruction as progress.
    """
    threshold = spec.lower_shelf.top_z + LIFT_CLEARANCE
    return tuple(o.track_id for o in scene.objects if o.pose.position[2] > threshold)


class ShelfRestockManager:
    def __init__(
        self, spec: ShelfRestockSpec, *, selector: Selector = _select_leftmost
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
