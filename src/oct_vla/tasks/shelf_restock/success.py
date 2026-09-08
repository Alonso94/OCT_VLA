"""Placement and compaction success, computed purely from ObjectScene + spec.

This checks geometry only (positions an estimator has already reported), not
contacts, collisions, or grasp quality -- those are simulation-privileged
diagnostics for the oracle/evaluator (a later commit), not a task-success
signal derivable from the canonical object schema alone.
"""

from dataclasses import dataclass

from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
from oct_vla.tasks.shelf_restock.geometry import horizontal_distance
from oct_vla.tasks.shelf_restock.spec import ShelfRestockSpec


def horizontal_radius(obj: ObjectState) -> float:
    """Half the object's larger horizontal dimension.

    Treating each object as a circle of this radius makes the neighbour gap
    size-aware and yaw-independent. It is conservative: for a non-square
    footprint the true surface gap along the line of centres is never
    smaller than the value this produces.
    """
    return max(obj.size_xyz[0], obj.size_xyz[1]) / 2.0


@dataclass(frozen=True)
class PlacementResult:
    target_on_upper_shelf: bool
    target_to_neighbor_gap: float | None
    compacted_to_neighbor: bool | None

    @property
    def success(self) -> bool:
        if self.compacted_to_neighbor is None:
            return self.target_on_upper_shelf
        return self.target_on_upper_shelf and self.compacted_to_neighbor


def check_placement(
    scene: ObjectScene, context: TaskContext, spec: ShelfRestockSpec
) -> PlacementResult:
    target = scene.get(context.target_track_id)
    if target is None:
        raise ValueError(f"TaskContext target_track_id {context.target_track_id!r} not in scene")

    on_upper_shelf = spec.upper_shelf.contains(target.pose.position)

    if context.previous_neighbor_track_id is None:
        return PlacementResult(on_upper_shelf, None, None)

    neighbor_id = context.previous_neighbor_track_id
    neighbor = scene.get(neighbor_id)
    if neighbor is None:
        raise ValueError(f"TaskContext previous_neighbor_track_id {neighbor_id!r} not in scene")
    # Gap between surfaces, not between centres: centre-to-centre cannot express
    # "compacted" at all, since two touching objects are already
    # (width_a + width_b) / 2 apart, which exceeds any sane threshold and made
    # this check unsatisfiable for every object size the spec allows.
    gap = (
        horizontal_distance(target.pose, neighbor.pose)
        - horizontal_radius(target)
        - horizontal_radius(neighbor)
    )
    return PlacementResult(on_upper_shelf, gap, gap <= spec.compaction_distance)
