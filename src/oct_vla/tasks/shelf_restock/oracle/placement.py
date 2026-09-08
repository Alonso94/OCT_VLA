"""Where a restocked object goes on the upper shelf, and where it ends up
after being compacted toward its previous neighbour.

Pure geometry over the spec and an `ObjectScene`; no SAPIEN, no planner. The
oracle turns these object poses into commanded poses with
`grasps.holding_tcp_pose`, since placing a held object is the same
relationship as grasping it, read in the other direction.

The atomic task is "restock one object, then compact it toward the previous
neighbour if one exists", so placement deliberately leaves a gap wider than
the success threshold: if the object were simply placed at its final
compacted position there would be no compaction to demonstrate.
"""

from dataclasses import dataclass

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectState
from oct_vla.tasks.shelf_restock.oracle.arms import Side
from oct_vla.tasks.shelf_restock.oracle.grasps import (
    DEFAULT_GRASP_DEPTH,
    backed_off,
    holding_tcp_pose,
)
from oct_vla.tasks.shelf_restock.spec import ShelfRestockSpec
from oct_vla.tasks.shelf_restock.success import horizontal_radius

#: Surface gap left when first setting an object down beside its neighbour.
#: Wider than ShelfRestockSpec.compaction_distance so that compaction is a
#: real motion rather than a no-op.
PLACEMENT_GAP = 0.06

#: Surface gap after compaction: close enough to read as "pushed together",
#: with a little clearance so the nudge does not rely on interpenetration.
COMPACTED_GAP = 0.005


class PlacementError(RuntimeError):
    """Raised when no valid placement exists on the shelf."""


@dataclass(frozen=True)
class Placement:
    """Object poses, plus the commanded poses that put the object there."""

    object_pose: Pose
    place_pose: Pose
    preplace_pose: Pose
    compacted_object_pose: Pose | None
    compact_pose: Pose | None

    @property
    def needs_compaction(self) -> bool:
        return self.compacted_object_pose is not None


def resting_z(spec: ShelfRestockSpec, obj: ObjectState) -> float:
    """Height of an object's centre when it sits on the upper deck."""
    return spec.upper_shelf.top_z + obj.size_xyz[2] / 2.0


def _offset_position(
    neighbour: ObjectState, obj: ObjectState, gap: float, direction: float, z: float
):
    reach = horizontal_radius(neighbour) + horizontal_radius(obj) + gap
    return (neighbour.pose.position[0] + direction * reach, neighbour.pose.position[1], z)


def _placement_direction(spec: ShelfRestockSpec, neighbour: ObjectState, obj: ObjectState) -> float:
    """Put the object on whichever side of the neighbour has shelf room.

    Prefers the side with more space so a row builds outward rather than
    running off the deck.
    """
    shelf = spec.upper_shelf
    low = shelf.center_xyz[0] - shelf.half_extent_xyz[0]
    high = shelf.center_xyz[0] + shelf.half_extent_xyz[0]
    neighbour_x = neighbour.pose.position[0]
    needed = horizontal_radius(neighbour) + horizontal_radius(obj) + PLACEMENT_GAP
    room_positive = high - (neighbour_x + needed)
    room_negative = (neighbour_x - needed) - low
    if max(room_positive, room_negative) < 0.0:
        raise PlacementError(
            f"no room beside neighbour at x={neighbour_x:.3f} for an object needing "
            f"{needed:.3f}m on a shelf spanning [{low:.3f}, {high:.3f}]"
        )
    return 1.0 if room_positive >= room_negative else -1.0


def plan_placement(
    spec: ShelfRestockSpec,
    obj: ObjectState,
    wrist_yaw: float,
    neighbour: ObjectState | None = None,
    *,
    standoff: float,
    grasp_depth: float = DEFAULT_GRASP_DEPTH,
    side: Side | None = None,
) -> Placement:
    """Where to put `obj`, and how to command the arm holding it there.

    With no neighbour the object goes to the middle of the deck. With one, it
    goes beside it leaving PLACEMENT_GAP, and a compaction pose closes that
    to COMPACTED_GAP. `side` is accepted for symmetry with the rest of the
    oracle and is not yet used to bias placement.
    """
    z = resting_z(spec, obj)
    if neighbour is None:
        position = (spec.upper_shelf.center_xyz[0], spec.upper_shelf.center_xyz[1], z)
        compacted = None
    else:
        direction = _placement_direction(spec, neighbour, obj)
        position = _offset_position(neighbour, obj, PLACEMENT_GAP, direction, z)
        compacted = _offset_position(neighbour, obj, COMPACTED_GAP, direction, z)

    if not spec.upper_shelf.contains(position):
        raise PlacementError(
            f"placement {tuple(round(v, 3) for v in position)} is not on the upper shelf"
        )

    place_pose = holding_tcp_pose(position, obj.size_xyz[2], wrist_yaw, grasp_depth=grasp_depth)
    compact_pose = (
        holding_tcp_pose(compacted, obj.size_xyz[2], wrist_yaw, grasp_depth=grasp_depth)
        if compacted is not None
        else None
    )
    return Placement(
        object_pose=Pose(position, obj.pose.orientation, WORKCELL_FRAME),
        place_pose=place_pose,
        preplace_pose=backed_off(place_pose, standoff),
        compacted_object_pose=(
            Pose(compacted, obj.pose.orientation, WORKCELL_FRAME) if compacted else None
        ),
        compact_pose=compact_pose,
    )
