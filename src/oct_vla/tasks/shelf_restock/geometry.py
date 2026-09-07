"""Pure geometric relations over shelf regions and object poses."""

from oct_vla.core.frames import Pose
from oct_vla.tasks.shelf_restock.spec import ShelfRestockSpec


def horizontal_distance(a: Pose, b: Pose) -> float:
    """XY-plane distance between two positions; ignores height so that a target
    resting slightly higher/lower than its neighbor still counts as compacted."""
    ax, ay, _ = a.position
    bx, by, _ = b.position
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def resting_region(position_xyz: tuple[float, float, float], spec: ShelfRestockSpec) -> str | None:
    """Name of the shelf region containing this position, or None if on neither."""
    if spec.lower_shelf.contains(position_xyz):
        return spec.lower_shelf.name
    if spec.upper_shelf.contains(position_xyz):
        return spec.upper_shelf.name
    return None
