"""Shelf-restocking task specification: geometry and per-episode variation ranges.

Pure data -- no SAPIEN, no RoboTwin. A scene builder (a later commit)
instantiates a live scene FROM this spec; nothing here depends on it.

Geometry defaults below are a starting point, not yet validated against a real
SAPIEN scene. The old reference implementation's shelf sat inside the arm's
straight-line base-to-source approach corridor and failed 0/545 collection
attempts for exactly that reason (docs/shelf_restock_grasp_investigation.md in
the old repo); the upper shelf here is deliberately offset in y from the
lower shelf/source region to avoid repeating that failure, but this must
still be confirmed once a real scene exists to plan motions against.
"""

import random
from dataclasses import dataclass

from oct_vla.core.frames import Pose
from oct_vla.core.geometry import exp, finite_values


def _range(value: tuple[float, float], name: str) -> tuple[float, float]:
    lo, hi = finite_values(value, 2)
    if lo > hi:
        raise ValueError(f"{name} must be a (low, high) pair with low <= high")
    return lo, hi


def _positive(value: float, name: str) -> float:
    (value,) = finite_values((value,), 1)
    if value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class ShelfRegion:
    """One shelf level: a thin physical deck plus the occupancy volume above it.

    ``half_extent_xyz`` describes the deck's own geometry (thin in z; used to
    build its collision/visual box and to compute ``top_z``). ``contains()``
    answers a different question -- "is this position resting on this
    shelf?" -- so it checks the deck's xy footprint but a taller z band above
    the deck surface, since a real object's *center* sits above the deck by
    roughly half its own height, not inside the deck's own thin body.
    """

    name: str
    center_xyz: tuple[float, float, float]
    half_extent_xyz: tuple[float, float, float]
    occupancy_height: float = 0.15
    occupancy_tolerance: float = 0.02

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ShelfRegion name must be a nonempty string")
        object.__setattr__(self, "center_xyz", finite_values(self.center_xyz, 3))
        half_extent = finite_values(self.half_extent_xyz, 3)
        if not all(v > 0 for v in half_extent):
            raise ValueError("half_extent_xyz must be finite and positive")
        object.__setattr__(self, "half_extent_xyz", half_extent)
        object.__setattr__(
            self, "occupancy_height", _positive(self.occupancy_height, "occupancy_height")
        )
        (tolerance,) = finite_values((self.occupancy_tolerance,), 1)
        if tolerance < 0:
            raise ValueError("occupancy_tolerance must be finite and nonnegative")
        object.__setattr__(self, "occupancy_tolerance", tolerance)

    @property
    def top_z(self) -> float:
        """World/workcell z of the deck surface, assuming center_xyz is the deck's center."""
        return self.center_xyz[2] + self.half_extent_xyz[2]

    def contains(self, position_xyz: tuple[float, float, float]) -> bool:
        x, y, z = finite_values(position_xyz, 3)
        cx, cy, _ = self.center_xyz
        hx, hy, _ = self.half_extent_xyz
        if abs(x - cx) > hx or abs(y - cy) > hy:
            return False
        return self.top_z - self.occupancy_tolerance <= z <= self.top_z + self.occupancy_height


@dataclass(frozen=True)
class ObjectVariation:
    """Uniform per-episode sampling ranges for one spawned object."""

    position_x_range: tuple[float, float]
    position_y_range: tuple[float, float]
    yaw_range: tuple[float, float]
    size_xyz_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position_x_range", _range(self.position_x_range, "position_x_range")
        )
        object.__setattr__(
            self, "position_y_range", _range(self.position_y_range, "position_y_range")
        )
        object.__setattr__(self, "yaw_range", _range(self.yaw_range, "yaw_range"))
        size_range = tuple(_range(axis, "size_xyz_range") for axis in self.size_xyz_range)
        if not all(lo > 0 for lo, _ in size_range):
            raise ValueError("size_xyz_range lower bounds must be positive")
        object.__setattr__(self, "size_xyz_range", size_range)

    def sample_pose(self, z: float, rng: random.Random) -> Pose:
        x = rng.uniform(*self.position_x_range)
        y = rng.uniform(*self.position_y_range)
        yaw = rng.uniform(*self.yaw_range)
        return Pose((x, y, z), exp((0.0, 0.0, yaw)))

    def sample_size(self, rng: random.Random) -> tuple[float, float, float]:
        return tuple(rng.uniform(lo, hi) for lo, hi in self.size_xyz_range)


@dataclass(frozen=True)
class ShelfRestockSpec:
    lower_shelf: ShelfRegion
    upper_shelf: ShelfRegion
    object_variation: ObjectVariation
    spawn_clearance: float = 0.001
    compaction_distance: float = 0.03
    instruction: str = (
        "Restock the selected object from the lower shelf to the upper shelf, "
        "then compact it toward the previous neighbor if one exists."
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "spawn_clearance", _positive(self.spawn_clearance, "spawn_clearance")
        )
        object.__setattr__(
            self, "compaction_distance", _positive(self.compaction_distance, "compaction_distance")
        )
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction must be a nonempty string")


DEFAULT_SPEC = ShelfRestockSpec(
    lower_shelf=ShelfRegion("lower_shelf", (0.0, -0.15, 0.74), (0.22, 0.10, 0.01)),
    upper_shelf=ShelfRegion("upper_shelf", (0.0, 0.10, 1.05), (0.20, 0.10, 0.015)),
    object_variation=ObjectVariation(
        position_x_range=(-0.18, 0.18),
        position_y_range=(-0.20, -0.10),
        yaw_range=(-0.4, 0.4),
        size_xyz_range=((0.03, 0.06), (0.03, 0.06), (0.03, 0.08)),
    ),
)
