"""Shelf-restocking task specification: geometry and per-episode variation ranges.

Pure data -- no SAPIEN, no RoboTwin. A scene builder instantiates a live
scene FROM this spec; nothing here depends on it.

`upper_shelf`'s position was corrected by live reachability sweeps against
the real scene (docs/architecture.md), not derived analytically. The first
choice -- center (0.0, 0.10, 1.05), i.e. y=0.10, top_z=1.065 -- failed a
top-down global plan for *both* arms at every height/standoff tried, even
though it was deliberately offset in y from the lower shelf/source region to
avoid the old reference implementation's corridor-collision failure (0/545
demo collection attempts; docs/shelf_restock_grasp_investigation.md in the
old repo). A position/arm sweep found the actual limiting factor was height
and forward (+y) reach, not which arm or x-position: targets around
z=0.90-1.00, y=-0.05 to 0.00 landed within 5-10mm of the requested pose for
both arms, including at x=0.0 dead center, while y=0.10 at any height from
1.065-1.185 failed or landed 0.12-0.27m off target regardless of arm. The
values below reflect that evidence, not the original corridor-avoidance
reasoning alone.
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


# The hand is wider than the object it grasps: cuRobo's own collision spheres
# put panda_hand's outermost centres at y=+-0.08 with radius 0.023, so it is
# ~0.10m from the grasp axis to the outside of the hand. Spawning objects
# closer than that means descending onto one shoves its neighbour -- observed
# live as panda_hand <-> restock_object_1 during a descent, which the oracle's
# contact check now rejects outright (docs/architecture.md).
#
# It lives here rather than in robotwin_env because it is task geometry, not
# simulator wiring: the spawn span in DEFAULT_SPEC is sized directly against
# it, and this module imports nothing from RoboTwin so that invariant can be
# asserted in a unit test.
MIN_OBJECT_SEPARATION = 0.15


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
    compaction_distance: float = 0.04
    #: Names the rule the selector actually uses. "The selected object" named
    #: nothing an observation could resolve, so a language-conditioned policy
    #: had no more to go on than a vision-only one.
    instruction: str = (
        "Restock the leftmost object from the lower shelf to the upper shelf, "
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


# Objects spawn on the left half: ROLE_ARMS (oracle/arms.py) fixes the left
# arm as the one that both grasps and places, with its base at x=-0.4 (the
# right arm's base is at x=+0.4), and spawning across the full width forced
# that arm into cross-body reaches. lower_shelf's x half-extent is sized
# purely so `contains()` still covers the whole spawn range -- it is the
# region test widening, not new physical deck geometry.
#
# The spawn span is deliberately sized for the LARGEST profile rather than the
# default one. Every profile -- two, three and four objects -- shares this one
# spec and differs only in `object_count` (robotwin_env.py), so that a
# count-shift evaluation result attributes to the object count alone. Sizing
# per profile instead, as an earlier revision did, meant the four-object
# profile also moved the spawn region, and a policy evaluated on it faced a
# combined count-and-position shift that could not be attributed to either.
# Four objects at MIN_OBJECT_SEPARATION (0.15m, defined above) reserve 0.45m
# of gap, so the span carries margin above that -- the sampler subtracts the
# reserved gaps and must not reject a harmless floating-point residual as
# negative. Two and three objects fit trivially inside the same span.
#
# The upper deck must also not overhang where objects spawn: a top-down grasp
# puts the wrist links above the object, so an object under the deck cannot be
# grasped at all (see docs/architecture.md). Beyond that, the deck moved
# forward from y=-0.02 to y=-0.06 so placements land at y < CROSS_BODY_Y
# (-0.05, oracle/arms.py): at y=-0.02, every placement past |x|=0.15 fell
# inside a measured cross-body-unreachable region, making it impossible for
# the right arm (compaction) to reach what the left arm had placed. It is
# deliberately not moved further forward than that.
#
# The deck's near edge sits at y=-0.12 and the spawn zone now starts at
# y=-0.20, so the grasp clearance between them is 0.08m -- down from the 0.12m
# the deck's position was originally chosen to guarantee, because the spawn
# rows were later moved forward. `contains()`-level overhang is still ruled
# out (tests/tasks/shelf_restock/test_spec.py asserts it), but the margin is
# thinner than it was; moving the spawn any further forward eats into the room
# the wrist needs above a spawned object.
#
# upper_shelf's x centre is offset to -0.06 rather than sitting on the
# workcell midline. The placement chain starts one step to the -x side of that
# centre and grows +x (see plan_placement), so with a centre of 0.0 a
# four-object row put its last placement near x=+0.13 -- past where the left
# arm could plan to at the deck's y, and the fourth transfer failed on both
# seeds tried. Offsetting the centre moves the whole row back into reach.
DEFAULT_SPEC = ShelfRestockSpec(
    lower_shelf=ShelfRegion("lower_shelf", (-0.07, -0.25, 0.74), (0.50, 0.10, 0.01)),
    upper_shelf=ShelfRegion("upper_shelf", (-0.06, -0.06, 0.92), (0.30, 0.06, 0.015)),
    object_variation=ObjectVariation(
        position_x_range=(-0.49, -0.02),
        position_y_range=(-0.28, -0.20),
        yaw_range=(-0.4, 0.4),
        size_xyz_range=((0.03, 0.06), (0.03, 0.06), (0.03, 0.08)),
    ),
)
