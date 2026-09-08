"""Thin RoboTwin task lifecycle adapter for the shelf-restock scene.

Builds exactly the scene described by tasks/shelf_restock/spec.py: a static
upper-shelf deck plus randomly placed box objects on the lower shelf (the
RoboTwin table itself, reused as the lower shelf's deck rather than a second
static box -- there is no shelf mesh asset anywhere in RoboTwin, so both
levels are procedural boxes, matching the only geometry RoboTwin actually
offers). No oracle lives here: play_once is not implemented until a later
commit can actually execute and verify an episode against this scene.

Loaded as an external task entrypoint (module:ClassName), not copied into
the RoboTwin checkout -- see RoboTwinNativePort._load_task_class.
"""

import random
from typing import Any

from envs._base_task import Base_Task
from envs.utils import create_box

from oct_vla.robots.robotwin.backend import encode_pose
from oct_vla.tasks.shelf_restock.oracle.world import install_world_patch, register_objects
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC, ShelfRestockSpec

# The hand is wider than the object it grasps: cuRobo's own collision spheres
# put panda_hand's outermost centres at y=+-0.08 with radius 0.023, so it is
# ~0.10m from the grasp axis to the outside of the hand. Spawning objects
# closer than that means descending onto one shoves its neighbour -- observed
# live as panda_hand <-> restock_object_1 during a descent, which the oracle's
# contact check now rejects outright (docs/architecture.md).
MIN_OBJECT_SEPARATION = 0.15
OBJECT_COLORS = (
    (0.8, 0.2, 0.2),
    (0.2, 0.6, 0.8),
    (0.3, 0.7, 0.3),
    (0.8, 0.6, 0.2),
)


class ShelfRestockTask(Base_Task):
    """RoboTwin task class; spec/object_count are class attributes because
    Base_Task instances are constructed with no arguments (see
    RoboTwinNativePort._load_task_class)."""

    spec: ShelfRestockSpec = DEFAULT_SPEC
    object_count: int = 3

    def setup_demo(self, **kwargs: Any) -> None:
        self._episode_seed = kwargs.get("seed", 0)
        # Must run before _init_task_env_ -> load_robot -> CuroboPlanner.__init__,
        # which is where RoboTwin sizes and loads cuRobo's obstacle cache.
        install_world_patch(self.spec, max_objects=self.object_count)
        super()._init_task_env_(**kwargs)

    def load_actors(self) -> None:
        self._load_upper_shelf()
        self._load_objects()
        register_objects(self, self.tracked_objects)

    def _load_upper_shelf(self) -> None:
        shelf = self.spec.upper_shelf
        self.upper_shelf_actor = create_box(
            scene=self,
            pose=_sapien_pose(shelf.center_xyz),
            half_size=shelf.half_extent_xyz,
            color=(0.82, 0.80, 0.75),
            is_static=True,
            name="upper_shelf",
        )

    def _load_objects(self) -> None:
        spec = self.spec
        rng = random.Random(self._episode_seed)
        z = spec.lower_shelf.top_z + spec.spawn_clearance

        self.tracked_objects: dict[str, Any] = {}
        for index, x in enumerate(self._spaced_x_positions(spec, rng)):
            sampled = spec.object_variation.sample_pose(z, rng)
            # x comes from the spacing construction; y/yaw stay as sampled.
            position = (x, sampled.position[1], sampled.position[2])
            size = spec.object_variation.sample_size(rng)
            actor = create_box(
                scene=self,
                pose=_sapien_pose(position, sampled.orientation),
                half_size=tuple(s / 2 for s in size),
                color=OBJECT_COLORS[index % len(OBJECT_COLORS)],
                name=f"restock_object_{index}",
            )
            self.tracked_objects[f"obj_{index}"] = TrackedObject(actor, size)

    def _spaced_x_positions(self, spec, rng) -> list[float]:
        """x positions guaranteed at least MIN_OBJECT_SEPARATION apart.

        Rejection sampling cannot reliably fit this many objects into the
        spawn strip at this separation -- it failed outright at seed 0 -- so
        the gaps are constructed rather than retried: sample `n` offsets from
        the span left over after reserving every gap, sort them, then add back
        `i * separation`. Separating in x alone is sufficient because it
        already lower-bounds the Euclidean distance whatever y is sampled.
        """
        low, high = spec.object_variation.position_x_range
        reserved = MIN_OBJECT_SEPARATION * (self.object_count - 1)
        free = (high - low) - reserved
        if free < 0:
            raise RuntimeError(
                f"cannot fit {self.object_count} objects {MIN_OBJECT_SEPARATION}m apart "
                f"in an x span of {high - low:.3f}m; widen position_x_range or spawn fewer"
            )
        offsets = sorted(rng.uniform(0.0, free) for _ in range(self.object_count))
        positions = [low + offset + i * MIN_OBJECT_SEPARATION for i, offset in enumerate(offsets)]
        # Decouple spatial order from object index so obj_0 is not always leftmost.
        rng.shuffle(positions)
        return positions

    def play_once(self) -> None:
        raise NotImplementedError("ShelfRestockTask has no oracle yet; see docs/architecture.md")

    def check_success(self) -> bool:
        """True once every spawned object has left the lower shelf.

        Reads actor poses directly rather than through ObjectScene/
        GroundTruthObjectStateEstimator: this is RoboTwin's own internal
        lifecycle hook, not policy-facing, so it has no reason to round-trip
        through the canonical perception layer.
        """
        return all(
            not self.spec.lower_shelf.contains(tuple(entry.actor.get_pose().p))
            for entry in self.tracked_objects.values()
        )


class TrackedObject:
    __slots__ = ("actor", "size_xyz")

    def __init__(self, actor: Any, size_xyz: tuple[float, float, float]) -> None:
        self.actor = actor
        self.size_xyz = size_xyz


def _sapien_pose(position, orientation=(0.0, 0.0, 0.0, 1.0)):
    """Scene geometry is authored directly in world coordinates (workcell ==
    world for this task; see docs/architecture.md)."""
    import sapien

    from oct_vla.core.frames import Pose

    values = encode_pose(Pose(position, orientation, "world"))
    return sapien.Pose(p=values[:3], q=values[3:])
