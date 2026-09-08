"""Thin RoboTwin task lifecycle adapter for the shelf-restock scene.

Builds exactly the scene described by tasks/shelf_restock/spec.py: a static
upper-shelf deck plus randomly placed coffee-box objects on the lower shelf
(the RoboTwin table itself, reused as the lower shelf's deck rather than a
second static box -- there is no shelf mesh asset anywhere in RoboTwin, so
the decks are procedural boxes, the only geometry RoboTwin offers for them).
The restocked objects are real scanned meshes. No oracle lives here:
play_once is not implemented until a later commit can actually execute and
verify an episode against this scene.

Loaded as an external task entrypoint (module:ClassName), not copied into
the RoboTwin checkout -- see RoboTwinNativePort._load_task_class.
"""

import random
from typing import Any

from envs._base_task import Base_Task
from envs.utils import create_actor, create_box

from oct_vla.core.geometry import exp, log, multiply, rotate
from oct_vla.robots.robotwin.assets import (
    UPRIGHT_ROTATION,
    available_model_ids,
    centered_upright_pose,
    upright_center_offset,
    upright_size,
)
from oct_vla.robots.robotwin.backend import decode_pose, encode_pose
from oct_vla.tasks.shelf_restock.oracle.world import install_world_patch, register_objects
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC, ShelfRestockSpec

# The hand is wider than the object it grasps: cuRobo's own collision spheres
# put panda_hand's outermost centres at y=+-0.08 with radius 0.023, so it is
# ~0.10m from the grasp axis to the outside of the hand. Spawning objects
# closer than that means descending onto one shoves its neighbour -- observed
# live as panda_hand <-> restock_object_1 during a descent, which the oracle's
# contact check now rejects outright (docs/architecture.md).
MIN_OBJECT_SEPARATION = 0.15
#: Head-camera placement, overriding the embodiment's stock pose. RoboTwin's
#: default framing was chosen for tabletop tasks and does not see the upper
#: shelf; this pulls the camera back and up so both shelf levels and the
#: objects on them are in frame, which is what the policy's RGB observation
#: has to contain for the task to be learnable at all.
#:
#: RoboTwin reads the static camera list from `left_embodiment_config` only
#: (envs/camera/camera.py), so overriding that one list is sufficient --
#: writing to the right arm's copy as well would have no effect.
HEAD_CAMERA_OVERRIDE = {
    "position": [-0.032, -0.9, 1.7],
    "forward": [0, 0.75, -0.66],
    "left": [-1, 0, 0],
}

#: The object being restocked: a real scanned mesh rather than a procedural
#: box. A box's uniform faces and exact symmetry would make the
#: object-centric conditioning this research evaluates easier than the real
#: task, and the policy should see the kind of object it would at deployment.
#:
#: Its dimensions are a property of the asset (create_actor overwrites its own
#: `scale` argument with the value inside model_data<N>.json), so
#: ObjectVariation's size sampling no longer applies to spawning; the
#: per-episode size variation is now which of the asset's variants is chosen.
OBJECT_MODEL = "113_coffee-box"


class ShelfRestockTask(Base_Task):
    """RoboTwin task class; spec/object_count are class attributes because
    Base_Task instances are constructed with no arguments (see
    RoboTwinNativePort._load_task_class)."""

    spec: ShelfRestockSpec = DEFAULT_SPEC
    object_count: int = 3

    def setup_demo(self, **kwargs: Any) -> None:
        self._episode_seed = kwargs.get("seed", 0)
        _override_head_camera(kwargs)
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

        self.tracked_objects: dict[str, Any] = {}
        for index, x in enumerate(self._spaced_x_positions(spec, rng)):
            model_id = rng.choice(available_model_ids(OBJECT_MODEL))
            size = upright_size(OBJECT_MODEL, model_id)
            offset = upright_center_offset(OBJECT_MODEL, model_id)
            sampled = spec.object_variation.sample_pose(0.0, rng)
            yaw = _yaw_of(sampled.orientation)
            orientation = exp((0.0, 0.0, yaw))
            # Where the object's *centre* should end up: resting on the deck.
            # The asset's own height decides that, so the spec's sampled size
            # no longer applies.
            center = (
                x,
                sampled.position[1],
                spec.lower_shelf.top_z + spec.spawn_clearance + size[2] / 2.0,
            )
            # The actor is posed by its mesh origin, which sits at the
            # object's base, so spawn it offset by however far the centre is
            # from that origin -- the inverse of what the evidence source adds
            # back when reporting the pose.
            rotated = rotate(orientation, offset)
            position = tuple(c - r for c, r in zip(center, rotated, strict=True))

            actor = create_actor(
                scene=self,
                pose=_sapien_pose(position, multiply(orientation, UPRIGHT_ROTATION)),
                modelname=OBJECT_MODEL,
                convex=True,
                model_id=model_id,
            )
            if actor is None:
                raise RuntimeError(f"RoboTwin could not build {OBJECT_MODEL} model_id={model_id}")
            # create_actor names every instance after the asset, so all three
            # would be '113_coffee-box'. Contact reports and the oracle's
            # allow_contact_with both key on this name, so identical names
            # would let a contact with any object be excused as the held one.
            actor.actor.set_name(f"restock_object_{index}")
            self.tracked_objects[f"obj_{index}"] = TrackedObject(
                actor, size, UPRIGHT_ROTATION, offset
            )

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
        through the canonical perception layer. It still applies the same
        mesh-to-object correction, because `ShelfRegion.contains` tests an
        occupancy band above the deck and the raw mesh origin sits at the
        object's base, right at the deck surface -- close enough to the band
        edge to decide this either way for the wrong reason.
        """
        return all(
            not self.spec.lower_shelf.contains(self._object_center(entry))
            for entry in self.tracked_objects.values()
        )

    @staticmethod
    def _object_center(entry: "TrackedObject") -> tuple[float, float, float]:
        pose = entry.actor.get_pose()
        measured = decode_pose((*pose.p, *pose.q))
        position, _ = centered_upright_pose(
            measured.position, measured.orientation, entry.upright_rotation, entry.center_offset
        )
        return position


class TrackedObject:
    """A spawned object and the metadata needed to report it canonically.

    `upright_rotation` and `center_offset` are carried alongside the actor
    because the measured mesh pose is neither upright nor centred on the
    object; see perception/robotwin/evidence.TrackedActor, which corrects
    both when reporting a canonical ObjectState.
    """

    __slots__ = ("actor", "size_xyz", "upright_rotation", "center_offset")

    def __init__(
        self,
        actor: Any,
        size_xyz: tuple[float, float, float],
        upright_rotation: tuple[float, float, float, float] = UPRIGHT_ROTATION,
        center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.actor = actor
        self.size_xyz = size_xyz
        self.upright_rotation = upright_rotation
        self.center_offset = center_offset


def _yaw_of(orientation: tuple[float, float, float, float]) -> float:
    """The z-rotation of a yaw-only orientation, as ObjectVariation samples."""
    return log(orientation)[2]


def _override_head_camera(kwargs: dict[str, Any]) -> None:
    """Point the head camera at the shelves, in place, before cameras load.

    Raises rather than passing silently: a missing head_camera entry would
    leave the stock tabletop framing, and the first sign of that would be a
    whole dataset recorded without the upper shelf in view.
    """
    try:
        cameras = kwargs["left_embodiment_config"]["static_camera_list"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "setup kwargs have no left_embodiment_config.static_camera_list to override"
        ) from error

    for camera in cameras:
        if camera.get("name") == "head_camera":
            camera.update(HEAD_CAMERA_OVERRIDE)
            return
    found = sorted(str(camera.get("name")) for camera in cameras)
    raise ValueError(f"No head_camera in the embodiment's static_camera_list; found {found}")


def _sapien_pose(position, orientation=(0.0, 0.0, 0.0, 1.0)):
    """Scene geometry is authored directly in world coordinates (workcell ==
    world for this task; see docs/architecture.md)."""
    import sapien

    from oct_vla.core.frames import Pose

    values = encode_pose(Pose(position, orientation, "world"))
    return sapien.Pose(p=values[:3], q=values[3:])
