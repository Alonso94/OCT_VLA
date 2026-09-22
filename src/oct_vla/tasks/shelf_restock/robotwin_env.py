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

import os
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
from oct_vla.tasks.shelf_restock.spec import (
    DEFAULT_SPEC,
    MIN_OBJECT_SEPARATION,
    ShelfRestockSpec,
)

#: Head-camera placement, overriding the embodiment's stock pose. RoboTwin's
#: default framing was chosen for tabletop tasks and does not see the upper
#: shelf; this pulls the camera back and up so both shelf levels and the
#: objects on them are in frame, which is what the policy's RGB observation
#: has to contain for the task to be learnable at all.
#:
#: RoboTwin reads the static camera list from `left_embodiment_config` only
#: (envs/camera/camera.py), so overriding that one list is sufficient --
#: writing to the right arm's copy as well would have no effect.
#:
#: Framed against the widest profile (four objects), not the default one. The
#: previous values were chosen when objects spawned within x in [-0.34, 0.0];
#: once the spawn span widened to cover four objects, a rendered frame showed
#: the row pushed onto the bottom edge with the outermost object clipped, so
#: an RGB-only policy had no view of the object it was being asked to restock.
#: Shifting the camera to the scene's own x centre and tilting slightly
#: further down puts every spawned object and the whole deck inside the frame
#: with margin. Checked by rendering a reset frame per profile rather than
#: derived -- see outputs/camera_check.
HEAD_CAMERA_OVERRIDE = {
    "position": [-0.11, -0.95, 1.75],
    "forward": [0, 0.72, -0.70],
    "left": [-1, 0, 0],
}

#: The objects being restocked: real scanned meshes rather than procedural
#: boxes. A box's uniform faces and exact symmetry would make the
#: object-centric conditioning this research evaluates easier than the real
#: task, and the policy should see the kind of object it would at deployment.
#:
#: Their dimensions are a property of the asset (create_actor overwrites its
#: own `scale` argument with the value inside model_data<N>.json), so
#: ObjectVariation's size sampling does not apply to spawning; size varies by
#: which asset and which of its variants is chosen.
#:
#: Several assets rather than one, sampled *per object*. With a single asset
#: every object in every scene was a coffee box, so "which object is this" had
#: no answer beyond its position, and a semantic channel in the object-centric
#: representation carried exactly zero information -- there was nothing for a
#: compositional claim to be about except object count. These six share a grasp
#: affordance, so the oracle's planner should discard at roughly the same rate,
#: while differing in mesh, size and appearance.
#: One asset, whose *variants* are the diversity. `113_coffee-box` ships seven
#: meshes and the collected corpus contains six of them, 18-81 episodes each --
#: visually distinct objects a policy must handle, already recorded.
#:
#: A multi-category set was attempted and abandoned. Probing candidates one at a
#: time showed the reason: of six assets that passed a geometric screen
#: (gripper width, shelf separation, deck clearance), five collected **0 of 8
#: seeds** on their own, almost all `global plan failed` -- the oracle's planner
#: cannot solve a top-down grasp around those meshes. Geometry does not predict
#: plannability, and a candidate must be probed before it is trusted. The coffee
#: box, unchanged, collects at the corpus rate.
OBJECT_MODELS = ("113_coffee-box",)

#: Kept as the single-asset default so a corpus can still be collected the old
#: way, and so every reference to "the" object model resolves to what the
#: existing corpora hold.
OBJECT_MODEL = OBJECT_MODELS[0]


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

    def refresh_planning_world(self, exclude: str | None = None) -> None:
        """Re-register every object with both planners at its current pose.

        Called by RoboTwinNativePort before each plan. Objects move, and a
        planner working from load-time poses believes a restocked object is
        still on the lower shelf while the space it now occupies is free.
        `exclude` omits the track_id the moving arm is engaging with.
        """
        register_objects(self, self.tracked_objects, exclude=exclude)

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
        # An asset and a variant *per object*, where this used to draw one
        # variant for the whole episode. The old comment justified that by
        # saying a mixed-size row "makes the placement/compaction geometry vary
        # per object for reasons the policy cannot see" -- which was right
        # while the policy saw only pixels and a pose. An object-centric
        # representation carries each object's size and category, so the
        # variation is now observable, and making it observable is the point.
        # An environment override so a single category can be probed in
        # isolation. Which assets the oracle's planner can actually solve is an
        # empirical question -- the first mixed set collected 0 of 20 seeds --
        # and a per-category measurement is the only way to tell a bad asset
        # from a bad set.
        override = os.environ.get("OCTVLA_OBJECT_MODELS", "").strip()
        models = (
            tuple(m for m in override.split(",") if m)
            or getattr(self, "object_models", None)
            or OBJECT_MODELS
        )
        # Whether a scene may mix *sizes*. Category and size are separable: a
        # scene can hold a coffee box and a stapler while every instance of an
        # asset uses one variant. The original code drew a single variant per
        # episode and its comment said why -- a mixed-size row changes the
        # placement and compaction geometry per object -- so this stays a knob
        # until a probe says which way the oracle can actually plan.
        per_object_variant = os.environ.get("OCTVLA_VARIANT_PER_OBJECT", "1") != "0"
        # Which mesh variants a scene may draw from. Evaluation needs this:
        # the training split holds out whole identities, so a rollout is only
        # interpretable if it can be pinned to seen, held-out, or
        # never-collected meshes rather than sampling across all three. Empty
        # means every variant the asset ships, which is what collection wants.
        wanted = os.environ.get("OCTVLA_MODEL_IDS", "").strip()
        allowed = {int(v) for v in wanted.split(",") if v} if wanted else None

        def variants_of(model: str) -> list[int]:
            available = list(available_model_ids(model))
            if allowed is None:
                return available
            chosen = [v for v in available if v in allowed]
            if not chosen:
                raise RuntimeError(
                    f"OCTVLA_MODEL_IDS={wanted} selects no variant of {model}; "
                    f"it ships {available}"
                )
            return chosen

        episode_variants = {m: rng.choice(variants_of(m)) for m in models}
        for index, x in enumerate(self._spaced_x_positions(spec, rng)):
            model = rng.choice(models)
            model_id = (
                rng.choice(variants_of(model)) if per_object_variant else episode_variants[model]
            )
            size = upright_size(model, model_id)
            offset = upright_center_offset(model, model_id)
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
                modelname=model,
                convex=True,
                model_id=model_id,
            )
            if actor is None:
                raise RuntimeError(f"RoboTwin could not build {model} model_id={model_id}")
            # create_actor names every instance after the asset, so all three
            # would be '113_coffee-box'. Contact reports and the oracle's
            # allow_contact_with both key on this name, so identical names
            # would let a contact with any object be excused as the held one.
            actor.actor.set_name(f"restock_object_{index}")
            self.tracked_objects[f"obj_{index}"] = TrackedObject(
                actor, size, UPRIGHT_ROTATION, offset, category=model
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


# Count-shift profiles. They deliberately inherit `spec` rather than defining
# their own: the shared spawn span is already sized for the largest of them
# (see DEFAULT_SPEC), so `object_count` is the only thing that differs and a
# count-shift result cannot be confounded by a change of geometry. Never used
# for training data -- ShelfRestockTask itself is the training profile.
class ShelfRestockTwoObjectTask(ShelfRestockTask):
    """Sparse count-shift evaluation profile."""

    object_count = 2


class ShelfRestockFourObjectTask(ShelfRestockTask):
    """Crowded count-shift evaluation profile."""

    object_count = 4


class TrackedObject:
    """A spawned object and the metadata needed to report it canonically.

    `upright_rotation` and `center_offset` are carried alongside the actor
    because the measured mesh pose is neither upright nor centred on the
    object; see perception/robotwin/evidence.TrackedActor, which corrects
    both when reporting a canonical ObjectState.
    """

    __slots__ = ("actor", "size_xyz", "upright_rotation", "center_offset", "category")

    def __init__(
        self,
        actor: Any,
        size_xyz: tuple[float, float, float],
        upright_rotation: tuple[float, float, float, float] = UPRIGHT_ROTATION,
        center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        category: str | None = None,
    ) -> None:
        self.actor = actor
        self.size_xyz = size_xyz
        self.upright_rotation = upright_rotation
        self.center_offset = center_offset
        #: The asset name, reported as the object's category.
        self.category = category


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
