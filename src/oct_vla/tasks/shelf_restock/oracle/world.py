"""cuRobo world-model registration for the shelf-restock scene.

RoboTwin sizes and loads cuRobo's collision-obstacle cache exactly once, in
``CuroboPlanner.__init__`` (``envs/robot/planner.py``), *before* any
task-specific scene exists -- confirmed live: with no changes here, cuRobo's
global planner has zero knowledge of this task's shelf or spawned objects,
so a "successful" plan is not evidence of physical collision safety (see
docs/architecture.md (at tag stageA-2026-09-23)). ``MotionGen.update_world`` can only ever replace
obstacles up to the cache size fixed at construction time (a hard cuRobo
limit when running with CUDA graphs, which RoboTwin does by default), so that
size must be reserved *before* ``CuroboPlanner.__init__`` runs.

This patches ``MotionGenConfig.load_from_robot_config`` in this process only
-- nothing on disk, inside the RoboTwin checkout or otherwise, is modified.
"""

from math import sqrt
from typing import Any

from oct_vla.core.frames import Pose, Transform
from oct_vla.robots.robotwin.assets import centered_upright_pose
from oct_vla.robots.robotwin.backend import decode_pose, encode_pose
from oct_vla.tasks.shelf_restock.spec import ShelfRestockSpec

_PATCHED_ATTR = "_shelf_restock_world_patch"


def shelf_cuboids(spec: ShelfRestockSpec) -> dict[str, dict[str, Any]]:
    """The part of the scene known before any object is spawned.

    The lower shelf is RoboTwin's own table, already registered as its
    default "table" obstacle; only the upper shelf is new geometry.
    """
    shelf = spec.upper_shelf
    return {
        "upper_shelf": {
            "dims": [2.0 * h for h in shelf.half_extent_xyz],
            "pose": [*shelf.center_xyz, 1.0, 0.0, 0.0, 0.0],
        }
    }


def planner_cuboids(
    cuboids: dict[str, dict[str, Any]], origin_pose_wxyz: tuple[float, ...]
) -> dict[str, dict[str, Any]]:
    """Convert world-frame cuboids into one cuRobo planner's own base frame.

    cuRobo's world is **not** in world coordinates. Each RoboTwin planner
    expresses obstacles relative to its robot base: its own table entry is
    posed at ``0.74 - robot_origion_pose.p[2]``, and ``plan_path`` runs every
    target through ``_trans_from_world_to_base`` before planning. Registering
    world-frame poses therefore put every obstacle -- shelf and objects alike
    -- somewhere the planner's arm was never going to be, which is why plans
    that reported ``Success`` still drove through the shelf and swept placed
    objects off it.

    This reproduces RoboTwin's own conversion exactly (``wRb.T @ (p - base)``).
    It deliberately does not apply the planner's ``frame_bias``, which
    ``plan_path`` adds to targets: it is ``[0, 0, 0]`` for franka-panda, and
    guessing at how a nonzero bias should apply to obstacles would be
    inventing a convention rather than matching one.
    """
    px, py, pz, *quaternion = origin_pose_wxyz
    norm = sqrt(sum(float(value) ** 2 for value in quaternion))
    if norm == 0.0:
        raise ValueError("Robot-base quaternion must be nonzero")
    origin = decode_pose((px, py, pz, *(float(value) / norm for value in quaternion)))
    world_to_base = Transform("robot_base", "world", origin.position, origin.orientation).inverse()
    converted = {}
    for name, cuboid in cuboids.items():
        pose = world_to_base.apply_pose(decode_pose(tuple(cuboid["pose"])))
        x, y, z, w = pose.orientation
        converted[name] = {**cuboid, "pose": [*pose.position, w, x, y, z]}
    return converted


def install_world_patch(spec: ShelfRestockSpec, *, max_objects: int, margin: int = 2) -> None:
    """Reserve cuRobo collision-cache headroom for this task's geometry.

    Idempotent (safe to call once per task instance, or across episodes in
    one process): re-patching after the first call is a no-op.
    """
    from curobo.wrap.reacher.motion_gen import MotionGenConfig

    if getattr(MotionGenConfig.load_from_robot_config, _PATCHED_ATTR, False):
        return

    original = MotionGenConfig.load_from_robot_config
    # table (RoboTwin default) + upper_shelf + spawned objects.
    capacity = 2 + max_objects + margin

    def patched(robot_cfg: Any, world_model: Any = None, *args: Any, **kwargs: Any) -> Any:
        # Only capacity is reserved here. The shelf is *not* injected at this
        # point: this hook has no access to the planner whose base frame the
        # world is expressed in, so anything added here would be in the wrong
        # frame. register_objects supplies the real geometry per planner.
        kwargs.setdefault("collision_cache", {"obb": capacity})
        return original(robot_cfg, world_model, *args, **kwargs)

    setattr(patched, _PATCHED_ATTR, True)
    MotionGenConfig.load_from_robot_config = staticmethod(patched)


def register_objects(
    task: Any, tracked_objects: dict[str, Any], *, exclude: str | None = None
) -> None:
    """Push the real spawned-object geometry into both arms' planners.

    Call after ``load_actors`` and again before every plan: object poses are
    read live each time, so an object that has been moved is represented
    where it now is. Registering once at load time left the planner believing
    a restocked object was still on the lower shelf and the space it now
    occupied was empty.

    ``update_world`` *replaces* the whole world rather than merging into it,
    so RoboTwin's own table obstacle has to be re-supplied here or it is
    silently dropped along with everything else.

    ``exclude`` drops one track_id from the world: the object currently in
    the gripper. A carried object travels with the arm, so leaving it in
    places a static obstacle exactly where the hand already is and every plan
    from that configuration starts in collision -- observed as a bare ``Fail``
    from cuRobo when placing. Excluding it is the narrow fix; it means the
    planner does not account for the held object's volume either, so a
    carried object can still clip scene geometry (cuRobo's
    ``attach_objects_to_robot`` is the fuller answer, not done here).

    Poses go through the same mesh-to-object correction the perception layer
    uses, because `size_xyz` describes the object's upright bounding box
    while the actor's own pose describes the mesh: a scanned asset's origin
    sits at its base and its frame is tilted, so using the raw pose would put
    cuRobo's obstacle half an object low and rotated away from where the
    oracle believes the object is.
    """
    from curobo.geom.types import WorldConfig

    table_pose = task.table.get_pose()
    cuboids = {
        # Base_Task.create_table_and_wall fixes these dimensions
        # (create_table(length=1.2, width=0.7, thickness=0.05)), and
        # create_table centres the tabletop slab half a thickness below the
        # actor origin, so the origin is the working surface.
        "table": {
            "dims": [1.2, 0.7, 0.05],
            "pose": [
                float(table_pose.p[0]),
                float(table_pose.p[1]),
                float(table_pose.p[2]) - 0.025,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
        },
        **shelf_cuboids(task.spec),
    }
    for track_id, entry in tracked_objects.items():
        if track_id == exclude:
            continue
        pose = entry.actor.get_pose()
        measured = decode_pose((*pose.p, *pose.q))
        position, orientation = centered_upright_pose(
            measured.position,
            measured.orientation,
            entry.upright_rotation,
            entry.center_offset,
        )
        cuboids[track_id] = {
            "dims": list(entry.size_xyz),
            "pose": list(encode_pose(Pose(position, orientation, measured.frame))),
        }

    # Each planner gets the same geometry expressed in its own base frame.
    for planner in (task.robot.left_planner, task.robot.right_planner):
        origin = planner.robot_origion_pose
        world = WorldConfig.from_dict({"cuboid": planner_cuboids(cuboids, (*origin.p, *origin.q))})
        for motion_gen in (planner.motion_gen, planner.motion_gen_batch):
            motion_gen.update_world(world)
