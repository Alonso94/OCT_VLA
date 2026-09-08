"""cuRobo world-model registration for the shelf-restock scene.

RoboTwin sizes and loads cuRobo's collision-obstacle cache exactly once, in
``CuroboPlanner.__init__`` (``envs/robot/planner.py``), *before* any
task-specific scene exists -- confirmed live: with no changes here, cuRobo's
global planner has zero knowledge of this task's shelf or spawned objects,
so a "successful" plan is not evidence of physical collision safety (see
docs/architecture.md). ``MotionGen.update_world`` can only ever replace
obstacles up to the cache size fixed at construction time (a hard cuRobo
limit when running with CUDA graphs, which RoboTwin does by default), so that
size must be reserved *before* ``CuroboPlanner.__init__`` runs.

This patches ``MotionGenConfig.load_from_robot_config`` in this process only
-- nothing on disk, inside the RoboTwin checkout or otherwise, is modified.
"""

from typing import Any

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


def install_world_patch(spec: ShelfRestockSpec, *, max_objects: int, margin: int = 2) -> None:
    """Reserve cuRobo collision-cache headroom and inject the upper shelf.

    Idempotent (safe to call once per task instance, or across episodes in
    one process): re-patching after the first call is a no-op.
    """
    from curobo.wrap.reacher.motion_gen import MotionGenConfig

    if getattr(MotionGenConfig.load_from_robot_config, _PATCHED_ATTR, False):
        return

    original = MotionGenConfig.load_from_robot_config
    # table (RoboTwin default) + upper_shelf (this patch) + spawned objects.
    capacity = 2 + max_objects + margin
    static_cuboids = shelf_cuboids(spec)

    def patched(robot_cfg: Any, world_model: Any = None, *args: Any, **kwargs: Any) -> Any:
        if isinstance(world_model, dict):
            cuboids = dict(world_model.get("cuboid", {}))
            cuboids.update(static_cuboids)
            world_model = {**world_model, "cuboid": cuboids}
        kwargs.setdefault("collision_cache", {"obb": capacity})
        return original(robot_cfg, world_model, *args, **kwargs)

    setattr(patched, _PATCHED_ATTR, True)
    MotionGenConfig.load_from_robot_config = staticmethod(patched)


def register_objects(task: Any, tracked_objects: dict[str, Any]) -> None:
    """Push the real spawned-object geometry into both arms' planners.

    Call once, after ``load_actors`` -- object poses/sizes are read once
    here, not tracked live; a moved object needs a fresh call to stay
    accurately represented for planning.
    """
    from curobo.geom.types import WorldConfig

    cuboids = dict(shelf_cuboids(task.spec))
    for track_id, entry in tracked_objects.items():
        pose = entry.actor.get_pose()
        cuboids[track_id] = {"dims": list(entry.size_xyz), "pose": [*pose.p, *pose.q]}

    world = WorldConfig.from_dict({"cuboid": cuboids})
    for planner in (task.robot.left_planner, task.robot.right_planner):
        for motion_gen in (planner.motion_gen, planner.motion_gen_batch):
            motion_gen.update_world(world)
