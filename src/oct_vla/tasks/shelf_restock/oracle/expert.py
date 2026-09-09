"""Oracle state machine: one atomic transfer, and the whole episode.

This is the sequence that was, until now, only exercised live from a
throwaway script -- there was no way to generate a demonstration from this
repo at all. It composes the already-verified pieces (`grasps`, `placement`,
`motion`, `arms`) into the fixed nine-phase transfer plus the optional
seven-phase compaction, exactly as verified live; nothing here is a new
design, only the wiring between modules that already existed separately.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from oct_vla.core.frames import Pose, Transform
from oct_vla.core.objects import ObjectScene, TaskContext
from oct_vla.robots.robotwin.backend import NativePort
from oct_vla.tasks.shelf_restock.manager import ShelfRestockManager
from oct_vla.tasks.shelf_restock.oracle.arms import Side, arm_for, is_cross_body_limited
from oct_vla.tasks.shelf_restock.oracle.grasps import backed_off, generate_top_down_grasps
from oct_vla.tasks.shelf_restock.oracle.motion import (
    STRAIGHT_LINE,
    ExecutedMotion,
    move_to,
    set_gripper,
)
from oct_vla.tasks.shelf_restock.oracle.placement import plan_placement, plan_push
from oct_vla.tasks.shelf_restock.spec import ShelfRestockSpec

Observe = Callable[[], ObjectScene]

#: Gripper commands. Named rather than inlined so every phase reads as
#: "open"/"closed" instead of a bare float whose meaning depends on convention.
OPEN = 1.0
CLOSED = 0.0


class ExpertError(RuntimeError):
    """Oracle-level refusal: the plan itself is not viable (missing object,
    no grasp fits, the geometry is unreachable). Distinct from `MotionError`,
    which is a planner/execution failure discovered only by trying to move."""


@dataclass(frozen=True)
class TransferRecord:
    """One atomic transfer: every motion actually commanded, in order."""

    context: TaskContext
    motions: tuple[tuple[str, ExecutedMotion], ...]
    compacted: bool


class ShelfRestockExpert:
    def __init__(
        self,
        spec: ShelfRestockSpec,
        port: NativePort,
        observe: Observe,
        world_to_workcell: Transform,
        body_names: Mapping[str, str],
        home_poses: Mapping[Side, Pose],
        *,
        standoff: float = 0.10,
        gripper_max_width: float = 0.08,
        push_slowdown: int = 5,
    ) -> None:
        """`body_names` maps track_id -> SAPIEN actor body name. Two different
        identifiers name the same object: `allow_contact_with` matches contact
        reports, which carry body names, while the planner world (and this
        oracle's own context) is keyed by track_id. Both are needed because
        neither can be derived from the other without this table.

        `home_poses` is each arm's measured rest pose, captured by the caller
        at reset, rather than a hand-chosen constant: a pose measured live is
        known-reachable by construction, so parking there can never itself
        fail to plan. A hand-chosen "home" carries no such guarantee.
        """
        self._spec = spec
        self._port = port
        self._observe = observe
        self._world_to_workcell = world_to_workcell
        self._body_names = body_names
        self._home_poses = home_poses
        self._standoff = standoff
        self._gripper_max_width = gripper_max_width
        self._push_slowdown = push_slowdown

    def _body_name(self, track_id: str) -> str:
        try:
            return self._body_names[track_id]
        except KeyError:
            raise ExpertError(
                f"no body name registered for track_id {track_id!r}; "
                "allow_contact_with cannot be built without it"
            ) from None

    def _move(self, side: Side, target: Pose, **kwargs) -> ExecutedMotion:
        return move_to(self._port, side, target, self._world_to_workcell, **kwargs)

    def _set_gripper(self, side: Side, value: float) -> ExecutedMotion:
        return set_gripper(self._port, side, value)

    def transfer(self, context: TaskContext) -> TransferRecord:
        scene = self._observe()
        target = scene.get(context.target_track_id)
        if target is None:
            raise ExpertError(f"target {context.target_track_id!r} not found in the observed scene")

        mover = arm_for("place")
        compactor = arm_for("compact")

        candidates = generate_top_down_grasps(
            target, gripper_max_width=self._gripper_max_width, standoff=self._standoff
        )
        if not candidates:
            raise ExpertError(
                f"object {target.track_id!r} is too wide for the gripper: "
                f"size_xyz={tuple(round(v, 4) for v in target.size_xyz)}, "
                f"gripper_max_width={self._gripper_max_width}"
            )
        candidate = candidates[0]

        neighbour = (
            scene.get(context.previous_neighbor_track_id)
            if context.previous_neighbor_track_id is not None
            else None
        )

        placement = plan_placement(
            self._spec,
            target,
            candidate.wrist_yaw,
            neighbour,
            standoff=self._standoff,
            side=compactor,
        )
        if is_cross_body_limited(mover, placement.object_pose.position):
            raise ExpertError(
                f"{mover} arm cannot place {target.track_id!r} at "
                f"{tuple(round(v, 3) for v in placement.object_pose.position)}: "
                "measured cross-body limit (see oracle/arms.py)"
            )

        body = self._body_name(target.track_id)
        track_id = target.track_id
        motions: list[tuple[str, ExecutedMotion]] = []

        def carry(pose: Pose) -> ExecutedMotion:
            # Every carrying/engaging phase shares the same allow/ignore pair:
            # the arm is either descending onto the target or holding it, so it
            # must be excused from planning against it and from the contact it
            # is deliberately making with it.
            return self._move(
                mover, pose, gripper=CLOSED, allow_contact_with=(body,), ignore_object=track_id
            )

        motions.append(("open", self._set_gripper(mover, OPEN)))
        motions.append(("pregrasp", self._move(mover, candidate.pregrasp_pose, gripper=OPEN)))
        motions.append(
            (
                "descend",
                self._move(
                    mover,
                    candidate.grasp_pose,
                    gripper=OPEN,
                    allow_contact_with=(body,),
                    ignore_object=track_id,
                ),
            )
        )
        motions.append(("close", self._set_gripper(mover, CLOSED)))
        motions.append(("lift", carry(candidate.pregrasp_pose)))
        motions.append(("preplace", carry(placement.preplace_pose)))
        motions.append(("place", carry(placement.place_pose)))
        motions.append(("release", self._set_gripper(mover, OPEN)))
        motions.append(
            (
                "retreat",
                self._move(
                    mover,
                    backed_off(placement.place_pose, self._standoff),
                    gripper=OPEN,
                    allow_contact_with=(body,),
                    ignore_object=track_id,
                ),
            )
        )

        if not placement.needs_compaction:
            return TransferRecord(context, tuple(motions), compacted=False)

        # Re-observe before planning the push, rather than reusing the
        # placement's predicted pose: the object lands a little off where it
        # was placed, so a push aimed at the *intended* pose misses. The push
        # must be computed from where the object actually ended up.
        placed = self._observe().get(track_id)
        if placed is None:
            raise ExpertError(f"placed object {track_id!r} not found in the re-observed scene")

        compacted_pose = placement.compacted_object_pose
        assert compacted_pose is not None  # guaranteed by needs_compaction above
        target_x = compacted_pose.position[0]
        if is_cross_body_limited(compactor, compacted_pose.position):
            raise ExpertError(
                f"{compactor} arm cannot compact {track_id!r} to x={target_x:.3f}: "
                "measured cross-body limit (see oracle/arms.py)"
            )
        push = plan_push(placed, target_x, standoff=self._standoff)

        # Neither arm is in the planner's own collision world (NativePort),
        # so sequencing is the only thing that keeps them apart. Compaction
        # sends the compacting arm exactly where the placing arm just was, so
        # the mover must be fully parked -- out of the compactor's path --
        # before the compactor moves in, and the mover must not move again
        # until the compactor is parked clear at the end.
        motions.append(("park mover", self._move(mover, self._home_poses[mover], gripper=OPEN)))
        motions.append(("compact close", self._set_gripper(compactor, CLOSED)))
        motions.append(
            (
                "compact approach",
                self._move(compactor, push.approach_pose, gripper=CLOSED, ignore_object=track_id),
            )
        )
        motions.append(
            (
                "compact contact",
                self._move(
                    compactor,
                    push.start_pose,
                    gripper=CLOSED,
                    allow_contact_with=(body,),
                    ignore_object=track_id,
                    constraint=STRAIGHT_LINE,
                ),
            )
        )
        motions.append(
            (
                "compact push",
                self._move(
                    compactor,
                    push.end_pose,
                    gripper=CLOSED,
                    allow_contact_with=(body,),
                    ignore_object=track_id,
                    constraint=push.constraint,
                    slowdown=self._push_slowdown,
                ),
            )
        )
        motions.append(
            (
                "compact retreat",
                self._move(
                    compactor,
                    push.retreat_pose,
                    gripper=CLOSED,
                    allow_contact_with=(body,),
                    ignore_object=track_id,
                ),
            )
        )
        motions.append(
            (
                "park compactor",
                self._move(compactor, self._home_poses[compactor], gripper=CLOSED),
            )
        )

        return TransferRecord(context, tuple(motions), compacted=True)

    def run(self, manager: ShelfRestockManager) -> tuple[TransferRecord, ...]:
        """Drive `manager` to completion. Lets `MotionError`/`ExpertError`
        propagate rather than catching them: a partial episode is a failed
        demonstration, not a shorter successful one, so it must not be
        silently returned as if it were complete."""
        records: list[TransferRecord] = []
        while True:
            scene = self._observe()
            context = manager.next_context(scene)
            if context is None:
                break
            record = self.transfer(context)
            manager.record_placement(context.target_track_id)
            records.append(record)
        return tuple(records)
