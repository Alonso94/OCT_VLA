"""The shelf's oracle next-interaction keyframe, one definition for export and rollout.

A function of the current scene and a little episode state, never of the
recording's phase labels, so the keyframe a policy trains on and the one the
evaluation server hands it are computed the same way. It calls the expert's own
planners (``oracle/grasps.py``, ``oracle/placement.py``) with the expert's
defaults: the expert takes ``candidates[0]`` and plans the placement against the
previous neighbour, deterministically, so the oracle's grasp and goal are the
ones the demonstrator executed.

State machine, per step::

    no target, or target placed and let go     -> select the next one
        (the manager's selector and neighbour rule; none left -> empty keyframe)
    target lifted clear of the lower shelf      -> event "place", last plan
    otherwise                                   -> event "grasp", replanned now

The manager alone cannot drive this: once a carried object rises past the lower
shelf's occupancy band, ``next_context`` already selects the *next* object. Nor
is "on the upper shelf" enough to call a transfer done: the object enters the
upper shelf's band while still being lowered, 15-25 frames before release. So
the target counts as placed only once the mover has also let go, read from
distance (aperture cannot: a wide object held closed reads up to 0.83, an open
gripper 0.833). Over the v4 corpus the held object stays 0.138-0.162 m from the
mover's EEF through every carrying phase, and the arm is >= 0.234 m away at the
end of its retreat.

A policy can leave the scene somewhere the expert never would, and then the
expert's planners can fail too (a neighbour-relative placement off the end of
the shelf, a tipped object with no top-down grasp). The oracle has no keyframe
to give then: it returns an empty mask, which the branch turns into exactly
kv, and counts the step in ``unplannable_steps`` so a rollout reports how often
its "oracle" condition was really kv. Export (``episode_keyframes``) runs on the
expert's own recordings, where this cannot happen, and refuses if it does.

Rows follow ``data/future_keyframes.py``: object at the event, the acting
gripper's commanded pose at the event, object at its goal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from oct_vla.core.objects import ObjectScene
from oct_vla.data.entity_tokens import ENTITY_TOKEN_DIM, _token
from oct_vla.core.frames import Pose
from oct_vla.data.future_keyframes import (
    FUTURE_DIM,
    FUTURE_DIM_WP,
    FUTURE_ROWS,
    FUTURE_ROLES,
    FUTURE_ROLES_WP,
    WAYPOINT_REACHED_XY,
    WAYPOINT_REACHED_Z,
    _row,
    approach_waypoint,
)
from oct_vla.tasks.shelf_restock.manager import (
    ShelfRestockManager,
    objects_lifted,
    objects_on_upper_shelf,
)
from oct_vla.tasks.shelf_restock.oracle.arms import arm_for
from oct_vla.tasks.shelf_restock.oracle.grasps import generate_top_down_grasps
from oct_vla.tasks.shelf_restock.oracle.placement import PlacementError, plan_placement
from oct_vla.tasks.shelf_restock.spec import ShelfRestockSpec

#: The expert's defaults (oracle/expert.py), which collection uses unchanged.
STANDOFF = 0.10
GRIPPER_MAX_WIDTH = 0.08
#: Normalised aperture of an open gripper (EEFState.gripper is in [0, 1]).
OPEN = 1.0
#: Mover-to-object distance beyond which a placed object has been let go.
RELEASE_DISTANCE = 0.20
#: A pending neighbour within this of its compacted pose has been compacted.
COMPACTED_REACHED = 0.01
#: Retreat (``retreat=True``): a placed target whose mover is farther than this has
#: been released and the mover is rising (the hold offset is 0.138-0.156 m over the v4
#: corpus). Observable from the scene and EEF; no aperture threshold, because the
#: measured aperture of a held wide object (0.79) is within 0.04 of an open one (0.83).
RETREAT_START = 0.17
#: The retreat is done once the mover is within this of the retreat height, or has
#: left horizontally by more than RETREAT_ABANDON (it went elsewhere without lifting;
#: the next target is then given rather than an unreachable one held forever). The
#: recorded expert lifts 0.093-0.102 m on 23 of 24 releases checked.
RETREAT_REACHED_Z = 0.02
RETREAT_ABANDON = 0.30


class NoGraspError(ValueError):
    """No top-down grasp candidate fits the target."""


@dataclass
class _Plan:
    grasp_pose: object
    closing: float
    object_pose: object
    place_pose: object
    #: Where compaction will push the object (None when no compaction is needed).
    compacted_pose: object = None


class ShelfKeyframeOracle:
    """Call once per step, in order, with that step's scene.

    ``waypoints=True``: the 25-wide layout, with a fourth row holding the mover's
    approach waypoint above the event pose (``future_keyframes.approach_waypoint``),
    masked from the step the mover reaches it until the event or target changes."""

    def __init__(self, spec: ShelfRestockSpec, waypoints: bool = False,
                 retreat: bool = False, anticipate_compaction: bool = False) -> None:
        if retreat and not waypoints:
            raise ValueError("the retreat phase uses the waypoint layout")
        self.spec = spec
        #: Plan the next placement beside the previous object's *compacted* pose while
        #: that object still awaits its push. Without it, the goal planned during the
        #: compaction window sits beside the uncompacted neighbour and jumps ~5 cm once
        #: the push lands (median 5.2 cm vs the recorded placement over the v4 corpus,
        #: against 0.5 cm outside the window); the effect oracle needs the goal the
        #: object will actually be placed at.
        self.anticipate_compaction = anticipate_compaction
        #: Placed objects still awaiting compaction -> the pose the push will leave them.
        self._pending: dict[str, object] = {}
        self.waypoints = waypoints
        self.retreat = retreat
        #: The frozen retreat EEF position while retreating, else None.
        self._retreat_to: np.ndarray | None = None
        self.roles = FUTURE_ROLES_WP if waypoints else FUTURE_ROLES
        self.dim = FUTURE_DIM_WP if waypoints else FUTURE_DIM
        #: (event, target) whose waypoint the mover has reached.
        self._reached: tuple[str, str] | None = None
        self.manager = ShelfRestockManager(spec)
        self.mover = arm_for("place")
        self.target: str | None = None
        self.neighbour: str | None = None
        self.plan: _Plan | None = None
        self._placed: set[str] = set()
        #: (event, target) per call, for diagnostics and tests.
        self.history: list[tuple[str | None, str | None]] = []
        #: Steps on which the expert's planners failed and the keyframe was empty.
        self.unplannable_steps = 0
        self.last_failure = ""

    def _select(self, scene: ObjectScene) -> None:
        context = self.manager.next_context(scene)
        self.target = None if context is None else context.target_track_id
        self.neighbour = None if context is None else context.previous_neighbor_track_id
        self.plan = None

    def _replan(self, scene: ObjectScene) -> None:
        target = scene.get(self.target)
        candidates = generate_top_down_grasps(target, gripper_max_width=GRIPPER_MAX_WIDTH,
                                              standoff=STANDOFF)
        if not candidates:
            raise NoGraspError(f"no top-down grasp fits {self.target!r}")
        grasp = candidates[0]
        neighbour = scene.get(self.neighbour) if self.neighbour is not None else None
        pending = self._pending.get(self.neighbour)
        if neighbour is not None and pending is not None:
            if np.linalg.norm(np.subtract(neighbour.pose.position, pending.position)) \
                    > COMPACTED_REACHED:
                neighbour = replace(neighbour, pose=pending)
            else:
                del self._pending[self.neighbour]
        placement = plan_placement(self.spec, target, grasp.wrist_yaw, neighbour,
                                   standoff=STANDOFF, side=arm_for("compact"))
        self.plan = _Plan(grasp.grasp_pose, min(grasp.closing_width / GRIPPER_MAX_WIDTH, OPEN),
                          placement.object_pose, placement.place_pose,
                          placement.compacted_object_pose)

    def _let_go(self, scene: ObjectScene, eef) -> bool:
        target = scene.get(self.target)
        mover = getattr(eef, self.mover).pose.position
        return float(np.linalg.norm(np.subtract(mover, target.pose.position))) > RELEASE_DISTANCE

    def __call__(self, scene: ObjectScene, eef) -> tuple[np.ndarray, np.ndarray]:
        """``[rows, dim]`` keyframe and ``[rows]`` mask for this step (3 x 24, or 4 x 25
        with waypoints)."""
        upper = objects_on_upper_shelf(scene, self.spec)
        # Mirror the server's record_placement, so the neighbour rule sees the
        # same placement order online and offline.
        for track_id in sorted(upper):
            if track_id not in self._placed:
                self._placed.add(track_id)
                self.manager.record_placement(track_id)
        if self.retreat and self.target is not None and scene.get(self.target) is not None \
                and self.target in upper and self.plan is not None:
            rows = self._retreat_rows(scene, eef)
            if rows is not None:
                return rows
        elif (self.target is None or scene.get(self.target) is None
                or (self.target in upper and self._let_go(scene, eef))):
            if (self.anticipate_compaction and self.target is not None
                    and self.plan is not None and self.plan.compacted_pose is not None):
                self._pending[self.target] = self.plan.compacted_pose
            self._select(scene)
        rows_n = len(self.roles)
        empty = (np.zeros((rows_n, self.dim), np.float32), np.zeros(rows_n, bool))
        if self.target is None:
            self.history.append((None, None))
            return empty
        target = scene.get(self.target)
        # Placed but still held counts as "place" too: the release is the event.
        if self.target in objects_lifted(scene, self.spec) and self.plan is not None:
            event = "place"
        else:
            event = "grasp"
            try:
                self._replan(scene)
            except (PlacementError, NoGraspError) as failure:
                self.plan = None
                self.unplannable_steps += 1
                self.last_failure = str(failure)
                self.history.append((None, self.target))
                return empty
        plan = self.plan
        size = target.size_xyz
        gripper = f"{self.mover}_gripper"
        if event == "grasp":
            obj = _token(target.pose, size, 0.0, "movable")
            pose, approach_aperture = plan.grasp_pose, OPEN
            grip = _token(pose, (0.0, 0.0, 0.0), plan.closing, gripper)
        else:
            obj = _token(plan.object_pose, size, 0.0, "movable")
            pose, approach_aperture = plan.place_pose, plan.closing
            grip = _token(pose, (0.0, 0.0, 0.0), OPEN, gripper)
        goal = _token(plan.object_pose, size, 0.0, "movable")
        rows = [_row(obj, "object_keyframe", event, self.roles),
                _row(grip, "gripper_keyframe", event, self.roles),
                _row(goal, "object_goal", event, self.roles)]
        mask = np.ones(len(self.roles), bool)
        if self.waypoints:
            others = [(o.pose.position, o.size_xyz) for o in scene.objects
                      if o.track_id != self.target]
            above = approach_waypoint(pose.position, others)
            mover = np.asarray(getattr(eef, self.mover).pose.position)
            key = (event, self.target)
            if (np.linalg.norm(mover[:2] - above[:2]) <= WAYPOINT_REACHED_XY
                    and abs(mover[2] - above[2]) <= WAYPOINT_REACHED_Z):
                self._reached = key
            if self._reached == key:
                rows.append(np.zeros(self.dim, np.float32))
                mask[3] = False
            else:
                way = _token(Pose(tuple(above), pose.orientation, pose.frame),
                             (0.0, 0.0, 0.0), approach_aperture, gripper)
                rows.append(_row(way, "gripper_waypoint", event, self.roles))
        self.history.append((event, self.target))
        return np.stack(rows), mask


    def _retreat_rows(self, scene: ObjectScene, eef):
        """Rows of the retreat phase, or None once it is over (the next target is then
        selected and the caller carries on). Before the mover has left the hold offset
        the placed target is still "place"; this returns None without selecting."""
        target = scene.get(self.target)
        mover = np.asarray(getattr(eef, self.mover).pose.position)
        placed_at = np.asarray(target.pose.position)
        if self._retreat_to is None:
            if np.linalg.norm(mover - placed_at) <= RETREAT_START:
                return None  # still holding or just releasing: the place rows
            others = [(o.pose.position, o.size_xyz) for o in scene.objects]
            self._retreat_to = approach_waypoint(self.plan.place_pose.position, others)
        to = self._retreat_to
        if (mover[2] >= to[2] - RETREAT_REACHED_Z
                or np.linalg.norm(mover[:2] - to[:2]) > RETREAT_ABANDON):
            self._retreat_to = None
            self._select(scene)
            return None
        pose = self.plan.place_pose
        gripper = f"{self.mover}_gripper"
        up = _token(Pose(tuple(to), pose.orientation, pose.frame), (0.0, 0.0, 0.0), OPEN, gripper)
        rows = np.stack([
            _row(_token(target.pose, target.size_xyz, 0.0, "movable"), "object_keyframe",
                 "release", self.roles),
            _row(up, "gripper_keyframe", "release", self.roles),
            _row(_token(self.plan.object_pose, target.size_xyz, 0.0, "movable"), "object_goal",
                 "release", self.roles),
            np.zeros(self.dim, np.float32),  # the retreat pose is the gripper row itself
        ])
        mask = np.array([True, True, True, False])
        self.history.append(("release", self.target))
        return rows, mask


def episode_keyframes(episode, spec: ShelfRestockSpec,
                      waypoints: bool = False, retreat: bool = False,
                      anticipate_compaction: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The oracle run over a recorded episode, frame by frame: ``[T, 3, 24]``, ``[T, 3]``
    (``[T, 4, 25]``, ``[T, 4]`` with waypoints)."""
    oracle = ShelfKeyframeOracle(spec, waypoints=waypoints, retreat=retreat,
                                 anticipate_compaction=anticipate_compaction)
    out = [oracle(sample.scene, sample.observation.eef) for sample in episode.samples]
    if oracle.unplannable_steps:
        raise ValueError(f"seed {episode.seed}: the expert's planners failed on "
                         f"{oracle.unplannable_steps} recorded frames ({oracle.last_failure})")
    return np.stack([k for k, _ in out]), np.stack([m for _, m in out])


class ShelfEffectOracle:
    """The online object-effect oracle (contract A of ``data/effects.py``).

    The keyframe oracle with ``anticipate_compaction``, projected to the goal-only effect
    rows the effect-conditioned VLA was trained on: row 0 the target object at its
    *current* pose, row 2 its planned placement, both as yaw-only tokens built exactly
    as the hindsight labels are (``effects.yaw_token``); gripper row masked, event
    empty. It switches targets where the hindsight label does (``RELEASE_DISTANCE``),
    and its goal is the placement beside the neighbour's compacted pose, so offline
    labels and online rows describe the same effect (checked per episode by
    ``scripts/add_effects.py``)."""

    def __init__(self, spec: ShelfRestockSpec) -> None:
        self.inner = ShelfKeyframeOracle(spec, anticipate_compaction=True)

    @property
    def unplannable_steps(self) -> int:
        return self.inner.unplannable_steps

    @property
    def last_failure(self) -> str:
        return self.inner.last_failure

    @property
    def history(self):
        return self.inner.history

    def __call__(self, scene: ObjectScene, eef) -> tuple[np.ndarray, np.ndarray]:
        from oct_vla.data.effects import yaw_token

        _, keyframe_mask = self.inner(scene, eef)
        rows = np.zeros((FUTURE_ROWS, FUTURE_DIM), np.float32)
        mask = np.zeros(FUTURE_ROWS, bool)
        target = None if self.inner.target is None else scene.get(self.inner.target)
        if not keyframe_mask.any() or target is None or self.inner.plan is None:
            return rows, mask
        goal = self.inner.plan.object_pose
        rows[0, :ENTITY_TOKEN_DIM] = yaw_token(target.pose.position, target.pose.orientation,
                                               target.size_xyz)
        rows[0, ENTITY_TOKEN_DIM + 0] = 1.0
        rows[2, :ENTITY_TOKEN_DIM] = yaw_token(goal.position, goal.orientation, target.size_xyz)
        rows[2, ENTITY_TOKEN_DIM + 2] = 1.0
        mask[[0, 2]] = True
        return rows, mask
