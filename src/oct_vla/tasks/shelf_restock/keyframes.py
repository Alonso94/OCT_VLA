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

from dataclasses import dataclass

import numpy as np

from oct_vla.core.objects import ObjectScene
from oct_vla.data.entity_tokens import _token
from oct_vla.core.frames import Pose
from oct_vla.data.future_keyframes import (
    FUTURE_DIM,
    FUTURE_DIM_WP,
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


class NoGraspError(ValueError):
    """No top-down grasp candidate fits the target."""


@dataclass
class _Plan:
    grasp_pose: object
    closing: float
    object_pose: object
    place_pose: object


class ShelfKeyframeOracle:
    """Call once per step, in order, with that step's scene.

    ``waypoints=True``: the 25-wide layout, with a fourth row holding the mover's
    approach waypoint above the event pose (``future_keyframes.approach_waypoint``),
    masked from the step the mover reaches it until the event or target changes."""

    def __init__(self, spec: ShelfRestockSpec, waypoints: bool = False) -> None:
        self.spec = spec
        self.waypoints = waypoints
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
        placement = plan_placement(self.spec, target, grasp.wrist_yaw, neighbour,
                                   standoff=STANDOFF, side=arm_for("compact"))
        self.plan = _Plan(grasp.grasp_pose, min(grasp.closing_width / GRIPPER_MAX_WIDTH, OPEN),
                          placement.object_pose, placement.place_pose)

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
        if (self.target is None or scene.get(self.target) is None
                or (self.target in upper and self._let_go(scene, eef))):
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


def episode_keyframes(episode, spec: ShelfRestockSpec,
                      waypoints: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The oracle run over a recorded episode, frame by frame: ``[T, 3, 24]``, ``[T, 3]``
    (``[T, 4, 25]``, ``[T, 4]`` with waypoints)."""
    oracle = ShelfKeyframeOracle(spec, waypoints=waypoints)
    out = [oracle(sample.scene, sample.observation.eef) for sample in episode.samples]
    if oracle.unplannable_steps:
        raise ValueError(f"seed {episode.seed}: the expert's planners failed on "
                         f"{oracle.unplannable_steps} recorded frames ({oracle.last_failure})")
    return np.stack([k for k, _ in out]), np.stack([m for _, m in out])
