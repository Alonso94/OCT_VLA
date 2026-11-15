"""The paired script oracle: each arm's next via-points, following the expert's own script.

The conditional-source flow (plan rev. 8) starts GR00T's action flow from a coarse chunk
built from targets. The skip-push keyframe oracle (``keyframes.py``) names one event for
one arm and skips the compaction, so while the expert parks its mover and pushes with
the other arm, that oracle already says "grasp the next object" (gate 1 failed on
exactly those frames: 40 % of grasp-approach frames had the expert's mover waiting).

This oracle instead follows the expert's script (``oracle/expert.py``) phase by phase,
for **both** arms, from observation only (the scene and both EEF poses; never phase
labels), and outputs per arm up to ``VIA`` via-points: pose plus the binary gripper
command applied on arriving there. Every pose comes from the expert's own planners
(``generate_top_down_grasps``, ``plan_placement``, ``backed_off``, ``plan_push``) and
its home poses are the arms' measured rest poses at the first call, as the expert's are.

Stages of one transfer (``STAGES``), with each arm's via-points:

- ``grasp``: mover pregrasp (open), grasp (closed on arrival), then the lookahead below.
  Replanned every step.
- ``carry`` (the target is lifted): mover lift (= pregrasp), then the lookahead. The plan
  is frozen at the grasp.
- lookahead (the rest of the transfer, so a chunk crossing a stage boundary is covered):
  preplace, place (opens on arrival), back off by the standoff.
- ``release`` (resting on the shelf, mover still at the hold offset): place, back off.
- ``retreat`` (the mover has left the hold offset): mover backs off by the standoff.
- ``park_mover`` (compaction pending): mover to its home pose.
- ``compact``: compactor closes in place, then approach, contact, push end, retreat, home,
  gripper closed throughout (planned once, against the re-observed object, as the
  expert does). The mover holds at home.

A via-point counts as reached within ``REACH``; reached via-points are dropped (latched)
until the stage changes, and so is one the arm passed short of REACH once it reaches the
next (a push that its load stops before the planned end). An idle arm has no via-points (it holds).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oct_vla.core.frames import Pose
from oct_vla.core.objects import ObjectScene
from oct_vla.tasks.shelf_restock.keyframes import (
    GRIPPER_MAX_WIDTH,
    RETREAT_START,
    STANDOFF,
    NoGraspError,
)
from oct_vla.tasks.shelf_restock.manager import (
    ShelfRestockManager,
    objects_lifted,
    objects_on_upper_shelf,
)
from oct_vla.tasks.shelf_restock.oracle.arms import arm_for
from oct_vla.tasks.shelf_restock.oracle.grasps import backed_off, generate_top_down_grasps
from oct_vla.tasks.shelf_restock.oracle.placement import PlacementError, plan_placement, plan_push
from oct_vla.tasks.shelf_restock.spec import ShelfRestockSpec

SIDES = ("left", "right")
#: Via-points per arm: the compaction has six.
VIA = 6
#: Columns of a via-point: position, xyzw quaternion, gripper command (1 open, 0 closed).
VIA_DIM = 8
REACH = 0.02
#: An object rests at its place once its centre is this close to the planned height.
REST_TOL = 0.01
#: A pending compaction is done once the object is this close to its compacted x.
COMPACT_TOL = 0.01
STAGES = ("none", "grasp", "carry", "release", "retreat", "park_mover", "compact")
OPEN, CLOSED = 1.0, 0.0


@dataclass
class _Transfer:
    target: str
    grasp_pose: Pose | None = None
    pregrasp_pose: Pose | None = None
    preplace_pose: Pose | None = None
    place_pose: Pose | None = None
    compact_x: float | None = None
    rest_z: float | None = None  # the placed object's planned centre height
    push: list | None = None  # [(pose, gripper)] of the compactor, frozen when planned


def _via(pose: Pose, grip: float) -> np.ndarray:
    return np.array([*pose.position, *pose.orientation, grip], np.float32)


class ShelfArmTargetOracle:
    """Call once per step, in order, with that step's scene and EEF state."""

    def __init__(self, spec: ShelfRestockSpec) -> None:
        self.spec = spec
        self.manager = ShelfRestockManager(spec)
        self.mover, self.compactor = arm_for("place"), arm_for("compact")
        self.home: dict[str, Pose] | None = None
        self.transfer: _Transfer | None = None
        self.stage = "none"
        self._placed: set[str] = set()
        self._dropped: dict[str, int] = {"left": 0, "right": 0}
        self._stage_key: tuple | None = None
        self.history: list[tuple[str, str | None]] = []
        self.unplannable_steps = 0
        self.last_failure = ""

    # ------------------------------------------------------------------ helpers
    def _select(self, scene: ObjectScene) -> None:
        context = self.manager.next_context(scene)
        self.transfer = None if context is None else _Transfer(context.target_track_id)
        self._neighbour = None if context is None else context.previous_neighbor_track_id

    def _plan_grasp(self, scene: ObjectScene) -> None:
        t = self.transfer
        target = scene.get(t.target)
        candidates = generate_top_down_grasps(target, gripper_max_width=GRIPPER_MAX_WIDTH,
                                              standoff=STANDOFF)
        if not candidates:
            raise NoGraspError(f"no top-down grasp fits {t.target!r}")
        c = candidates[0]
        neighbour = scene.get(self._neighbour) if self._neighbour is not None else None
        placement = plan_placement(self.spec, target, c.wrist_yaw, neighbour, standoff=STANDOFF,
                                   side=self.compactor)
        t.grasp_pose, t.pregrasp_pose = c.grasp_pose, c.pregrasp_pose
        t.preplace_pose, t.place_pose = placement.preplace_pose, placement.place_pose
        t.rest_z = placement.object_pose.position[2]
        t.compact_x = (placement.compacted_object_pose.position[0]
                       if placement.needs_compaction else None)

    def _eef(self, eef, side) -> np.ndarray:
        return np.asarray(getattr(eef, side).pose.position, np.float64)

    def _compaction_pending(self, scene) -> bool:
        t = self.transfer
        if t.compact_x is None:
            return False
        obj = scene.get(t.target)
        return t.push is not None or abs(obj.pose.position[0] - t.compact_x) > COMPACT_TOL

    # --------------------------------------------------------------------- step
    def __call__(self, scene: ObjectScene, eef) -> tuple[np.ndarray, np.ndarray]:
        """``[2, VIA, VIA_DIM]`` via-points (left, right) and their ``[2, VIA]`` mask."""
        if self.home is None:
            self.home = {s: getattr(eef, s).pose for s in SIDES}
        upper = objects_on_upper_shelf(scene, self.spec)
        for track_id in sorted(upper):
            if track_id not in self._placed:
                self._placed.add(track_id)
                self.manager.record_placement(track_id)
        try:
            vias = self._stage_vias(scene, eef, upper)
        except (PlacementError, NoGraspError) as failure:
            self.unplannable_steps += 1
            self.last_failure = str(failure)
            self.history.append(("none", self.transfer.target if self.transfer else None))
            return np.zeros((2, VIA, VIA_DIM), np.float32), np.zeros((2, VIA), bool)
        key = (self.stage, self.transfer.target if self.transfer else None)
        if key != self._stage_key:
            self._stage_key, self._dropped = key, {"left": 0, "right": 0}
        out = np.zeros((2, VIA, VIA_DIM), np.float32)
        mask = np.zeros((2, VIA), bool)
        for a, side in enumerate(SIDES):
            points = vias.get(side, [])
            # Drop (latch) reached via-points from the front, never the last one of a
            # stage whose exit is decided by the scene, so the arm keeps a target.
            pos = self._eef(eef, side)
            near = lambda k: np.linalg.norm(pos - points[k][:3]) <= REACH
            switches = lambda k: k > 0 and points[k][7] != points[k - 1][7]
            while self._dropped[side] < len(points) - 1:
                k = self._dropped[side]
                if near(k):
                    self._dropped[side] = k + 1
                elif near(k + 1) and not switches(k):
                    # Passed k short of REACH (a push stopped by its load). Never skip a
                    # gripper switch: the lift sits where the pregrasp was, so "at the
                    # pregrasp" would otherwise read as "past the grasp".
                    self._dropped[side] = min(k + 2, len(points) - 1)
                else:
                    break
            left = points[self._dropped[side]:]
            if left:
                out[a, :len(left)] = np.stack(left)
                mask[a, :len(left)] = True
        self.history.append((self.stage, self.transfer.target if self.transfer else None))
        return out, mask

    def _stage_vias(self, scene, eef, upper) -> dict[str, list[np.ndarray]]:
        """Advance the stage machine and return this step's via-points per arm."""
        t = self.transfer
        if t is None or scene.get(t.target) is None:
            self._select(scene)
            t = self.transfer
            if t is None:
                self.stage = "none"
                return {}
        target = scene.get(t.target)
        held = t.target in objects_lifted(scene, self.spec)
        # "On the upper shelf" holds while it is still carried above it; placed means
        # resting at the planned height (else the preplace would be skipped).
        placed = (t.target in upper and t.rest_z is not None
                  and abs(target.pose.position[2] - t.rest_z) <= REST_TOL)
        mover = self._eef(eef, self.mover)
        if self.stage in ("park_mover", "compact") or (placed and t.place_pose is not None):
            return self._after_place(scene, eef, mover, target)
        if held and t.place_pose is not None:
            self.stage = "carry"
            return {self.mover: [_via(t.pregrasp_pose, CLOSED), *self._rest_of_transfer()]}
        self.stage = "grasp"
        self._plan_grasp(scene)
        return {self.mover: [_via(t.pregrasp_pose, OPEN), _via(t.grasp_pose, CLOSED),
                             _via(t.pregrasp_pose, CLOSED), *self._rest_of_transfer()]}

    def _rest_of_transfer(self) -> list[np.ndarray]:
        """Lookahead: the mover's script after the lift (preplace, place and open, back
        off), so a 40-step chunk that crosses a stage boundary is covered. The stage
        itself still advances only on observation."""
        t = self.transfer
        return [_via(t.preplace_pose, CLOSED), _via(t.place_pose, OPEN),
                _via(backed_off(t.place_pose, STANDOFF), OPEN)]

    def _after_place(self, scene, eef, mover, target) -> dict[str, list[np.ndarray]]:
        t = self.transfer
        if self.stage not in ("retreat", "park_mover", "compact"):
            if np.linalg.norm(mover - np.asarray(target.pose.position)) <= RETREAT_START:
                self.stage = "release"
                return {self.mover: [_via(t.place_pose, OPEN),
                                     _via(backed_off(t.place_pose, STANDOFF), OPEN)]}
            self.stage = "retreat"
        if self.stage == "retreat":
            back = backed_off(t.place_pose, STANDOFF)
            # Straight up: done at the height, even if a little off in x/y.
            if (np.linalg.norm(mover - np.asarray(back.position)) > REACH
                    and mover[2] < back.position[2] - REACH):
                return {self.mover: [_via(back, OPEN)]}
            if not self._compaction_pending(scene):
                return self._next(scene, eef)
            self.stage = "park_mover"
        if self.stage == "park_mover":
            if np.linalg.norm(mover - np.asarray(self.home[self.mover].position)) > REACH:
                return {self.mover: [_via(self.home[self.mover], OPEN)]}
            self.stage = "compact"
        # compact: plan once, against the object where it actually is.
        if t.push is None:
            push = plan_push(target, t.compact_x, standoff=STANDOFF)
            start = getattr(eef, self.compactor).pose
            t.push = [_via(start, CLOSED), _via(push.approach_pose, CLOSED),
                      _via(push.start_pose, CLOSED), _via(push.end_pose, CLOSED),
                      _via(push.retreat_pose, CLOSED), _via(self.home[self.compactor], CLOSED)]
        comp = self._eef(eef, self.compactor)
        if (self._dropped[self.compactor] >= len(t.push) - 1
                and np.linalg.norm(comp - t.push[-1][:3]) <= REACH):
            return self._next(scene, eef)
        return {self.compactor: list(t.push), self.mover: [_via(self.home[self.mover], OPEN)]}

    def _next(self, scene, eef) -> dict[str, list[np.ndarray]]:
        self._select(scene)
        self.stage = "none"
        if self.transfer is None:
            return {}
        return self._stage_vias(scene, eef, objects_on_upper_shelf(scene, self.spec))


def episode_arm_targets(episode, spec: ShelfRestockSpec) -> tuple[np.ndarray, np.ndarray, list]:
    """The oracle over a recorded episode: ``[T, 2, VIA, 8]``, ``[T, 2, VIA]``, stages."""
    oracle = ShelfArmTargetOracle(spec)
    out = [oracle(s.scene, s.observation.eef) for s in episode.samples]
    if oracle.unplannable_steps:
        raise ValueError(f"seed {episode.seed}: the expert's planners failed on "
                         f"{oracle.unplannable_steps} recorded frames ({oracle.last_failure})")
    return (np.stack([v for v, _ in out]), np.stack([m for _, m in out]),
            [stage for stage, _ in oracle.history])
