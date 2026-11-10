"""Simulator-free generator of bimanual interaction episodes.

An episode is a scene (supports, 1-4 movable boxes, two grippers) and a program
(programs.py), executed analytically into a sequence of *interaction keyframes*. A
keyframe is the full entity table at the moment of a contact-mode change:

    grasp    a gripper closes on an object       (object becomes held)
    regrasp  the second gripper closes on a held object (dual-held; handover only)
    release  the giving gripper opens            (receiver now holds it)
    place    the holding gripper opens with the object on its destination (supported)

The physics is a set of explicit checks rather than integration:
- a grasp needs the object narrower than the hand's opening;
- reach is a radius from each arm's base, plus the measured cross-body limit (oracle/arms.py);
- a held object follows its gripper rigidly;
- a placement needs its footprint on the destination and no overlap with earlier placements;
- a stack needs the centre of mass over the lower object and a bounded height.

An infeasible program/scene pair is resampled, never repaired, so every kept episode is
a successful demonstration (GEMIL kept only successful demos too).

A training item is K support episodes of one program on different scenes, plus a query:
the initial and the current state of another scene under the same program. The target is
the next keyframe: which object, which gripper, which event, and the poses after it.
Conflict items share a query state and the support layouts with a partner program whose
next event differs, so only the support (or the instruction) can tell them apart. That
is the data design that made GEMIL's model use its context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import programs as P
from .schema import (
    APERTURE,
    GRASP,
    INITIAL,
    LEFT_GRIPPER,
    MAX_OPENING,
    MOVABLE,
    N_FEATURES,
    OPEN,
    PLACE,
    POS,
    REGRASP,
    RELEASE,
    RIGHT_GRIPPER,
    SIZE,
    SUPPORT,
    YAW,
    yaw_features,
)

MAX_OBJECTS = 4
MAX_SUPPORTS = 3
N_MAX = MAX_OBJECTS + 2 + MAX_SUPPORTS  # entity columns of a table
S_MAX = 1 + 4 * MAX_OBJECTS  # keyframe rows of a demo (handover: 4 per object)
K_MAX = 4

#: Arm geometry, in the workcell frame (oracle/arms.py: bases at x = -0.4 / +0.4).
ARM_BASE_X = (-0.4, 0.4)
ARM_BASE_Y = -0.45
REACH = 0.85
CROSS_BODY_X, CROSS_BODY_Y = 0.15, -0.05
HAND_CLEARANCE = 0.12  # minimum centre spacing of neighbouring objects (spec: 0.15)
PLACE_GAP = 0.04  # spacing of a row on the destination (spec: compaction_distance)
EDGE_MARGIN = 0.02
MAX_STACK_HEIGHT = 0.22
LIFT = 0.10  # how far a gripper retreats upward after it opens


class Infeasible(Exception):
    """The program cannot be executed on this scene; the caller resamples."""


@dataclass
class Scene:
    obj_pos: np.ndarray  # [n, 3] centres
    obj_yaw: np.ndarray  # [n]
    obj_size: np.ndarray  # [n, 3]
    sup_pos: np.ndarray  # [m, 3] deck centres
    sup_size: np.ndarray  # [m, 3]
    grip_pos: np.ndarray  # [2, 3] left, right
    source: int = 0  # index of the support the objects start on

    @property
    def n(self) -> int:
        return len(self.obj_pos)

    def sup_top(self, i: int) -> float:
        return float(self.sup_pos[i, 2] + self.sup_size[i, 2] / 2)


@dataclass
class Keyframe:
    feats: np.ndarray  # [N, 9] entity table (movables, left, right, supports)
    event: int
    gripper: int  # 0 left, 1 right, -1 for "initial"
    obj: int  # acted object index, -1 for "initial"


@dataclass
class Episode:
    scene: Scene
    program: P.Program
    keyframes: list[Keyframe] = field(default_factory=list)
    final_pose: dict[int, np.ndarray] = field(default_factory=dict)  # obj -> (x, y, z, yaw)

    @property
    def types(self) -> np.ndarray:
        n, m = self.scene.n, len(self.scene.sup_pos)
        return np.array([MOVABLE] * n + [LEFT_GRIPPER, RIGHT_GRIPPER] + [SUPPORT] * m)


# ----------------------------------------------------------------------------- scenes


def sample_scene(
    rng: np.random.Generator, n: int | None = None, n_dest: int | None = None
) -> Scene:
    """A source support with n objects in a row and one or two destination supports.

    Every range is wider than the shelf task's (spec.py DEFAULT_SPEC), so the shelf
    is inside the prior's support without the prior being the shelf.
    """
    n = int(rng.integers(1, MAX_OBJECTS + 1)) if n is None else n
    n_dest = int(rng.choice([1, 2], p=[0.3, 0.7])) if n_dest is None else n_dest
    shift = rng.uniform(-0.05, 0.05, 3)
    src_size = np.array([rng.uniform(0.6, 1.0), rng.uniform(0.15, 0.25), 0.02])
    src_top = rng.uniform(0.70, 0.80)
    src_pos = (
        np.array([rng.uniform(-0.12, 0.08), rng.uniform(-0.30, -0.22), src_top - 0.01]) + shift
    )
    sups = [(src_pos, src_size)]
    for _ in range(200):
        dests = []
        for _ in range(n_dest):
            size = np.array([rng.uniform(0.25, 0.6), rng.uniform(0.10, 0.16), 0.03])
            top = src_top + rng.uniform(-0.05, 0.25)
            dests.append(
                (
                    np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.12, 0.04), top - 0.015])
                    + shift,
                    size,
                )
            )
        if n_dest == 2:
            (p0, s0), (p1, s1) = dests
            if abs(p0[2] - p1[2]) < 0.06 or abs(p0[0] - p1[0]) < (s0[0] + s1[0]) / 2 + 0.02:
                continue
        # Destinations must not overhang the source row (a top-down grasp needs the space).
        if any(p[1] - s[1] / 2 < src_pos[1] + src_size[1] / 2 + 0.04 for p, s in dests):
            continue
        sups += dests
        break
    else:
        raise Infeasible("could not place destination supports")
    sizes = np.stack(
        [rng.uniform(0.03, 0.06, n), rng.uniform(0.03, 0.06, n), rng.uniform(0.03, 0.08, n)], -1
    )
    xs = _spaced(
        rng,
        n,
        src_pos[0] - src_size[0] / 2 + 0.06,
        src_pos[0] + src_size[0] / 2 - 0.06,
        rng.uniform(HAND_CLEARANCE, 0.20),
    )
    ys = src_pos[1] + rng.uniform(-1.0, 1.0, n) * (src_size[1] / 2 - 0.04)
    obj_pos = np.stack([xs, ys, src_top + sizes[:, 2] / 2], -1)
    grip = np.array([[-0.30, -0.18, src_top + 0.30], [0.30, -0.18, src_top + 0.30]]) + rng.uniform(
        -0.05, 0.05, (2, 3)
    )
    return Scene(
        obj_pos=obj_pos,
        obj_yaw=rng.uniform(-0.4, 0.4, n),
        obj_size=sizes,
        sup_pos=np.stack([p for p, _ in sups]),
        sup_size=np.stack([s for _, s in sups]),
        grip_pos=grip,
    )


def _spaced(rng, n, lo, hi, gap):
    span = hi - lo - gap * (n - 1)
    if span < 0:
        raise Infeasible("source too short for the objects")
    u = np.sort(rng.uniform(0, span, n))
    return lo + u + gap * np.arange(n)


# ----------------------------------------------------------------------------- program semantics


def destination(scene: Scene, rule: str) -> int:
    cands = [i for i in range(len(scene.sup_pos)) if i != scene.source]
    if rule == "higher":
        return max(cands, key=scene.sup_top)
    if rule == "lower":
        return min(cands, key=scene.sup_top)
    if rule == "left_side":
        return min(cands, key=lambda i: scene.sup_pos[i, 0])
    if rule == "right_side":
        return max(cands, key=lambda i: scene.sup_pos[i, 0])
    raise ValueError(rule)


def order(scene: Scene, selector: str, dest: int) -> list[int]:
    x, y = scene.obj_pos[:, 0], scene.obj_pos[:, 1]
    idx = list(range(scene.n))
    if selector == "left_to_right":
        return sorted(idx, key=lambda i: x[i])
    if selector == "right_to_left":
        return sorted(idx, key=lambda i: -x[i])
    if selector == "front_to_back":
        _require_distinct(y, 0.02)
        return sorted(idx, key=lambda i: y[i])
    if selector == "back_to_front":
        _require_distinct(y, 0.02)
        return sorted(idx, key=lambda i: -y[i])
    if selector == "middle_out":
        if scene.n % 2 == 0:
            raise Infeasible("middle_out needs an odd count")
        s = sorted(idx, key=lambda i: x[i])
        mid = len(s) // 2
        out = [s[mid]]
        for k in range(1, mid + 1):
            out += [s[mid - k], s[mid + k]]
        return out
    if selector == "outside_in":
        s, out = sorted(idx, key=lambda i: x[i]), []
        while s:
            out.append(s.pop(0))
            if s:
                out.append(s.pop())
        return out
    if selector in ("near_dest_first", "far_dest_first"):
        d = np.linalg.norm(scene.obj_pos[:, :2] - scene.sup_pos[dest, :2], axis=-1)
        _require_distinct(d, 0.03)
        return sorted(idx, key=lambda i: d[i] if selector == "near_dest_first" else -d[i])
    if selector in ("small_to_large", "large_to_small"):
        v = np.prod(scene.obj_size, -1)
        _require_distinct(np.log(v), np.log(1.25))
        return sorted(idx, key=lambda i: v[i] if selector == "small_to_large" else -v[i])
    raise ValueError(selector)


def _require_distinct(values, gap):
    s = np.sort(values)
    if len(s) > 1 and np.min(np.diff(s)) < gap:
        raise Infeasible("the selector's key is ambiguous on this scene")


def _reachable(arm: int, p: np.ndarray) -> bool:
    if np.hypot(p[0] - ARM_BASE_X[arm], p[1] - ARM_BASE_Y) > REACH:
        return False
    if p[1] >= CROSS_BODY_Y:
        return p[0] < CROSS_BODY_X if arm == 0 else p[0] > -CROSS_BODY_X
    return True


def _half_x(size, yaw):
    return abs(np.cos(yaw)) * size[0] / 2 + abs(np.sin(yaw)) * size[1] / 2


# ----------------------------------------------------------------------------- execution


def execute(scene: Scene, program: P.Program, rng: np.random.Generator) -> Episode:
    """Run a program to completion as keyframes, or raise Infeasible."""
    dest = destination(scene, program.destination)
    seq = order(scene, program.selector, dest)
    obj_pos, obj_yaw = scene.obj_pos.copy(), scene.obj_yaw.copy()
    grip_pos, aperture = scene.grip_pos.copy(), np.full(2, OPEN)
    ep = Episode(scene, program)

    def snap(event, gripper, obj):
        ep.keyframes.append(
            Keyframe(_table(scene, obj_pos, obj_yaw, grip_pos, aperture), event, gripper, obj)
        )

    snap(INITIAL, -1, -1)
    dx, dy, _ = scene.sup_size[dest]
    top = scene.sup_top(dest)
    placed: list[int] = []
    for i in seq:
        size = scene.obj_size[i]
        if min(size[0], size[1]) > 0.95 * MAX_OPENING:
            raise Infeasible("object wider than the hand")
        side = 0 if obj_pos[i, 0] < 0 else 1
        giver = {"left": 0, "right": 1, "nearest": side, "handover": side}[program.arm]
        holder = 1 - giver if program.arm == "handover" else giver

        # Destination pose (aligned with the support, yaw 0).
        if program.arrangement == "stack":
            if placed:
                below = placed[-1]
                if np.prod(size[:2]) > 1.3 * np.prod(scene.obj_size[below, :2]):
                    raise Infeasible("stack: upper footprint too large for the lower object")
                base = obj_pos[below, 2] + scene.obj_size[below, 2] / 2
                xy = obj_pos[below, :2]
            else:
                base, xy = top, scene.sup_pos[dest, :2] + rng.uniform(-0.03, 0.03, 2) * [dx, dy] * 4
            target = np.array([xy[0], xy[1], base + size[2] / 2])
            if target[2] + size[2] / 2 - top > MAX_STACK_HEIGHT:
                raise Infeasible("stack too tall")
        else:
            sign = 1.0 if program.arrangement == "row_left" else -1.0
            edge = scene.sup_pos[dest, 0] - sign * dx / 2
            used = sum(2 * _half_x(scene.obj_size[j], 0.0) + PLACE_GAP for j in placed)
            cx = edge + sign * (EDGE_MARGIN + used + _half_x(size, 0.0))
            if abs(cx - scene.sup_pos[dest, 0]) + _half_x(size, 0.0) > dx / 2:
                raise Infeasible("row overflows the destination")
            target = np.array([cx, scene.sup_pos[dest, 1], top + size[2] / 2])
        if abs(target[1] - scene.sup_pos[dest, 1]) + size[1] / 2 > dy / 2 + 1e-9:
            raise Infeasible("footprint off the destination")

        # Grasp: the giver closes on the object, fingertips a little below its top.
        grip_off = np.array([0.0, 0.0, size[2] / 2 - min(0.02, size[2] / 2)])
        if not _reachable(giver, obj_pos[i]):
            raise Infeasible("grasp out of reach")
        grip_pos[giver] = obj_pos[i] + grip_off
        aperture[giver] = min(size[0], size[1]) / MAX_OPENING
        snap(GRASP, giver, i)

        if program.arm == "handover":
            meet = np.array(
                [
                    rng.uniform(-0.05, 0.05),
                    rng.uniform(-0.20, -0.10),
                    max(scene.sup_top(scene.source), top) + rng.uniform(0.12, 0.20),
                ]
            )
            if not (_reachable(giver, meet) and _reachable(holder, meet)):
                raise Infeasible("handover point out of reach")
            obj_pos[i] = meet
            grip_pos[giver] = meet + grip_off
            side_off = np.array([0.03 if holder == 1 else -0.03, 0.0, 0.0])
            grip_pos[holder] = meet + side_off
            aperture[holder] = min(size[0], size[2]) / MAX_OPENING
            snap(REGRASP, holder, i)
            aperture[giver] = OPEN
            grip_pos[giver] = grip_pos[giver] + [0.0, 0.0, LIFT]
            snap(RELEASE, giver, i)
            hold_off = side_off
        else:
            hold_off = grip_off

        if not _reachable(holder, target):
            raise Infeasible("placement out of reach")
        obj_pos[i], obj_yaw[i] = target, 0.0
        grip_pos[holder] = target + hold_off
        aperture[holder] = OPEN
        snap(PLACE, holder, i)
        grip_pos[holder] = grip_pos[holder] + [0.0, 0.0, LIFT]
        ep.final_pose[i] = np.array([*target, 0.0])
        placed.append(i)
    return ep


def _table(scene, obj_pos, obj_yaw, grip_pos, aperture):
    n, m = scene.n, len(scene.sup_pos)
    t = np.zeros((n + 2 + m, N_FEATURES))
    t[:n, POS], t[:n, YAW], t[:n, SIZE] = obj_pos, yaw_features(obj_yaw), scene.obj_size
    t[n : n + 2, POS], t[n : n + 2, APERTURE] = grip_pos, aperture
    t[n : n + 2, YAW] = yaw_features(np.zeros(2))
    t[n + 2 :, POS], t[n + 2 :, SIZE] = scene.sup_pos, scene.sup_size
    t[n + 2 :, YAW] = yaw_features(np.zeros(m))
    return t


def interpolate(a: Keyframe, b: Keyframe, alpha: float) -> np.ndarray:
    """The scene partway from keyframe a to b. Positions are linear (a held object keeps
    its offset to its gripper); apertures stay at a's values until the event."""
    t = a.feats.copy()
    t[:, POS] = (1 - alpha) * a.feats[:, POS] + alpha * b.feats[:, POS]
    yaw = (1 - alpha) * a.feats[:, YAW] + alpha * b.feats[:, YAW]
    t[:, YAW] = yaw / np.linalg.norm(yaw, axis=-1, keepdims=True).clip(1e-6)
    return t


def sample_episode(
    rng: np.random.Generator, program: P.Program, scene: Scene | None = None, tries: int = 50
) -> Episode:
    """A successful episode of `program`, resampling scenes that cannot run it."""
    for _ in range(tries if scene is None else 1):
        try:
            return execute(scene if scene is not None else sample_scene(rng), program, rng)
        except Infeasible:
            continue
    raise Infeasible(f"no feasible scene for {program} in {tries} tries")
