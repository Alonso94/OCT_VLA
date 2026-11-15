"""Hindsight object effects: what the robot is about to change, from observation alone.

The effect-conditioned VLA is trained on recorded demonstrations whose every frame is
labelled with the next object-level effect Z = (object, desired pose), and at test
time takes Z from an oracle or a PFN reading actionless demonstrations. This module
derives Z from **object poses, end-effector positions and the measured gripper
aperture only** -- no phase labels, no planner, no task rule -- so the same label
could be computed for any recorded interaction.

Per transfer it builds a timeline (frame indices):

    grasp      the mover's gripper starts closing from fully open, before the lift
    lift       the object rises LIFT_TOL above its initial height
    place      it rests in the upper-shelf band (height still for REST_FRAMES)
    release    the gripper starts opening (or, for an object held nearly open, the
               mover starts leaving it)
    let_go     placed and the mover beyond RELEASE_DISTANCE -- the keyframe oracle's
               own switch test (tasks/shelf_restock/keyframes.py), so offline and
               online switch on the same observable
    push       the object moving with the other arm (the compactor) the nearer one;
               of such runs the one that moves it furthest is its own push
    settled    after its push, the compactor CLEAR_DIST further away than at push end

Two label contracts, predeclared (paper/5, plan rev. 11):

- ``"A"`` (primary) **placement effect**: Z_t = (o_k, its pose at let_go) for the
  first object, in pick order, whose let_go is after t. During retreat and
  compaction Z is already the next object -- exactly how ``ShelfKeyframeOracle``
  advances, and the place-frame goal the real-data P2 labels use.
- ``"B"`` **placement then compaction**: an object that is pushed keeps Z until it
  has settled, with its post-push pose as the goal.

Rows use the goal-only 3x24 layout a goal-only prior emits
(``prior/shelf_adapter.keyframe_rows``): row 0 the object now, row 2 its goal, both
yaw-only poses built through the same token path, gripper row masked, event empty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.data.entity_tokens import ENTITY_TOKEN_DIM, _token
from oct_vla.data.future_keyframes import FUTURE_DIM, FUTURE_ROWS

CONTRACTS = ("A", "B")

LIFT_TOL = 0.01         # m above the initial height
STEP_TOL = 0.0005       # m per frame that counts as moving (pushes)
CLEAR_DIST = 0.10       # m an arm retreats beyond its distance at the event
FULL_OPEN_TOL = 0.003   # aperture within this of the arm's maximum is fully open
PUSH_GAP = 10           # frames: pauses inside one push
REST_FRAMES = 3         # frames the height must be still to count as placed
PLACE_DZ = 0.001        # m per frame of height change: stopped descending
OPENING = 0.01          # aperture rise above the holding value that starts a release
LEAVING = 0.02          # m the mover moves off the object: the fallback release cue
#: tasks/shelf_restock/keyframes.py RELEASE_DISTANCE (asserted equal in the tests).
RELEASE_DISTANCE = 0.20
SIDES = ("left", "right")


def _first(cond, start: int = 0) -> int | None:
    idx = np.flatnonzero(cond[start:])
    return int(idx[0] + start) if len(idx) else None


def object_timeline(p: np.ndarray, eef: np.ndarray, aperture: np.ndarray,
                    upper: np.ndarray) -> dict:
    """One object's timeline. ``p`` [T, 3] its position, ``eef`` [T, 2, 3] both arms,
    ``aperture`` [T, 2] measured (continuous), ``upper`` [T] in the upper-shelf band.
    Keys are absent (or None) for events that did not happen."""
    T = len(p)
    out: dict = {}
    lift = _first(p[:, 2] > p[0, 2] + LIFT_TOL)
    out["lift"] = lift
    if lift is None:
        return out
    dist = np.linalg.norm(eef - p[:, None, :], axis=-1)
    side = int(np.argmin(dist[lift]))
    other = 1 - side
    out["mover"] = SIDES[side]
    full = aperture[:, side] >= aperture[:, side].max() - FULL_OPEN_TOL
    opens = np.flatnonzero(full[:lift + 1])
    out["grasp"] = int(opens[-1] + 1) if len(opens) else None
    dz = np.r_[0.0, np.abs(np.diff(p[:, 2]))]
    resting = np.array([dz[max(0, t - REST_FRAMES + 1):t + 1].max() < PLACE_DZ
                        for t in range(T)])
    out["place"] = _first(upper & resting, lift)
    if out["place"] is None:
        return out
    q = out["place"]
    opening = aperture[:, side] > aperture[q, side] + OPENING
    leaving = dist[:, side] > dist[q, side] + LEAVING
    out["release"] = _first(opening | leaving, q)
    if out["release"] is None:
        return out
    r = out["release"]
    out["let_go"] = _first(upper & (dist[:, side] > RELEASE_DISTANCE), r)
    out["clear"] = _first(dist[:, side] > dist[r, side] + CLEAR_DIST, r)
    steps = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=-1)]
    pushing = (steps > STEP_TOL) & (dist[:, other] < dist[:, side])
    moving = np.flatnonzero(pushing[r:]) + r
    if len(moving):
        runs, start, end = [], moving[0], moving[0]
        for f in moving[1:]:
            if f - end > PUSH_GAP:
                runs.append((start, end))
                start = f
            end = f
        runs.append((start, end))
        shift = [np.linalg.norm(p[e] - p[max(b - 1, 0)]) for b, e in runs]
        start, end = runs[int(np.argmax(shift))]
        out["push_start"], out["push_end"] = int(start), int(end)
        out["push_runs"] = len(runs)
        out["pusher"] = SIDES[other]
        out["settled"] = _first(dist[:, other] > dist[end, other] + CLEAR_DIST, end)
    return out


@dataclass(frozen=True)
class Tracks:
    """An episode as arrays: object ids, positions [T, N, 3], quaternions [T, N, 4]
    (xyzw), sizes [N, 3], both arms' positions [T, 2, 3] and apertures [T, 2], and the
    upper-shelf membership [T, N]."""

    ids: tuple[str, ...]
    pos: np.ndarray
    quat: np.ndarray
    sizes: np.ndarray
    eef: np.ndarray
    aperture: np.ndarray
    upper: np.ndarray


def episode_tracks(episode, spec) -> Tracks:
    samples = episode.samples
    ids = tuple(o.track_id for o in samples[0].scene.objects)
    objs = [[s.scene.get(i) for i in ids] for s in samples]
    if any(o is None for row in objs for o in row):
        raise ValueError(f"seed {episode.seed}: an object leaves the scene mid-episode")
    pos = np.array([[o.pose.position for o in row] for row in objs], float)
    quat = np.array([[o.pose.orientation for o in row] for row in objs], float)
    eef = np.array([[getattr(s.observation.eef, side).pose.position for side in SIDES]
                    for s in samples], float)
    aperture = np.array([[getattr(s.observation.eef, side).gripper for side in SIDES]
                         for s in samples], float)
    upper = np.array([[spec.upper_shelf.contains(tuple(x)) for x in row] for row in pos])
    sizes = np.array([o.size_xyz for o in objs[0]], float)
    return Tracks(ids, pos, quat, sizes, eef, aperture, upper)


def timelines(tracks: Tracks) -> dict[str, dict]:
    return {tid: object_timeline(tracks.pos[:, j], tracks.eef, tracks.aperture,
                                 tracks.upper[:, j])
            for j, tid in enumerate(tracks.ids)}


def segments(lines: dict[str, dict], contract: str) -> list[tuple[str, int, int]]:
    """(object, end frame, goal frame) per transfer in pick order. Z_t is the first
    segment whose end is after t; its goal is the object's pose at the goal frame."""
    if contract not in CONTRACTS:
        raise ValueError(f"contract must be one of {CONTRACTS}, got {contract!r}")
    out = []
    ordered = sorted(((k, v) for k, v in lines.items() if v.get("lift") is not None),
                     key=lambda kv: kv[1]["lift"])
    for i, (tid, v) in enumerate(ordered):
        if v.get("let_go") is None:
            raise ValueError(f"{tid}: lifted at {v['lift']} but never let go "
                             f"(place {v.get('place')}, release {v.get('release')})")
        end, goal = v["let_go"], v["let_go"]
        # Its own push is compaction before the next object is lifted; a push after
        # that is a neighbour's, nudging it (the first-placed object is never pushed).
        next_lift = ordered[i + 1][1]["lift"] if i + 1 < len(ordered) else None
        own = v.get("push_end") is not None and (next_lift is None
                                                 or v["push_start"] < next_lift)
        if contract == "B" and own:
            goal = v["push_end"]
            end = v["settled"] if v.get("settled") is not None else v["push_end"]
        out.append((tid, int(end), int(goal)))
    ends = [e for _, e, _ in out]
    if ends != sorted(ends):
        raise ValueError(f"segments overlap: ends {ends} are not in pick order")
    return out


def yaw_token(position, quaternion, size) -> tuple[float, ...]:
    """The object token as a goal-only prior writes it: its yaw read from the token's
    rotation exactly as ``prior.schema.from_entity_tokens`` does, then re-posed
    upright with that yaw (``prior.shelf_adapter._yaw_pose``)."""
    full = _token(Pose(tuple(position), tuple(quaternion), WORKCELL_FRAME), size, 0.0,
                  "movable")
    yaw = float(np.arctan2(full[4], full[3]))
    upright = Pose(tuple(float(v) for v in position),
                   (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))), WORKCELL_FRAME)
    return _token(upright, size, 0.0, "movable")


def effect_rows(tracks: Tracks, lines: dict[str, dict], contract: str
                ) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    """Rows [T, 3, 24], mask [T, 3] and the effect's object per frame (None after the
    last segment)."""
    T = len(tracks.pos)
    rows = np.zeros((T, FUTURE_ROWS, FUTURE_DIM), np.float32)
    mask = np.zeros((T, FUTURE_ROWS), bool)
    target: list[str | None] = [None] * T
    segs = segments(lines, contract)
    start = 0
    for tid, end, goal_frame in segs:
        j = tracks.ids.index(tid)
        size = tuple(tracks.sizes[j])
        goal = yaw_token(tracks.pos[goal_frame, j], tracks.quat[goal_frame, j], size)
        for t in range(start, end):
            rows[t, 0, :ENTITY_TOKEN_DIM] = yaw_token(tracks.pos[t, j], tracks.quat[t, j],
                                                       size)
            rows[t, 0, ENTITY_TOKEN_DIM + 0] = 1.0
            rows[t, 2, :ENTITY_TOKEN_DIM] = goal
            rows[t, 2, ENTITY_TOKEN_DIM + 2] = 1.0
            mask[t] = (True, False, True)
            target[t] = tid
        start = end
    return rows, mask, target


def hindsight_effects(episode, spec, contract: str = "A"):
    """The episode's effect rows, mask, per-frame object and timelines."""
    tracks = episode_tracks(episode, spec)
    lines = timelines(tracks)
    rows, mask, target = effect_rows(tracks, lines, contract)
    return rows, mask, target, lines
