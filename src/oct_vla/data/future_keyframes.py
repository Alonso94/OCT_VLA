"""The future-keyframe row contract, and a phase-label reference for it.

The PFN-VLA future branch (``policies/conditioning/future.py``) attends a small
set of rows describing the *next* interaction. The learned prior predicts it;
the oracle that the dataset and the evaluation server use is
``tasks/shelf_restock/keyframes.py``, which needs only the scene. This module
holds the row layout they share, and an independent definition read off a
recording's phase labels, which ``scripts/add_future_keyframe.py`` uses to
cross-check the oracle episode by episode. Nothing here is visible to the
learned path.

Events are the two the prior predicts and the recording labels: ``grasp`` at
the last frame of a ``close`` run and ``place`` at the last frame of a
``release`` run. The compactor's ``compact close`` is deliberately not an
event: it is a push, which the prior has no vocabulary for, and the oracle must
predict the same thing the prior does or the oracle-to-PFN gap is meaningless.

For frame ``t`` the keyframe is the first event strictly after ``t``. After the
last place the mask is all False, which the branch turns into an exact zero.

Row layout (``FUTURE_DIM`` = 17 + 3 + 4 columns), one row per role::

    [ 0:17]  entity-schema-v2 token (data/entity_tokens.py)
    [17:20]  role one-hot, FUTURE_ROLES order
    [20:24]  event one-hot, FUTURE_EVENTS order

**Waypoint layout** (``FUTURE_DIM_WP`` = 17 + 4 + 4 = 25, ``fkwp`` views): a fourth
role, ``gripper_waypoint``, the acting gripper's approach pose. That is the event's
gripper pose raised to a height that clears every nearby object
(``approach_waypoint``), so the policy is told to arrive *above* the event and
descend, not to cut in sideways past a placed object. The row is phase-dependent:
it is present until the mover reaches it (``WAYPOINT_REACHED_*``), then masked until
the next event, so it says "go here first" and then "now descend". Its mask is
therefore not all-or-nothing; whether a keyframe exists at all is the first row's. The first three rows are the
same as the 24-wide layout's, with one more role column. The 24-wide layout and every
view written with it stay valid; ``FUTURE_LAYOUTS`` maps a width to its roles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oct_vla.data.entity_tokens import ENTITY_TOKEN_DIM, _token
from oct_vla.data.episode import Episode

FUTURE_ROLES = ("object_keyframe", "gripper_keyframe", "object_goal")
#: The same four as ``oct_vla.prior.schema.TARGET_EVENTS``; the oracle emits the
#: first two.
FUTURE_EVENTS = ("grasp", "place", "regrasp", "release")
FUTURE_EXTRA = len(FUTURE_ROLES) + len(FUTURE_EVENTS)
FUTURE_DIM = ENTITY_TOKEN_DIM + FUTURE_EXTRA
FUTURE_ROWS = len(FUTURE_ROLES)

FUTURE_ROLES_WP = FUTURE_ROLES + ("gripper_waypoint",)
FUTURE_DIM_WP = ENTITY_TOKEN_DIM + len(FUTURE_ROLES_WP) + len(FUTURE_EVENTS)
FUTURE_ROWS_WP = len(FUTURE_ROLES_WP)
#: Row width -> the roles of that layout.
FUTURE_LAYOUTS = {FUTURE_DIM: FUTURE_ROLES, FUTURE_DIM_WP: FUTURE_ROLES_WP}

#: The approach waypoint (``approach_waypoint``). Heights are of the EEF frame,
#: which sits ``EEF_TO_FINGERTIP`` above the fingertips when pointing down.
WAYPOINT_LIFT = 0.10  # at least this far above the event pose
WAYPOINT_CLEARANCE = 0.05  # fingertips this far above the tallest nearby top
WAYPOINT_RADIUS = 0.10  # horizontal reach of the fingers around the event, beyond an object's half-extent
EEF_TO_FINGERTIP = 0.13
#: The waypoint counts as reached (and its row is masked until the next event) once
#: the mover's EEF is this close to it horizontally and vertically. The recorded expert
#: passes within ~1 cm of it on 23 of 24 approaches checked (it uses the same 0.10 m
#: standoff), so the switch is defined on the training data.
WAYPOINT_REACHED_XY = 0.02
WAYPOINT_REACHED_Z = 0.02

#: Phase label -> event, for the phases the oracle reads.
_PHASE_EVENTS = {"close": "grasp", "release": "place"}
#: The EEF pose sits ~0.14 m from a grasped object's centre (0.138-0.156 m over
#: all 285 grasps of the v4 corpus; the idle arm was >= 0.469 m away). A grasp
#: outside this, or without a clear margin over the other arm, is a labelling
#: or recording fault; refuse rather than pick the wrong arm.
MAX_GRASP_DISTANCE = 0.25
MIN_ARM_MARGIN = 2.0


@dataclass(frozen=True)
class Event:
    frame: int
    kind: str  # "grasp" | "place"
    track_id: str
    side: str  # "left" | "right"


def _object(sample, track_id):
    for item in sample.scene.objects:
        if item.track_id == track_id:
            return item
    raise ValueError(f"t={sample.timestamp}: target {track_id!r} not in the scene")


def _nearest_side(sample, track_id) -> tuple[str, float, float]:
    """The arm nearest the object, its distance, and the other arm's."""
    target = np.asarray(_object(sample, track_id).pose.position)
    eef = sample.observation.eef
    dist = {side: float(np.linalg.norm(np.asarray(getattr(eef, side).pose.position) - target))
            for side in ("left", "right")}
    near, far = sorted(dist, key=dist.get)
    return near, dist[near], dist[far]


def interaction_events(episode: Episode) -> list[Event]:
    """Every grasp and place of the episode, in order, with the acting arm."""
    samples = episode.samples
    runs: list[tuple[int, str, str]] = []  # (last frame, event, track)
    for index, sample in enumerate(samples):
        kind = _PHASE_EVENTS.get(sample.phase)
        if kind is None:
            continue
        track = sample.context.target_track_id
        nxt = samples[index + 1] if index + 1 < len(samples) else None
        if nxt is None or nxt.phase != sample.phase or nxt.context.target_track_id != track:
            runs.append((index, kind, track))
    events: list[Event] = []
    holder: dict[str, str] = {}
    for frame, kind, track in runs:
        if kind == "grasp":
            side, dist, other = _nearest_side(samples[frame], track)
            if dist > MAX_GRASP_DISTANCE or other < MIN_ARM_MARGIN * dist:
                raise ValueError(f"seed {episode.seed}: grasp of {track} at frame {frame} has no "
                                 f"clear acting arm ({side} {dist:.3f} m, other {other:.3f} m)")
            holder[track] = side
        else:
            if track not in holder:
                raise ValueError(f"seed {episode.seed}: {track} placed at frame {frame} "
                                 "without a grasp")
            side = holder.pop(track)
        events.append(Event(frame, kind, track, side))
    if holder:
        raise ValueError(f"seed {episode.seed}: grasped and never placed: {sorted(holder)}")
    return events


def _row(token, role: str, event: str, roles: tuple[str, ...] = FUTURE_ROLES) -> np.ndarray:
    row = np.zeros(ENTITY_TOKEN_DIM + len(roles) + len(FUTURE_EVENTS), dtype=np.float32)
    row[:ENTITY_TOKEN_DIM] = token
    row[ENTITY_TOKEN_DIM + roles.index(role)] = 1.0
    row[ENTITY_TOKEN_DIM + len(roles) + FUTURE_EVENTS.index(event)] = 1.0
    return row


def event_columns(dim: int) -> slice:
    """The event one-hot's columns in a row of width ``dim``."""
    if dim not in FUTURE_LAYOUTS:
        raise ValueError(f"no keyframe layout is {dim} wide; have {sorted(FUTURE_LAYOUTS)}")
    start = ENTITY_TOKEN_DIM + len(FUTURE_LAYOUTS[dim])
    return slice(start, start + len(FUTURE_EVENTS))


def approach_waypoint(event_position, obstacles,
                      frame_to_tip: float = EEF_TO_FINGERTIP) -> np.ndarray:
    """The EEF position above ``event_position`` from which a vertical descent clears
    every obstacle.

    ``obstacles``: ``(centre, size_xyz)`` pairs, the objects other than the one held
    or grasped. One whose footprint, widened by ``WAYPOINT_RADIUS``, covers the event's
    horizontal position raises the waypoint until the fingertips clear its top by
    ``WAYPOINT_CLEARANCE``. The waypoint is never lower than ``WAYPOINT_LIFT`` above
    the event. ``frame_to_tip`` is the height of the position's frame above the
    fingertips: the EEF's on the real shelf, ~0 for the synthetic prior's gripper point
    (``prior/shelf_adapter.py``). The real-shelf oracle and the synthetic prior's labels
    both call this, so the two define the waypoint identically."""
    event = np.asarray(event_position, dtype=np.float64)
    z = event[2] + WAYPOINT_LIFT
    for centre, size in obstacles:
        centre, size = np.asarray(centre, np.float64), np.asarray(size, np.float64)
        reach = WAYPOINT_RADIUS + 0.5 * float(np.max(size[:2]))
        if np.linalg.norm(centre[:2] - event[:2]) <= reach:
            z = max(z, centre[2] + 0.5 * size[2] + frame_to_tip + WAYPOINT_CLEARANCE)
    return np.array([event[0], event[1], z])


def event_rows(episode: Episode, event: Event, goal_frame: int) -> np.ndarray:
    """The ``[FUTURE_ROWS, FUTURE_DIM]`` keyframe describing one event."""
    at, goal = episode.samples[event.frame], episode.samples[goal_frame]
    item = _object(at, event.track_id)
    arm = getattr(at.observation.eef, event.side)
    final = _object(goal, event.track_id)
    return np.stack([
        _row(_token(item.pose, item.size_xyz, 0.0, "movable"), "object_keyframe", event.kind),
        _row(_token(arm.pose, (0.0, 0.0, 0.0), arm.gripper, f"{event.side}_gripper"),
             "gripper_keyframe", event.kind),
        _row(_token(final.pose, final.size_xyz, 0.0, "movable"), "object_goal", event.kind),
    ])


def oracle_future_keyframes(episode: Episode) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, the next event's keyframe: ``[T, FUTURE_ROWS, FUTURE_DIM]`` and a ``[T, FUTURE_ROWS]`` mask."""
    events = interaction_events(episode)
    if not events:
        raise ValueError(f"seed {episode.seed}: no grasp or place in the episode")
    place_of = {e.track_id: e.frame for e in events if e.kind == "place"}
    rows = [event_rows(episode, e, place_of[e.track_id]) for e in events]
    count = len(episode.samples)
    keyframes = np.zeros((count, FUTURE_ROWS, FUTURE_DIM), dtype=np.float32)
    mask = np.zeros((count, FUTURE_ROWS), dtype=bool)
    current = 0
    for t in range(count):
        while current < len(events) and events[current].frame <= t:
            current += 1
        if current < len(events):
            keyframes[t], mask[t] = rows[current], True
    return keyframes, mask
