"""Oracle next-interaction keyframes, read off a recorded episode.

The PFN-VLA future branch (``policies/conditioning/future.py``) attends a small
set of rows describing the *next* interaction. The learned prior predicts it;
this module computes the oracle answer from the recording, as an upper bound
and as training targets. Nothing here is visible to the learned path.

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


def _row(token, role: str, event: str) -> np.ndarray:
    row = np.zeros(FUTURE_DIM, dtype=np.float32)
    row[:ENTITY_TOKEN_DIM] = token
    row[ENTITY_TOKEN_DIM + FUTURE_ROLES.index(role)] = 1.0
    row[ENTITY_TOKEN_DIM + len(FUTURE_ROLES) + FUTURE_EVENTS.index(event)] = 1.0
    return row


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
