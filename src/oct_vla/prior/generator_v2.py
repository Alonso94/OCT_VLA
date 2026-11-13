"""v2 episodes: v1's analytic execution, plus gripper orientation and compaction pushes.

`execute_v2` runs v1's `generator.execute` on the program's v1 part, then rewrites its
keyframes:

- **Gripper orientation**, as a 6D rotation: the approach axis then the closing axis, both
  unit vectors, appended to every entity row (zeros for objects and supports, which keep
  v1's planar yaw).
  - Grasp: top-down, closing across the object's narrow horizontal axis.
  - Regrasp: the receiver approaches from the side.
  - Push: horizontal along the push direction, with a random closing axis (horizontal or
    vertical). This is the push prior's distinct end-effector orientation.
- **Compaction** (`ProgramV2.compact == "toward_neighbour"`). After each place next to an
  earlier object in the row, the *other* arm:
  1. closes and touches the placed object's far face (`contact` keyframe);
  2. pushes it along the row until the gap to its neighbour is `COMPACT_GAP` (`push`
     keyframe).

  The pushed object keeps its compacted pose in every later keyframe, and its final
  pose is updated. The compactor then returns to its previous pose and orientation.
  Nothing else moves. Both push poses must be reachable, otherwise `Infeasible`.

With `compact == "none"`, the first 9 columns of every keyframe equal v1's exactly
(tested): v2 only adds columns. [limitation] Pushes are kinematic: no friction, no
toppling. The keyframes say where things end up, not how contact unfolds.
"""

from __future__ import annotations

import numpy as np

from .batches import N_OBJECTS_P
from .generator import (
    LIFT,
    Episode,
    Infeasible,
    Keyframe,
    _reachable,
    execute,
    sample_scene,
)
from .programs_v2 import ProgramV2
from .schema import EVENTS, GRASP, N_FEATURES, PLACE, POS, REGRASP, SIZE, YAW

#: v2 events: v1's, then the two push keyframes.
EVENTS_V2 = EVENTS + ("contact", "push")
CONTACT, PUSH = EVENTS_V2.index("contact"), EVENTS_V2.index("push")
TARGET_EVENTS_V2 = tuple(range(1, len(EVENTS_V2)))
#: Columns of the 6D gripper rotation (approach axis, closing axis), after v1's 9.
ROT = slice(N_FEATURES, N_FEATURES + 6)
N_FEATURES_V2 = N_FEATURES + 6
#: Gap left between compacted neighbours, and the fingertip offset from a contacted face.
COMPACT_GAP = 0.01
FINGER = 0.02
CLOSED = 0.0


def _top_down(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, -1.0, np.cos(yaw), np.sin(yaw), 0.0])


def _horizontal(direction: float, closing: np.ndarray) -> np.ndarray:
    return np.concatenate([[direction, 0.0, 0.0], closing])


def _widen(feats: np.ndarray, orient: list[np.ndarray], n: int) -> np.ndarray:
    out = np.zeros((len(feats), N_FEATURES_V2))
    out[:, :N_FEATURES] = feats
    out[n, ROT], out[n + 1, ROT] = orient[0], orient[1]
    return out


def _yaw(row) -> float:
    return float(np.arctan2(row[YAW][0], row[YAW][1]))


def execute_v2(scene, program: ProgramV2, rng: np.random.Generator) -> Episode:
    """A v2 episode (keyframe tables of `N_FEATURES_V2` columns), or `Infeasible`."""
    ep1 = execute(scene, program.v1, rng)
    n = scene.n
    orient = [_top_down(0.0), _top_down(0.0)]
    shift = np.zeros((n, 3))
    placed: list[int] = []
    out = Episode(scene, program)
    out.final_pose = {k: v.copy() for k, v in ep1.final_pose.items()}
    for kf in ep1.keyframes:
        feats = kf.feats.copy()
        feats[:n, POS] += shift
        g, o = kf.gripper, kf.obj
        if kf.event == GRASP:
            size = scene.obj_size[o]
            across = np.pi / 2 if size[0] > size[1] else 0.0
            orient[g] = _top_down(_yaw(feats[o]) + across)
        elif kf.event == REGRASP:
            side = 1.0 if g == 1 else -1.0  # the receiver sits on its own side of the object
            orient[g] = _horizontal(-side, np.array([0.0, 0.0, 1.0]))
        out.keyframes.append(Keyframe(_widen(feats, orient, n), kf.event, g, o))
        if kf.event != PLACE:
            continue
        if program.compact == "toward_neighbour" and placed:
            neighbour = placed[-1]
            p_o, p_nb = feats[o, POS].copy(), feats[neighbour, POS]
            direction = float(np.sign(p_nb[0] - p_o[0])) or 1.0
            half_o = scene.obj_size[o][0] / 2
            gap = abs(p_o[0] - p_nb[0]) - half_o - scene.obj_size[neighbour][0] / 2
            if gap > COMPACT_GAP + 1e-9:
                compactor = 1 - g
                delta = np.array([(gap - COMPACT_GAP) * direction, 0.0, 0.0])
                contact = p_o + np.array([-direction * (half_o + FINGER), 0.0, 0.0])
                end = contact + delta
                if not (_reachable(compactor, contact) and _reachable(compactor, end)):
                    raise Infeasible("the compactor cannot reach the push")
                before = orient[compactor]
                closing = (np.array([0.0, 1.0, 0.0]) if rng.random() < 0.5
                           else np.array([0.0, 0.0, 1.0]))
                orient[compactor] = _horizontal(direction, closing)
                holder_lifted = feats[n + g, POS] + [0.0, 0.0, LIFT]
                at = feats.copy()
                at[n + g, POS] = holder_lifted
                at[n + compactor, POS] = contact
                at[n + compactor, 8] = CLOSED
                out.keyframes.append(Keyframe(_widen(at, orient, n), CONTACT, compactor, o))
                pushed = at.copy()
                pushed[o, POS] = p_o + delta
                pushed[n + compactor, POS] = end
                out.keyframes.append(Keyframe(_widen(pushed, orient, n), PUSH, compactor, o))
                orient[compactor] = before
                shift[o] += delta
                out.final_pose[o] = out.final_pose[o].copy()
                out.final_pose[o][:3] += delta
        placed.append(o)
    return out


def scene_for_v2(rng: np.random.Generator, progs, tries: int = 60) -> list[Episode]:
    """`batches._scene_for` for v2 programs: one scene every program in `progs` can run."""
    count = int(rng.choice(len(N_OBJECTS_P), p=N_OBJECTS_P)) + 1
    for _ in range(tries):
        try:
            scene = sample_scene(rng, n=count)
            return [execute_v2(scene, p, rng) for p in progs]
        except Infeasible:
            continue
    raise Infeasible(f"no common feasible scene for {progs}")


__all__ = ["EVENTS_V2", "CONTACT", "PUSH", "TARGET_EVENTS_V2", "ROT", "N_FEATURES_V2",
           "COMPACT_GAP", "execute_v2", "scene_for_v2", "SIZE"]
