"""An independent geometric checker for physical episodes (synthetic or recorded).

It shares no logic with the generator: it reads only the trajectory arrays, the static
sizes and the evaluator annotations, and reports violations of physical laws:

- ``moved_without_contact``: an object moved on a row where no closed gripper's grip
  point was at (or right beside) it;
- ``distractor_moved``: an object outside ``annotations["selected"]`` moved at all;
- ``not_resting``: a moved object does not end resting on a fixture's top surface;
- ``overlap``: two objects' final footprints overlap on the same surface;
- ``bad_quaternion``: a non-unit quaternion, or a sign flip between consecutive rows.

A clean generator yields zero violations on every program; the tests hold it to that.
"""

from __future__ import annotations

import numpy as np

from . import quat as Q
from .schema import FIXTURE, MOVABLE, Episode
from .trajectory import grip_point

STILL = 1e-7          # m per row: below this an object has not moved
CONTACT_MARGIN = 0.03  # m beyond the object's half extents a pushing grip point may be
CLOSED = 0.5           # aperture below this is a closed gripper
REST_TOL = 2e-3        # m


def _grip(wrist: np.ndarray) -> np.ndarray:
    return grip_point(wrist)


def _local(obj_pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    return Q.rotate(Q.inverse(obj_pose[3:]), point - obj_pose[:3])


def _footprint(pose, size) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array([[sx, sy, 0] for sx in (-1, 1) for sy in (-1, 1)]) * size / 2
    world = Q.rotate(np.broadcast_to(pose[3:], (4, 4)), corners)[:, :2] + pose[:2]
    return world.min(0), world.max(0)


def check(ep: Episode) -> dict[str, list]:
    out = {k: [] for k in ("moved_without_contact", "distractor_moved", "not_resting",
                           "overlap", "bad_quaternion")}
    movables = [j for j in range(len(ep.track_ids)) if ep.kind[j] == MOVABLE]
    fixtures = [j for j in range(len(ep.track_ids)) if ep.kind[j] == FIXTURE]
    selected = set(ep.annotations.get("selected", []))
    grips = _grip(ep.wrist)                                         # [T, 2, 3]
    for j in movables:
        tid = ep.track_ids[j]
        step = np.linalg.norm(np.diff(ep.objects[:, j, :3], axis=0), axis=1)
        rows = np.flatnonzero(step > STILL) + 1
        if len(rows) and tid not in selected:
            out["distractor_moved"].append((tid, int(rows[0])))
        for t in rows:
            near = False
            for a in range(2):
                if ep.aperture[t, a] >= CLOSED:
                    continue
                local = np.abs(_local(ep.objects[t, j], grips[t, a]))
                if np.all(local <= ep.size[j] / 2 + CONTACT_MARGIN):
                    near = True
            if not near:
                out["moved_without_contact"].append((tid, int(t)))
                break
        if len(rows):
            final = ep.objects[-1, j]
            bottom = final[2] - ep.size[j, 2] / 2
            rests = any(abs(bottom - (ep.objects[-1, f, 2] + ep.size[f, 2] / 2)) < REST_TOL
                        and abs(final[1] - ep.objects[-1, f, 1]) <= ep.size[f, 1] / 2
                        for f in fixtures)
            if fixtures and not rests:
                out["not_resting"].append((tid, float(bottom)))
    final = {j: _footprint(ep.objects[-1, j], ep.size[j]) for j in movables}
    for a in movables:
        for b in movables:
            if a >= b or abs(ep.objects[-1, a, 2] - ep.objects[-1, b, 2]) > 0.05:
                continue
            lo = np.maximum(final[a][0], final[b][0])
            hi = np.minimum(final[a][1], final[b][1])
            if np.all(hi - lo > 1e-4):
                out["overlap"].append((ep.track_ids[a], ep.track_ids[b]))
    for name, arr in (("wrist", ep.wrist), ("objects", ep.objects)):
        q = arr[..., 3:]
        if not np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-5) or \
                np.any((q[1:] * q[:-1]).sum(-1) < 0):
            out["bad_quaternion"].append(name)
    return out


def violations(ep: Episode) -> int:
    return sum(len(v) for v in check(ep).values())
