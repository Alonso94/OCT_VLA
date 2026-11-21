"""Physical trajectories on one clock: both wrists, apertures and attached objects.

Every tick advances both wrists together at ``schema.HZ``. Positions follow minimum-jerk
profiles and orientations shortest-arc SLERP *within* a phase. An attached object is never
interpolated on its own: its pose is recomputed each tick from the wrist pose and the
fixed wrist->object transform taken at the moment of attachment (rigid carry).

Two ways to drive it:
- ``segment``: open-loop phases with fixed durations;
- ``tick``: one step with explicit poses, for reactive controllers that decide each step
  from the current physical state (``branches.py``).
"""

from __future__ import annotations

import numpy as np

from oct_vla.bpfn import quat
from oct_vla.bpfn.schema import DT, HZ, Episode

#: The wrist convention is RoboTwin's recorded EEF frame. Its **tool (approach) axis is
#: local x**, and a grasped object's centre sits ``TOOL_OFFSET`` along it: measured on the
#: v4 recordings at 18 grasp onsets, 0.1432 m (std 3 mm), with local x = (0, 0, -1) and
#: local y = (-1, 0, 0) in world for a yaw-0 top-down grasp. (The shelf adapter's 0.128 m
#: along world z was a different quantity.) ``scripts/bpfn_calibrate_tool.py`` re-measures it.
TOOL_AXIS = np.array([1.0, 0.0, 0.0])
TOOL_OFFSET = 0.143
OPEN, CLOSED = 1.0, 0.0


def orientation(tool_dir, z_hint) -> np.ndarray:
    """The wrist quaternion whose tool axis (local x) points along ``tool_dir`` and whose
    local z is ``z_hint`` made orthogonal to it."""
    x = np.asarray(tool_dir, float)
    x = x / np.linalg.norm(x)
    z = np.asarray(z_hint, float) - (np.asarray(z_hint, float) @ x) * x
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    return quat.from_matrix(np.stack([x, y, z], axis=1))


#: A yaw-0 top-down grasp in the recorded convention.
TOP_DOWN = orientation((0.0, 0.0, -1.0), (0.0, 1.0, 0.0))


def min_jerk(n: int) -> np.ndarray:
    """Progress s in (0, 1] at ticks 1..n of a minimum-jerk profile (s = 1 at tick n)."""
    tau = np.arange(1, n + 1) / n
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def ticks(duration: float) -> int:
    return max(1, int(round(duration * HZ)))


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pose a * b (apply b in a's frame)."""
    return np.concatenate([a[:3] + quat.rotate(a[3:], b[:3]), quat.multiply(a[3:], b[3:])])


def relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a^-1 * b: b expressed in a's frame."""
    qi = quat.inverse(a[3:])
    return np.concatenate([quat.rotate(qi, b[:3] - a[:3]), quat.multiply(qi, b[3:])])


def invert(a: np.ndarray) -> np.ndarray:
    """The inverse pose a^-1."""
    qi = quat.inverse(a[3:])
    return np.concatenate([-quat.rotate(qi, a[:3]), qi])


def grip_point(wrist: np.ndarray) -> np.ndarray:
    """Where a grasped object's centre sits, for wrist pose(s) ``[..., 7]``."""
    wrist = np.asarray(wrist, float)
    return wrist[..., :3] + quat.rotate(wrist[..., 3:], TOOL_AXIS * TOOL_OFFSET)


def wrist_for_grip(point, orientation) -> np.ndarray:
    """The wrist pose whose grip point is ``point`` with tool orientation ``orientation``."""
    q = quat.normalize(np.asarray(orientation, float))
    return np.concatenate([np.asarray(point, float) - quat.rotate(q, TOOL_AXIS * TOOL_OFFSET), q])


class Builder:
    def __init__(self, wrist0: np.ndarray, objects0: np.ndarray,
                 aperture0=(OPEN, OPEN)) -> None:
        self.wrist = [np.array(wrist0, float)]
        self.aperture = [np.array(aperture0, float)]
        self.objects = [np.array(objects0, float)]
        #: arm index -> (object index, wrist->object transform)
        self.attached: dict[int, tuple[int, np.ndarray]] = {}

    @property
    def now(self) -> int:
        """Index of the latest row."""
        return len(self.wrist) - 1

    def attach(self, arm: int, obj: int) -> None:
        for other, (o, _) in self.attached.items():
            if o == obj and other != arm:
                raise ValueError(f"object {obj} is already held by arm {other}; use transfer()")
        self.attached[arm] = (obj, relative(self.wrist[-1][arm], self.objects[-1][obj]))

    def detach(self, arm: int) -> None:
        self.attached.pop(arm)

    def transfer(self, giver: int, receiver: int) -> None:
        """Hand the giver's object to the receiver, keeping the receiver's current grasp."""
        obj, _ = self.attached.pop(giver)
        self.attached[receiver] = (obj, relative(self.wrist[-1][receiver], self.objects[-1][obj]))

    def tick(self, wrist: np.ndarray, aperture) -> None:
        wrist = np.array(wrist, float)
        wrist[:, 3:] = quat.normalize(wrist[:, 3:])
        objects = self.objects[-1].copy()
        for arm, (obj, rel) in self.attached.items():
            objects[obj] = compose(wrist[arm], rel)
        self.wrist.append(wrist)
        self.aperture.append(np.array(aperture, float))
        self.objects.append(objects)

    def segment(self, duration: float, targets: dict | None = None,
                aperture: dict | None = None) -> None:
        """Move the arms in ``targets`` (arm -> pose7) and ramp ``aperture`` (arm -> value)
        over ``duration``; arms not named hold still."""
        targets, aperture = targets or {}, aperture or {}
        n = ticks(duration)
        start_w, start_a = self.wrist[-1].copy(), self.aperture[-1].copy()
        s = min_jerk(n)
        paths = {}
        for arm, goal in targets.items():
            goal = np.asarray(goal, float)
            pos = start_w[arm, :3] + s[:, None] * (goal[:3] - start_w[arm, :3])
            rot = quat.slerp(start_w[arm, 3:], goal[3:], s)
            paths[arm] = np.concatenate([pos, rot], axis=1)
        lin = np.arange(1, n + 1) / n
        for k in range(n):
            w = start_w.copy()
            for arm, path in paths.items():
                w[arm] = path[k]
            a = start_a.copy()
            for arm, value in aperture.items():
                a[arm] = start_a[arm] + lin[k] * (value - start_a[arm])
            self.tick(w, a)

    def hold(self, duration: float) -> None:
        self.segment(duration)

    def episode(self, kind, colour, category, size, track_ids, annotations=None) -> Episode:
        wrist = np.stack(self.wrist)
        wrist[..., 3:] = np.stack([quat.continuous(wrist[:, a, 3:]) for a in range(2)], axis=1)
        objects = np.stack(self.objects)
        objects[..., 3:] = np.stack([quat.continuous(objects[:, j, 3:])
                                     for j in range(objects.shape[1])], axis=1)
        T = len(wrist)
        return Episode(np.arange(T) * DT, wrist, np.stack(self.aperture), objects,
                       np.asarray(kind, np.int8), np.asarray(colour, np.int8),
                       np.asarray(category, np.int8), np.asarray(size, float),
                       tuple(track_ids), dict(annotations or {}))
