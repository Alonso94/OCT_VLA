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

from . import quat as Q
from .schema import DT, HZ, Episode

#: EEF -> grip point along the tool z axis (m). The value the shelf adapter uses
#: (``prior/shelf_adapter.EEF_TO_GRIP``); replaced by ``configs/bimanual_pfn/calibration.json``
#: once the tool transform is calibrated (plan 0.4). Uncalibrated until then.
TOOL_OFFSET = 0.128
#: Tool z pointing straight down (180 deg about x): a top-down grasp.
TOP_DOWN = np.array([1.0, 0.0, 0.0, 0.0])
OPEN, CLOSED = 1.0, 0.0


def min_jerk(n: int) -> np.ndarray:
    """Progress s in (0, 1] at ticks 1..n of a minimum-jerk profile (s = 1 at tick n)."""
    tau = np.arange(1, n + 1) / n
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def ticks(duration: float) -> int:
    return max(1, int(round(duration * HZ)))


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pose a * b (apply b in a's frame)."""
    return np.concatenate([a[:3] + Q.rotate(a[3:], b[:3]), Q.multiply(a[3:], b[3:])])


def relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a^-1 * b: b expressed in a's frame."""
    qi = Q.inverse(a[3:])
    return np.concatenate([Q.rotate(qi, b[:3] - a[:3]), Q.multiply(qi, b[3:])])


def invert(a: np.ndarray) -> np.ndarray:
    """The inverse pose a^-1."""
    qi = Q.inverse(a[3:])
    return np.concatenate([-Q.rotate(qi, a[:3]), qi])


def grip_point(wrist: np.ndarray) -> np.ndarray:
    return wrist[:3] + Q.rotate(wrist[3:], np.array([0.0, 0.0, TOOL_OFFSET]))


def wrist_for_grip(point, orientation) -> np.ndarray:
    """The wrist pose whose grip point is ``point`` with tool orientation ``orientation``."""
    orientation = Q.normalize(np.asarray(orientation, float))
    return np.concatenate([np.asarray(point, float)
                           - Q.rotate(orientation, np.array([0.0, 0.0, TOOL_OFFSET])),
                           orientation])


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
        wrist[:, 3:] = Q.normalize(wrist[:, 3:])
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
            rot = Q.slerp(start_w[arm, 3:], goal[3:], s)
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
        wrist[..., 3:] = np.stack([Q.continuous(wrist[:, a, 3:]) for a in range(2)], axis=1)
        objects = np.stack(self.objects)
        objects[..., 3:] = np.stack([Q.continuous(objects[:, j, 3:])
                                     for j in range(objects.shape[1])], axis=1)
        T = len(wrist)
        return Episode(np.arange(T) * DT, wrist, np.stack(self.aperture), objects,
                       np.asarray(kind, np.int8), np.asarray(colour, np.int8),
                       np.asarray(category, np.int8), np.asarray(size, float),
                       tuple(track_ids), dict(annotations or {}))
