"""Gripper commands for the closed loop: an observable rule, or the server's oracle
(privileged, labelled as such). Every trigger is logged."""

from __future__ import annotations

import numpy as np

from oct_vla.bpfn.benchmark_tools.metrics import LOWER_Z
from oct_vla.bpfn.data.syn_data.trajectory import grip_point
from oct_vla.bpfn.schema import MOVABLE, Episode

CLOSE_RADIUS = 0.02     # m: grip point to object centre
OPEN_HEIGHT = 0.86      # m: an object centre above this has reached the destination band
DWELL = 3               # control steps


class ObservableGripper:
    """Close on dwell at an object, open on dwell at the destination height; logged."""

    def __init__(self) -> None:
        self.closed = [False, False]
        self.dwell = [0, 0]
        self.held: list[int | None] = [None, None]
        self.log: list[dict] = []

    def __call__(self, ep: Episode, step: int) -> np.ndarray:
        last = ep.T - 1
        grips = grip_point(ep.wrist[last])
        for a in range(2):
            movable = [j for j in range(len(ep.track_ids)) if ep.kind[j] == MOVABLE]
            if not self.closed[a]:
                near = [j for j in movable if ep.objects[last, j, 2] < LOWER_Z
                        and np.linalg.norm(ep.objects[last, j, :3] - grips[a]) < CLOSE_RADIUS]
                self.dwell[a] = self.dwell[a] + 1 if near else 0
                if self.dwell[a] >= DWELL:
                    self.closed[a], self.held[a], self.dwell[a] = True, near[0], 0
                    self.log.append({"step": step, "arm": a, "event": "close",
                                     "object": ep.track_ids[near[0]]})
            else:
                j = self.held[a]
                resting = (j is not None and ep.objects[last, j, 2] > OPEN_HEIGHT and last > 0
                           and abs(ep.objects[last, j, 2] - ep.objects[last - 1, j, 2]) < 1e-3)
                self.dwell[a] = self.dwell[a] + 1 if resting else 0
                if self.dwell[a] >= DWELL:
                    self.closed[a], self.held[a], self.dwell[a] = False, None, 0
                    self.log.append({"step": step, "arm": a, "event": "open",
                                     "object": ep.track_ids[j]})
        return np.array([0.0 if c else 1.0 for c in self.closed])


class OracleGripper:
    """Privileged: the server's per-arm target oracle's next gripper command."""

    privileged = True

    def __init__(self) -> None:
        self.log: list[dict] = []
        self.last = np.array([1.0, 1.0])

    def __call__(self, obs, step: int) -> np.ndarray:
        if obs.arm_targets is None:
            raise RuntimeError("OracleGripper needs the server's arm targets (reset with arm_targets=True)")
        vias, mask = np.asarray(obs.arm_targets), np.asarray(obs.arm_targets_mask, bool)
        for a in range(2):
            if mask[a].any():
                g = float(vias[a][int(np.flatnonzero(mask[a])[0])][7])
                if g != self.last[a]:
                    self.log.append({"step": step, "arm": a, "event": "close" if g < 0.5 else "open"})
                self.last[a] = g
        return self.last.copy()
