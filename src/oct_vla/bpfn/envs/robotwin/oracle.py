"""Privileged bridge check: track the server's live per-arm oracle via-points."""

from __future__ import annotations

import numpy as np

from oct_vla.bpfn import quat
from oct_vla.bpfn.schema import Episode

MAX_STEP = 0.015        # m per control step
MAX_TURN = 0.08         # rad per control step


class ExpertWindowController:
    """Privileged bridge check: move each wrist toward its live oracle via-point."""

    privileged = True

    def __call__(self, ep: Episode, obs) -> np.ndarray:
        if obs.arm_targets is None:
            raise RuntimeError("needs the server's arm targets (reset with arm_targets=True)")
        vias, mask = np.asarray(obs.arm_targets), np.asarray(obs.arm_targets_mask, bool)
        cur = ep.wrist[-1].copy()
        out = cur.copy()
        for a in range(2):
            if not mask[a].any():
                continue
            tgt = vias[a][int(np.flatnonzero(mask[a])[0])][:7]
            d = tgt[:3] - cur[a, :3]
            n = np.linalg.norm(d)
            out[a, :3] = cur[a, :3] + (d if n <= MAX_STEP else d / n * MAX_STEP)
            ang = float(quat.angle(cur[a, 3:], tgt[3:]))
            s = 1.0 if ang <= MAX_TURN else MAX_TURN / ang
            out[a, 3:] = quat.slerp(cur[a, 3:], tgt[3:], np.array([s]))[0]
        return out
