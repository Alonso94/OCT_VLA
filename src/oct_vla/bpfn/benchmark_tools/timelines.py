"""When each object was let go in a recorded run, from measured state only.

The same rule as ``oct_vla.data.effects.object_timeline`` up to ``let_go`` (the push and
settle events are not needed here); tests/bpfn/test_standalone.py checks they agree.
"""

from __future__ import annotations

import numpy as np

LIFT_TOL = 0.01          # m above the start height: lifted
FULL_OPEN_TOL = 0.003    # aperture within this of its maximum: fully open
REST_FRAMES = 3          # frames of rest for a placement
PLACE_DZ = 0.001         # m per frame: resting
OPENING = 0.01           # aperture opening after placement: released
LEAVING = 0.02           # m the mover moves away after placement: released
RELEASE_DISTANCE = 0.20  # m between mover and object: let go


def _first(cond: np.ndarray, start: int = 0) -> int | None:
    idx = np.flatnonzero(cond[start:])
    return int(idx[0] + start) if len(idx) else None


def let_go(p: np.ndarray, eef: np.ndarray, aperture: np.ndarray, upper: np.ndarray) -> int | None:
    """The let-go row of one object, or None.

    Args:
        p: the object's positions [T, 3].
        eef: both wrists' positions [T, 2, 3].
        aperture: both measured apertures [T, 2].
        upper: whether the object is in the destination band [T].
    """
    T = len(p)
    lift = _first(p[:, 2] > p[0, 2] + LIFT_TOL)
    if lift is None:
        return None
    dist = np.linalg.norm(eef - p[:, None, :], axis=-1)
    side = int(np.argmin(dist[lift]))
    dz = np.r_[0.0, np.abs(np.diff(p[:, 2]))]
    resting = np.zeros(T, bool)
    for t in range(T):
        resting[t] = dz[max(0, t - REST_FRAMES + 1):t + 1].max() < PLACE_DZ
    place = _first(upper & resting, lift)
    if place is None:
        return None
    opening = aperture[:, side] > aperture[place, side] + OPENING
    leaving = dist[:, side] > dist[place, side] + LEAVING
    release = _first(opening | leaving, place)
    if release is None:
        return None
    return _first(upper & (dist[:, side] > RELEASE_DISTANCE), release)
