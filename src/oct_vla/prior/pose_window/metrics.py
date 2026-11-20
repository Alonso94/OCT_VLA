"""Scores for predicted wrist windows (plan 0.4), shared by synthetic and real evaluation.

- ``errors``: translation (mm) and sign-invariant angle (deg) per horizon, moving-wrist
  error (wrists whose target moves > 5 mm over the window) and idle drift (predicted motion
  of a wrist whose target stays within 5 mm).
- ``decision``: which object a window heads to. A pose window has no object id, so a fixed
  geometric rule decides (relative horizontal distance gain over the candidates still on
  the source shelf; see ``decision``). A window in which no wrist moves ``MIN_MOVE`` is
  *unassignable*, and unassignable items stay in every denominator.
- baselines: ``hold`` (both wrists stay put) and ``replay`` (the support's wrist motion at
  the same elapsed time, as a displacement from the query's current wrists).
"""

from __future__ import annotations

import numpy as np

from . import quat as Q
from .schema import LEFT_WRIST, MOVABLE, RIGHT_WRIST, Table
from .trajectory import grip_point

MIN_MOVE = 0.01          # m: a window that moves less decides nothing
MOVING = 0.005           # m: a target that moves more is a moving wrist


def errors(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict:
    """``pred``/``target`` ``[N, H, 2, 7]``, ``valid`` ``[N, H]``."""
    v = np.broadcast_to(valid[..., None], pred.shape[:-1])
    trans = np.linalg.norm(pred[..., :3] - target[..., :3], axis=-1)
    # Angles only where the target is valid: an invalid horizon (past the episode's end)
    # carries no orientation to compare with.
    ang = np.zeros(v.shape)
    if v.any():
        ang[v] = np.degrees(Q.angle(pred[..., 3:][v], target[..., 3:][v]))
    # Moving vs idle from the first to the *last valid* horizon of each item.
    last = np.maximum(valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1), 0)
    end = target[np.arange(len(target)), last]                                    # [N, 2, 7]
    tgt_move = np.linalg.norm(end[..., :3] - target[:, 0, :, :3], axis=-1)          # [N, 2]
    tgt_move[~valid.any(1)] = 0.0
    moving = (tgt_move > MOVING)[:, None, :] & v
    idle = (tgt_move <= MOVING)[:, None, :] & v
    pred_move = np.linalg.norm(pred[..., :3] - pred[:, :1, :, :3], axis=-1)
    per_h = [float(trans[:, h][v[:, h]].mean() * 1000) if v[:, h].any() else None
             for h in range(pred.shape[1])]
    return {"trans_mm": float(trans[v].mean() * 1000) if v.any() else None,
            "rot_deg": float(ang[v].mean()) if v.any() else None,
            "trans_mm_per_h": per_h,
            "moving_mm": float(trans[moving].mean() * 1000) if moving.any() else None,
            "idle_drift_mm": float(pred_move[idle].mean() * 1000) if idle.any() else None,
            "n": int(v[..., 0].any(-1).sum())}


def _grip(pose7: np.ndarray) -> np.ndarray:
    return grip_point(pose7)


LOWER_Z = 0.86     # m: an object centre below this is still on the source shelf


def candidates(query: Table) -> set[str]:
    """Objects a next pick can name: movables still on the source shelf at the last row
    (placed objects and objects in a hand are not candidates)."""
    last = query.pose[-1]
    return {query.tracks[c] for c in range(len(query.kind))
            if query.kind[c] == MOVABLE and last[c, 2] < LOWER_Z}


def decision(window: np.ndarray, query: Table, cands: set[str] | None = None) -> str | None:
    """The candidate the window heads to, or None (unassignable).

    For each wrist that moves at least ``MIN_MOVE``, and each candidate, the *relative*
    horizontal distance gain (d_start - d_end) / d_start of the wrist's grip point; the
    best (wrist, candidate) wins. Relative gain, because after a placement the arm sits
    20 cm in front of candidates that are 15 cm apart, where heading angles barely differ
    (recorded v4 windows: 100 % with relative gain at H = 16, against 78 % for a heading
    cosine). The window is the only input: no rule, no target id.
    """
    from .trajectory import grip_point as _gp

    last = query.pose[-1]
    cands = candidates(query) if cands is None else cands
    best, score = None, -np.inf
    for a, name in enumerate(("left", "right")):
        c0 = query.column(name)
        if np.linalg.norm(window[-1, a, :3] - last[c0, :3]) < MIN_MOVE:
            continue
        start, end = _gp(last[c0]), _gp(window[-1, a])
        for c in range(len(query.kind)):
            if query.kind[c] != MOVABLE or query.tracks[c] not in cands:
                continue
            d0 = np.linalg.norm((last[c, :3] - start)[:2])
            d1 = np.linalg.norm((last[c, :3] - end)[:2])
            gain = (d0 - d1) / max(d0, 1e-6)
            if gain > score:
                best, score = query.tracks[c], gain
    return best


def hold(query: Table, horizon: int) -> np.ndarray:
    last = query.pose[-1]
    wr = np.stack([last[query.column("left")], last[query.column("right")]])
    return np.broadcast_to(wr, (horizon, 2, 7)).copy()


def replay(support: Table, query: Table, horizon: int) -> np.ndarray:
    """The support's wrists at the same elapsed time as the query's last row, applied as
    displacements (translation only) to the query's current wrists."""
    ts = support.time - support.time[0]
    elapsed = query.time[-1] - query.time[0] if len(query.time) > 1 else 0.0
    i0 = int(np.searchsorted(ts, elapsed))
    out = hold(query, horizon)
    for a, name in enumerate(("left", "right")):
        c = support.column(name)
        base = support.pose[min(i0, len(ts) - 1), c, :3]
        for h in range(horizon):
            i = min(i0 + h + 1, len(ts) - 1)
            out[h, a, :3] += support.pose[i, c, :3] - base
    return out


def paired_both_correct(decisions: dict[int, list[tuple[str | None, str]]]) -> dict:
    """``decisions``: pair id -> [(decided track, correct first pick), x2]."""
    both = sum(all(d == want for d, want in pair) for pair in decisions.values()
               if len(pair) == 2)
    n = sum(len(pair) == 2 for pair in decisions.values())
    unassignable = sum(d is None for pair in decisions.values() for d, _ in pair)
    return {"pairs": n, "both_correct": both / n if n else None,
            "unassignable": unassignable / max(1, 2 * n)}
