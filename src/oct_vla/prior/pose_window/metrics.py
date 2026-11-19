"""Scores for predicted wrist windows (plan 0.4), shared by synthetic and real evaluation.

- ``errors``: translation (mm) and sign-invariant angle (deg) per horizon, moving-wrist
  error (wrists whose target moves > 5 mm over the window) and idle drift (predicted motion
  of a wrist whose target stays within 5 mm).
- ``decision``: which object a window heads to. A pose window has no object id, so a fixed
  geometric rule decides: take the wrist whose predicted window moves furthest; if it moves
  less than ``MIN_MOVE`` the item is *unassignable*; otherwise the object whose
  **horizontal** direction from the wrist's grip point best matches the horizontal
  displacement (cosine), among movables that are not yet in a hand. Horizontal, because
  every candidate lies below a wrist at its start pose, and the shared vertical drop made
  a 3D cosine pick the right pair only 52 % of the time on the expert's own windows (100 %
  horizontally). Unassignable items stay in every denominator.
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
    ang = np.degrees(Q.angle(pred[..., 3:], target[..., 3:]))
    tgt_move = np.linalg.norm(target[:, -1, :, :3] - target[:, 0, :, :3], axis=-1)   # [N, 2]
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


def decision(window: np.ndarray, query: Table) -> str | None:
    """The track the window heads to, or None (unassignable)."""
    last = query.pose[-1]
    cols = {LEFT_WRIST: query.column("left"), RIGHT_WRIST: query.column("right")}
    move = [np.linalg.norm(window[-1, a, :3] - last[cols[k], :3])
            for a, k in enumerate((LEFT_WRIST, RIGHT_WRIST))]
    arm = int(np.argmax(move))
    if move[arm] < MIN_MOVE:
        return None
    start = _grip(last[cols[(LEFT_WRIST, RIGHT_WRIST)[arm]]])
    disp = _grip(window[-1, arm]) - start
    best, score = None, -np.inf
    for c in range(len(query.kind)):
        if query.kind[c] != MOVABLE:
            continue
        to = last[c, :3] - start
        if np.linalg.norm(to) < 0.03:     # already in (or at) a hand: not a candidate
            continue
        d2, t2 = disp[:2], to[:2]
        cos = float(d2 @ t2 / (np.linalg.norm(d2) * np.linalg.norm(t2) + 1e-9))
        if cos > score:
            best, score = query.tracks[c], cos
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
