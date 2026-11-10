"""Non-learned predictors of the next keyframe, for the bars the PFN must clear.

copy_current
    The acted object stays where it is and the acting gripper does not move. Its pointer
    is uniform over the movables, so its pointer accuracy is 1 / #movables. It is the bar
    for the pose heads (GEMIL: a dense future model lost to copy-last on stationary entities).

retrieval
    A hand-built in-context method: nearest demo by geometric alignment.
    - It describes every object by its normalised ranks in the *initial* layout: x, y,
      volume, and distance to each non-source support ordered by height.
    - It averages the descriptor of the object the demos acted on at the same event
      index, and picks the query's matching candidate.
    - Event type and gripper are majority votes of the demos at that event index.
    - Destinations are taken relative to the matching support (the same height rank).

    It is given the true event index of the query (`events_done`), which a deployed
    model would have to infer, so it is a generous baseline.

Both return the same dict as `KeyframePFN.forward`, in metric units: positions are in
metres, not the model's normalised ones.
"""

from __future__ import annotations

import numpy as np

from .schema import (
    GRASP,
    GRIPPER_TYPES,
    MOVABLE,
    POS,
    SIZE,
    SUPPORT,
    YAW,
)


def copy_current(b: dict) -> dict:
    types, cur = b["q_types"], b["q_feats"][:, 1]
    n_items = len(types)
    movable = types == MOVABLE
    pointer = np.where(movable, 0.0, -1e9)
    obj = _take(cur, b["t_obj"])
    gcol = _gripper_col(types, b["t_grip"])
    grip = _take(cur, gcol)
    return {
        "pointer": pointer,
        "gripper": np.zeros((n_items, 2)),
        "event": np.zeros((n_items, 4)),
        "obj": np.concatenate([obj[:, POS], obj[:, YAW]], -1),
        "grip_pos": grip[:, POS],
        "aperture": grip[:, 8],
        "final": np.concatenate([obj[:, POS], obj[:, YAW]], -1),
    }


def _take(x, idx):
    return x[np.arange(len(x)), idx]


def _gripper_col(types, grip):
    want = np.asarray(GRIPPER_TYPES)[grip]
    return (types == want[:, None]).argmax(-1)


def _ranks(values: np.ndarray) -> np.ndarray:
    n = len(values)
    if n == 1:
        return np.full(1, 0.5)
    return np.argsort(np.argsort(values)) / (n - 1)


def _supports(feats, types):
    """Non-source supports, by height. The source is the support the objects rest on."""
    sup = np.flatnonzero(types == SUPPORT)
    obj = np.flatnonzero(types == MOVABLE)
    z_obj_bottom = (feats[obj, 2] - feats[obj, 7] / 2).min() if len(obj) else -np.inf
    tops = feats[sup, 2] + feats[sup, 7] / 2
    source = sup[np.argmin(np.abs(tops - z_obj_bottom))]
    rest = [s for s in sup if s != source]
    return sorted(rest, key=lambda s: feats[s, 2])


def _descriptors(feats, types):
    obj = np.flatnonzero(types == MOVABLE)
    sups = _supports(feats, types)
    cols = [_ranks(feats[obj, 0]), _ranks(feats[obj, 1]), _ranks(np.prod(feats[obj][:, SIZE], -1))]
    for s in (sups[0], sups[-1]) if sups else ():
        cols.append(_ranks(np.linalg.norm(feats[obj, :2] - feats[s, :2], axis=-1)))
    return obj, np.stack(cols, -1), sups


def retrieval(b: dict) -> dict:
    n_items = len(b["q_types"])
    out = {k: [] for k in ("pointer", "gripper", "event", "obj", "grip_pos", "aperture", "final")}
    for i in range(n_items):
        p = _retrieve_one({k: v[i] for k, v in b.items()})
        for k in out:
            out[k].append(p[k])
    return {k: np.stack(v) for k, v in out.items()}


def _support_desc(feats, sups):
    """(height rank, x rank) of each non-source support."""
    if not sups:
        return np.zeros((0, 2))
    return np.stack([_ranks(feats[sups, 2]), _ranks(feats[sups, 0])], -1)


def _resting_support(feats, sups, k):
    """The non-source support under object k (footprint contains it), or None."""
    for s in sups:
        if (
            np.all(np.abs(feats[k, :2] - feats[s, :2]) <= feats[s, 5:7] / 2 + 1e-6)
            and feats[k, 2] > feats[s, 2]
        ):
            return s
    return None


def _retrieve_one(it: dict) -> dict:
    j = int(it["events_done"])
    types, init, cur = it["q_types"], it["q_feats"][0], it["q_feats"][1]
    q_obj, q_desc, q_sups = _descriptors(init, types)
    q_sdesc = _support_desc(init, q_sups)
    descs, events, grips, apert, rel_grip = [], [], [], [], []
    place, final = [], []  # (support descriptor, offset from it) or absolute positions
    for d in np.flatnonzero(it["demo_valid"]):
        rows = int(it["ctx_rows"][d].sum())
        if j + 1 >= rows:
            continue
        dt = it["ctx_types"][d]
        f0, fn, last = (
            it["ctx_feats"][d, 0],
            it["ctx_feats"][d, j + 1],
            it["ctx_feats"][d, rows - 1],
        )
        acted = [c for c in np.flatnonzero(it["ctx_flag"][d, j + 1]) if dt[c] == MOVABLE]
        if not acted:
            continue
        a = acted[0]
        d_obj, d_desc, d_sups = _descriptors(f0, dt)
        d_sdesc = _support_desc(f0, d_sups)
        descs.append(d_desc[list(d_obj).index(a)])
        events.append(int(it["ctx_event"][d, j + 1]))
        g = int(it["ctx_gripper"][d, j + 1])
        grips.append(g)
        gcol = int(np.flatnonzero(dt == GRIPPER_TYPES[g])[0])
        rel_grip.append(fn[gcol, POS] - fn[a, POS])
        apert.append(fn[gcol, 8])
        for frame, store in ((fn, place), (last, final)):
            s = _resting_support(frame, d_sups, a)
            if s is None:
                store.append((None, frame[a, POS]))
            else:
                store.append((d_sdesc[d_sups.index(s)], frame[a, POS] - frame[s, POS]))
    pointer = np.where(types == MOVABLE, 0.0, -1e9)
    if not descs:  # no usable demo: fall back to copy
        obj = cur[q_obj[0]] if len(q_obj) else cur[0]
        z = np.concatenate([obj[POS], obj[YAW]])
        return {
            "pointer": pointer,
            "gripper": np.zeros(2),
            "event": np.zeros(4),
            "obj": z,
            "grip_pos": obj[POS],
            "aperture": 1.0,
            "final": z,
        }
    event = max(set(events), key=events.count)
    grip = max(set(grips), key=grips.count)
    moved = np.linalg.norm(cur[q_obj, :3] - init[q_obj, :3], axis=-1) > 0.02
    if event == GRASP:
        cands = [k for k in range(len(q_obj)) if not moved[k]] or list(range(len(q_obj)))
    else:  # the object being carried
        gcol_q = int(np.flatnonzero(types == GRIPPER_TYPES[grip])[0])
        cands = [int(np.argmin(np.linalg.norm(cur[q_obj, :3] - cur[gcol_q, :3], axis=-1)))]
    target_desc = np.mean(descs, 0)
    best = min(cands, key=lambda k: np.linalg.norm(q_desc[k] - target_desc))
    pointer[q_obj[best]] = 1.0
    o = cur[q_obj[best]]

    def locate(store):
        """Support-relative when most demos rest the object on a support, else absolute
        (a handover point is in free space)."""
        anchored = [(sd, off) for sd, off in store if sd is not None]
        if anchored and q_sups and 2 * len(anchored) >= len(store):
            sd = np.mean([sd for sd, _ in anchored], 0)
            s = q_sups[int(np.argmin(np.linalg.norm(q_sdesc - sd, axis=-1)))]
            return cur[s, POS] + np.mean([off for _, off in anchored], 0)
        free = [off for sd, off in store if sd is None]
        return np.mean(free, 0) if free else o[POS]

    obj_pos = o[POS] if event == GRASP else locate(place)
    yaw = o[YAW] if event == GRASP else np.array([0.0, 1.0])
    return {
        "pointer": pointer,
        "gripper": np.eye(2)[grip],
        "event": np.eye(4)[event - 1],
        "obj": np.concatenate([obj_pos, yaw]),
        "grip_pos": obj_pos + np.mean(rel_grip, 0),
        "aperture": float(np.mean(apert)),
        "final": np.concatenate([locate(final), [0.0, 1.0]]),
    }
