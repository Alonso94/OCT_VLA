"""Offline evaluation of a pose-window PFN on recorded demonstrations (plan stage 2a).

A query is a recorded run of rule r at a decision time t; the support is one *other*
recorded run, of rule r (``same``), of the other rule (``other``), or absent (``none``).

**Decision times** come from observation only (``data.effects`` timelines): the first
row, and each object's let-go (the mover is clear of a placed object; the next approach
starts). **What counts as correct** is decided by an independent eligible-set scorer,
never by the recorded future, which is ground truth only for the rule actually executed:
for an order rule, the rule's next object among those still on the lower shelf; for a
colour rule, any remaining object of that colour.

**Pose errors** (mm / deg per horizon, moving-wrist error, idle drift) are scored only
under the ``same`` support, against the recorded next measured poses
(``next_measured_pose``). Baselines: hold and time-aligned support replay.

**Paired both-correct**: the same query time under the ``same`` and ``other`` supports,
counted when both decisions are correct and the two eligible sets are disjoint.

**Count transfer** (``count``): queries with a different number of eligible objects
than the support. ``continuation``: after the third let-go with a fourth eligible object
left, the window heads to an eligible object. ``stopping``: after the last eligible
let-go, the window moves less than ``metrics.MIN_MOVE`` (dwell / clear).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oct_vla.bpfn.data import batches
from . import metrics as Me
from .real import read_recording
from oct_vla.bpfn.schema import (MOVABLE, Episode, colour_id, empty_support, query_table, support_table,
                     target_window)

LOWER_Z = 0.86     # m: an object centre below this is still on the lower shelf


@dataclass
class Corpus:
    rule: str
    runs: list[Path]


def find_runs(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).glob("**/seed_*/episode_*") if (p / "episode.json").exists())


def remaining(ep: Episode, t: int) -> list[str]:
    return [ep.track_ids[j] for j in range(len(ep.track_ids))
            if ep.kind[j] == MOVABLE and ep.objects[t, j, 2] < LOWER_Z]


def expected(ep: Episode, t: int, rule: str) -> set[str]:
    """The objects a rule may take next at row t (empty: nothing left to do)."""
    left = remaining(ep, t)
    if not left:
        return set()
    x = {tid: ep.objects[t, ep.track_ids.index(tid), 0] for tid in left}
    if rule == "leftmost":
        return {min(left, key=x.get)}
    if rule == "rightmost":
        return {max(left, key=x.get)}
    if rule.startswith("restock_"):
        c = colour_id(rule.split("_", 1)[1])
        return {tid for tid in left if ep.colour[ep.track_ids.index(tid)] == c}
    raise ValueError(f"no eligible-set scorer for rule {rule!r}")


def decision_times(ep: Episode, horizon: int = batches.HORIZON) -> list[int]:
    """The approach onset after row 0 and after every object's let-go (observation-only
    timelines for the anchors)."""
    from oct_vla.data import effects as E

    movable = [j for j in range(len(ep.track_ids)) if ep.kind[j] == MOVABLE]
    lower_top = LOWER_Z
    upper = ep.objects[:, movable, 2] > lower_top
    tracks = E.Tracks(tuple(ep.track_ids[j] for j in movable), ep.objects[:, movable, :3],
                      ep.objects[:, movable, 3:], ep.size[movable], ep.wrist[..., :3],
                      ep.aperture, upper)
    lines = E.timelines(tracks)
    lets = sorted(v["let_go"] for v in lines.values() if v.get("let_go") is not None)
    anchors = [0] + [int(t) for t in lets if t < ep.T - 1]
    return [t for t in (approach_onset(ep, a, horizon) for a in anchors) if t is not None]


def approach_onset(ep: Episode, anchor: int, horizon: int = batches.HORIZON,
                   min_gain: float = 0.02, search: int = 120) -> int | None:
    """The first row at or after ``anchor`` at which some wrist's next ``horizon`` rows close
    at least ``min_gain`` of horizontal distance to *some* object still on the source shelf:
    where the next approach becomes visible in one window. Evaluator-side and rule-free (it
    reads the recorded future only to *place* the query, never as an input); without it
    most anchors fall in a pause or a retreat that heads nowhere."""
    from oct_vla.bpfn.data.syn_data.trajectory import grip_point

    grips = grip_point(ep.wrist)                                   # [T, 2, 3]
    for t in range(anchor, min(ep.T - horizon - 1, anchor + search)):
        rem = [ep.track_ids.index(x) for x in remaining(ep, t)]
        if not rem:
            return None
        d0 = np.linalg.norm(grips[t, :, None, :2] - ep.objects[t, rem, :2][None], axis=-1)
        d1 = np.linalg.norm(grips[t + horizon, :, None, :2] - ep.objects[t, rem, :2][None], axis=-1)
        if (d0 - d1).max() >= min_gain:
            return t
    return None


def _item(support: Episode | None, query: Episode, t: int, s_rows: int, history: int,
          horizon: int) -> dict:
    if support is None:
        s_tab = empty_support(support_table(query, np.array([0])))   # columns, no rows
    else:
        s_tab = support_table(support, batches.support_rows(support, s_rows))
    q_tab = query_table(query, t, history)
    window, valid = target_window(query, t, horizon)
    return {"support": s_tab, "query": q_tab, "target": window, "target_valid": valid,
            "meta": {}}


def evaluate(model, corpora: dict[str, Corpus], *, conditions=("same", "other", "none"),
             queries_per_rule: int | None = None, supports_per_query: int = 1,
             s_rows: int = batches.S_ROWS, history: int = batches.HISTORY, horizon: int = batches.HORIZON,
             seed: int = 0, device: str = "cpu", support_pool: dict | None = None) -> dict:
    """Score ``model`` on ``corpora`` (rule -> runs used as queries). ``support_pool``
    (rule -> runs) supplies supports; default: the corpora themselves, never the query's
    own run."""
    from .train import predict

    rules = list(corpora)
    pool = support_pool or {r: c.runs for r, c in corpora.items()}
    rng = np.random.default_rng(seed)
    cache: dict[Path, Episode] = {}

    def load(path: Path) -> Episode:
        if path not in cache:
            if len(cache) > 24:            # keep memory bounded on a login node
                cache.clear()
                gc.collect()
            cache[path] = read_recording(path)
        return cache[path]

    records = []
    for rule in rules:
        runs = corpora[rule].runs[:queries_per_rule] if queries_per_rule else corpora[rule].runs
        for qpath in runs:
            query = load(qpath)
            times = decision_times(query, horizon)
            for cond in conditions:
                if cond == "none":
                    sup_paths = [None]
                else:
                    srule = rule if cond == "same" else next(r for r in rules if r != rule)
                    cands = [p for p in pool[srule] if p != qpath]
                    sup_paths = [cands[i] for i in rng.choice(len(cands), supports_per_query,
                                                              replace=False)]
                for spath in sup_paths:
                    support = None if spath is None else load(spath)
                    srule = None if cond == "none" else (rule if cond == "same" else
                                                        next(r for r in rules if r != rule))
                    items = [_item(support, query, t, s_rows, history, horizon) for t in times]
                    windows = predict(model, items,
                                      max(1, max(len(i["support"].time) for i in items)), device)
                    for t, it, w in zip(times, items, windows):
                        want = expected(query, t, srule) if srule else set()
                        records.append({
                            "query": str(qpath), "rule": rule, "condition": cond,
                            "support": None if spath is None else str(spath),
                            "support_rule": srule, "t": int(t),
                            "decision": Me.decision(w, it["query"], set(remaining(query, t))),
                            "expected": sorted(want),
                            "expert_decision": Me.decision(it["target"], it["query"],
                                                           set(remaining(query, t))),
                            "window": w, "target": it["target"], "valid": it["target_valid"],
                            "hold": Me.hold(it["query"], horizon),
                            "replay": (None if support is None else
                                       Me.replay(it["support"], it["query"], horizon)),
                        })
    return summarise(records)


def summarise(records: list[dict]) -> dict:
    out: dict = {"records": len(records)}
    by = {}
    for r in records:
        by.setdefault((r["rule"], r["condition"]), []).append(r)
    for (rule, cond), rs in sorted(by.items()):
        scored = [r for r in rs if r["expected"]]
        acc = (np.mean([r["decision"] in r["expected"] for r in scored]) if scored else None)
        cell = {"n": len(rs), "decisions": len(scored),
                "decision_accuracy": None if acc is None else float(acc),
                "unassignable": float(np.mean([r["decision"] is None for r in scored]))
                if scored else None}
        if cond == "same":
            # Scorer triage: the decision rule applied to the *recorded* windows must
            # recover the executed rule's eligible object, or the scorer is broken.
            cell["expert_decision_accuracy"] = float(np.mean(
                [r["expert_decision"] in r["expected"] for r in scored])) if scored else None
            W = np.stack([r["window"] for r in rs])
            T = np.stack([r["target"] for r in rs])
            V = np.stack([r["valid"] for r in rs])
            cell["pose"] = Me.errors(W, T, V)
            cell["hold"] = Me.errors(np.stack([r["hold"] for r in rs]), T, V)
            reps = [r for r in rs if r["replay"] is not None]
            if reps:
                cell["replay"] = Me.errors(np.stack([r["replay"] for r in reps]),
                                           np.stack([r["target"] for r in reps]),
                                           np.stack([r["valid"] for r in reps]))
        out[f"{rule}/{cond}"] = cell
    # Paired: same query and time under the same- and other-rule supports.
    pairs = {}
    for r in records:
        if r["condition"] in ("same", "other"):
            pairs.setdefault((r["query"], r["t"]), {})[r["condition"]] = r
    both = [p for p in pairs.values() if len(p) == 2 and p["same"]["expected"]
            and p["other"]["expected"]
            and not set(p["same"]["expected"]) & set(p["other"]["expected"])]
    out["paired"] = {"pairs": len(both), "both_correct": float(np.mean(
        [p["same"]["decision"] in p["same"]["expected"]
         and p["other"]["decision"] in p["other"]["expected"] for p in both])) if both else None}
    return out
