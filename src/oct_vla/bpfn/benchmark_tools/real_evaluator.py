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

from pathlib import Path

import numpy as np

from oct_vla.bpfn.benchmark_tools import metrics
from oct_vla.bpfn.benchmark_tools.evaluator import predict
from oct_vla.bpfn.data import batches
from oct_vla.bpfn.data.real import RealData
from oct_vla.bpfn.schema import (MOVABLE, Episode, colour_id, empty_support, query_table,
                                 support_table, target_window)

LOWER_Z = 0.86     # m: an object centre below this is still on the lower shelf


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
    from oct_vla.data import effects

    movable = [j for j in range(len(ep.track_ids)) if ep.kind[j] == MOVABLE]
    lower_top = LOWER_Z
    upper = ep.objects[:, movable, 2] > lower_top
    tracks = effects.Tracks(tuple(ep.track_ids[j] for j in movable), ep.objects[:, movable, :3],
                      ep.objects[:, movable, 3:], ep.size[movable], ep.wrist[..., :3],
                      ep.aperture, upper)
    lines = effects.timelines(tracks)
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


class RealEvaluator:
    """Scores a model on recorded runs under the same-rule, other-rule and no support.

    Args:
        queries: the corpora whose runs are the queries (rule -> runs).
        supports: where supports come from (None: the query corpora; never the query's own run).
        queries_per_rule: cap on query runs per rule (None: all).
        supports_per_query: supports drawn per (query, condition).
        seed: seed of the support draws.
        device: where the model runs.
    """

    CONDITIONS = ("same", "other", "none")

    def __init__(self, queries: RealData, supports: RealData | None = None,
                 queries_per_rule: int | None = None, supports_per_query: int = 1,
                 s_rows: int = batches.S_ROWS, history: int = batches.HISTORY,
                 horizon: int = batches.HORIZON, seed: int = 0, device: str = "cpu") -> None:
        self.queries = queries
        self.supports = supports or queries
        self.queries_per_rule = queries_per_rule
        self.supports_per_query = supports_per_query
        self.s_rows = s_rows
        self.history = history
        self.horizon = horizon
        self.seed = seed
        self.device = device

    def run(self, model) -> dict:
        rng = np.random.default_rng(self.seed)
        records = []
        for rule in self.queries.rules:
            runs = self.queries.runs[rule]
            if self.queries_per_rule:
                runs = runs[:self.queries_per_rule]
            for query_path in runs:
                query = self.queries.episode(query_path)
                times = decision_times(query, self.horizon)
                for condition in self.CONDITIONS:
                    support_rule = self._support_rule(rule, condition)
                    for support_path in self._support_paths(support_rule, query_path, rng):
                        records += self._records(model, rule, condition, support_rule, query,
                                                 query_path, support_path, times)
        return summarise(records)

    def _support_rule(self, rule: str, condition: str) -> str | None:
        if condition == "none":
            return None
        if condition == "same":
            return rule
        return next(other for other in self.queries.rules if other != rule)

    def _support_paths(self, support_rule: str | None, query_path: Path,
                       rng: np.random.Generator) -> list[Path | None]:
        if support_rule is None:
            return [None]
        candidates = [p for p in self.supports.runs[support_rule] if p != query_path]
        picks = rng.choice(len(candidates), self.supports_per_query, replace=False)
        return [candidates[i] for i in picks]

    def _records(self, model, rule: str, condition: str, support_rule: str | None,
                 query: Episode, query_path: Path, support_path: Path | None,
                 times: list[int]) -> list[dict]:
        support = None if support_path is None else self.supports.episode(support_path)
        items = [_item(support, query, t, self.s_rows, self.history, self.horizon) for t in times]
        s_rows = max(1, max(len(i["support"].time) for i in items))
        windows = predict(model, items, s_rows, self.device)
        records = []
        for t, item, window in zip(times, items, windows):
            left = set(remaining(query, t))
            wanted = expected(query, t, support_rule) if support_rule else set()
            replay = None
            if support is not None:
                replay = metrics.replay(item["support"], item["query"], self.horizon)
            records.append({
                "query": str(query_path), "rule": rule, "condition": condition,
                "support": None if support_path is None else str(support_path),
                "support_rule": support_rule, "t": int(t),
                "decision": metrics.decision(window, item["query"], left),
                "expected": sorted(wanted),
                "expert_decision": metrics.decision(item["target"], item["query"], left),
                "window": window, "target": item["target"], "valid": item["target_valid"],
                "hold": metrics.hold(item["query"], self.horizon), "replay": replay,
            })
        return records


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
            cell["pose"] = metrics.errors(W, T, V)
            cell["hold"] = metrics.errors(np.stack([r["hold"] for r in rs]), T, V)
            reps = [r for r in rs if r["replay"] is not None]
            if reps:
                cell["replay"] = metrics.errors(np.stack([r["replay"] for r in reps]),
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
