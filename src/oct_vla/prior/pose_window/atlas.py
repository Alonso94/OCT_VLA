"""The validation atlas's first scan: support-dependent selection by a reference marker.

Two matched objects at x = -0.10 and +0.10 m; a reference marker (a fixture column) swept
along x over 31 positions. One support pushes the object *nearest* the marker with the
left arm, an independently laid-out support pushes the *farthest*. Same query, same
initial wrists, for both. The analytic boundary is at x = 0 and flips with the support;
the central cell is a tie and is marked, never scored.

``marker_scan(predict)`` returns one row per (marker position, support): analytic target,
decided target, margin, tie flag, active-wrist error against the analytic window and idle
drift. ``predict(items) -> windows`` is any model or baseline. ``render`` draws it.
"""

from __future__ import annotations

import numpy as np

from oct_vla.bpfn import quat
from oct_vla.bpfn.benchmark_tools import metrics
from oct_vla.bpfn.schema import FIXTURE, MOVABLE, colour_id, category_id, query_table, support_table, target_window
from oct_vla.bpfn.data.syn_data.trajectory import CLOSED, OPEN, TOP_DOWN, Builder, wrist_for_grip

X_OBJ = (-0.10, 0.10)
MARKERS = np.linspace(-0.15, 0.15, 31)
Y0, Z_TOP = -0.25, 0.745
BOX = np.array([0.05, 0.05, 0.07])
PUSH = 0.10
LEFT = np.concatenate([[-0.30, -0.12, 1.15], TOP_DOWN])
RIGHT = np.concatenate([[0.30, -0.12, 1.15], TOP_DOWN])


def _push_episode(xs, marker, rule: str, seed: int = 0):
    """The left arm pushes the object nearest / farthest from the marker forward."""
    rng = np.random.default_rng(seed)
    n = len(xs)
    objs = [np.array([x, Y0, Z_TOP + BOX[2] / 2, *quat.IDENTITY]) for x in xs]
    deck = np.array([0.0, Y0 + 0.06, Z_TOP - 0.01, *quat.IDENTITY])
    mark = np.array([marker, Y0 + 0.12, Z_TOP + 0.005, *quat.IDENTITY])
    b = Builder(np.stack([LEFT, RIGHT]), np.stack(objs + [deck, mark]))
    d = np.abs(np.asarray(xs) - marker)
    j = int(np.argmin(d) if rule == "nearest" else np.argmax(d))
    obj = b.objects[-1][j]
    contact = obj[:3] - np.array([0, BOX[1] / 2 + 0.015, 0])
    q = quat.multiply(quat.from_axis_angle([0, 0, 1], np.pi / 2), TOP_DOWN)
    speed = rng.uniform(0.9, 1.1)
    b.segment(1.0 * speed, {0: wrist_for_grip(contact + [0, 0, 0.10], q)}, aperture={0: CLOSED})
    b.segment(0.5 * speed, {0: wrist_for_grip(contact, q)})
    b.attach(0, j)
    b.segment(1.0 * speed, {0: wrist_for_grip(contact + [0, PUSH, 0], q)})
    b.detach(0)
    b.segment(0.5, {0: wrist_for_grip(contact + [0, PUSH, 0.10], q)}, aperture={0: OPEN})
    b.hold(0.8)
    ids = [f"obj_{k}" for k in range(n)] + ["deck", "marker"]
    return b.episode([MOVABLE] * n + [FIXTURE, FIXTURE], [colour_id("red")] * n + [0, 0],
                     [category_id("box")] * n + [0, 0],
                     np.concatenate([np.tile(BOX, (n, 1)), [[0.84, 0.24, 0.02], [0.03, 0.03, 0.01]]]),
                     ids, {"rule": rule, "selected": [f"obj_{j}"], "events": []}), j


def supports():
    """Three-object supports with several informative selections, independently laid out."""
    near, _ = _push_episode([-0.22, 0.02, 0.18], marker=0.12, rule="nearest", seed=1)
    far, _ = _push_episode([-0.18, -0.02, 0.20], marker=-0.14, rule="farthest", seed=2)
    return {"nearest": near, "farthest": far}


def marker_scan(predict, horizon: int = 16) -> list[dict]:
    """One row per (marker, support). ``predict(list_of_items) -> list of windows``."""
    sups = supports()
    rows, items, keys = [], [], []
    for m in MARKERS:
        for name, sup in sups.items():
            q_ep, want = _push_episode(list(X_OBJ), m, name, seed=3)
            item = {"support": support_table(sup), "query": query_table(q_ep, 0),
                    "target": None, "target_valid": None, "meta": {}}
            item["target"], item["target_valid"] = target_window(q_ep, 0, horizon)
            items.append(item)
            keys.append((float(m), name, q_ep.track_ids[want], item))
    windows = predict(items)
    for (m, name, want, item), w in zip(keys, windows):
        tie = abs(abs(m - X_OBJ[0]) - abs(m - X_OBJ[1])) < 1e-9
        cands = {"obj_0", "obj_1"}
        decided = metrics.decision(w, item["query"], cands)
        err = metrics.errors(w[None], item["target"][None], item["target_valid"][None])
        rows.append({"marker_x": m, "support": name, "analytic": None if tie else want,
                     "decided": decided, "tie": tie,
                     "correct": None if tie else decided == want,
                     "active_mm": err["moving_mm"], "idle_drift_mm": err["idle_drift_mm"]})
    return rows


def expert_predict(items):
    return [it["target"] for it in items]


def render(rows: list[dict], out, title: str = "") -> list:
    """Decision field per support over the marker sweep, analytic boundary and ties."""
    from .render import _plt, _save

    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.2), sharex=True)
    for ax, name in zip(axes, ("nearest", "farthest")):
        rs = [r for r in rows if r["support"] == name]
        x = [r["marker_x"] for r in rs]
        ax.axvline(0.0, color="#999999", lw=0.8, ls="--")
        for r in rs:
            colour = ("#bbbbbb" if r["tie"] else "#1a9850" if r["correct"] else "#d73027")
            y = 0 if r["decided"] == "obj_0" else 1 if r["decided"] == "obj_1" else 0.5
            ax.scatter(r["marker_x"], y, color=colour, s=18, zorder=3)
            if r["analytic"] is not None:
                ax.scatter(r["marker_x"], 0 if r["analytic"] == "obj_0" else 1, marker="_",
                           color="#333333", s=60, zorder=2)
        ax.set_yticks([0, 0.5, 1], ["obj at -0.10", "unassignable", "obj at +0.10"], fontsize=7)
        ax.set_title(f"support: push the {name} object to the marker", fontsize=8)
        acc = [r["correct"] for r in rs if r["correct"] is not None]
        ax.text(0.01, 0.5, f"correct {np.mean(acc):.2f} (n={len(acc)}, tie excluded)",
                transform=ax.transAxes, ha="left", va="center", fontsize=7)
        ax.set_xlim(min(x) - 0.01, max(x) + 0.01)
    axes[-1].set_xlabel("marker x (m)  ·  dash: analytic target  ·  dot: decided (green right, red wrong, grey tie)",
                        fontsize=7)
    fig.suptitle(title or "Marker scan: the decision boundary must flip with the support", fontsize=8)
    paths = _save(fig, out)
    plt.close(fig)
    return paths
