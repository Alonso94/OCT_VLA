"""Deterministic CPU fixtures for the first slice (plan stage 0, acceptance).

**Colour fixture.** Two independently laid-out supports, one restocking every red object
and one every blue object (four physical objects each: three eligible, one distractor of
the other colour), and one mixed-colour query. Checks:
- the expected first-pick sets for the query are disjoint between the two supports;
- they survive independent column shuffles of support and query;
- they are recovered under one semantic relabeling shared by support and query, and
  broken by independent relabelings (the negative control).

"Expected" is computed by an evaluator-side reader that sees only model-visible fields
(poses, kinds, colours): it is a test that the support *contains* the answer, not a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oct_vla.bpfn import quat
from oct_vla.bpfn.schema import (COLOURS, MOVABLE, Episode, Table, category_id, colour_id, query_table,
                     relabel, semantic_relabeling, shuffle_columns, support_table)
from oct_vla.bpfn.data.syn_data.trajectory import CLOSED, OPEN, TOP_DOWN, Builder, grip_point, wrist_for_grip

# Synthetic shelf geometry (workcell metres), close to the RoboTwin shelf task.
LOWER_Y, LOWER_TOP = -0.25, 0.745
UPPER_Y, UPPER_TOP = -0.06, 0.935
SLOT_X0, SLOT_PITCH = -0.20, 0.09
BOX = np.array([0.05, 0.05, 0.07])
LEFT_HOME = np.concatenate([[-0.30, -0.12, 1.20], TOP_DOWN])
RIGHT_HOME = np.concatenate([[0.30, -0.12, 1.20], TOP_DOWN])
LIFT = 0.10
MOVED = 0.05        # m: a movable that ends this far from its start was moved
LOWER_BAND = 0.85   # m: an object centre below this is still on the lower shelf


def _object(x: float, y: float, top: float, size=BOX) -> np.ndarray:
    return np.array([x, y, top + size[2] / 2, *quat.IDENTITY])


def restock_episode(xs, colours, rule_colour: str, arm: int = 0, seed: int = 0) -> Episode:
    """Carry every object of ``rule_colour``, leftmost first, to consecutive upper-shelf
    slots with one arm; the other arm idles; then a final dwell."""
    rng = np.random.default_rng(seed)
    n = len(xs)
    objects0 = np.stack([_object(x, LOWER_Y, LOWER_TOP) for x in xs])
    b = Builder(np.stack([LEFT_HOME, RIGHT_HOME]), objects0)
    order = [j for j in np.argsort(xs) if colours[j] == rule_colour]
    other = 1 - arm
    for slot, j in enumerate(order):
        obj = b.objects[-1][j]
        pre = wrist_for_grip(obj[:3] + [0, 0, LIFT], TOP_DOWN)
        grasp = wrist_for_grip(obj[:3], TOP_DOWN)
        dest = np.array([SLOT_X0 + slot * SLOT_PITCH, UPPER_Y, UPPER_TOP + BOX[2] / 2])
        jitter = rng.uniform(0.9, 1.1)
        b.segment(1.0 * jitter, {arm: pre})
        b.segment(0.5, {arm: grasp})
        b.segment(0.3, aperture={arm: CLOSED})
        b.attach(arm, j)
        b.segment(0.5, {arm: pre})
        b.segment(1.2 * jitter, {arm: wrist_for_grip(dest + [0, 0, LIFT], TOP_DOWN)})
        b.segment(0.5, {arm: wrist_for_grip(dest, TOP_DOWN)})
        b.segment(0.3, aperture={arm: OPEN})
        b.detach(arm)
        b.segment(0.5, {arm: wrist_for_grip(dest + [0, 0, LIFT], TOP_DOWN)})
    b.segment(1.0, {arm: (LEFT_HOME, RIGHT_HOME)[arm], other: (LEFT_HOME, RIGHT_HOME)[other]})
    b.hold(1.0)
    return b.episode(
        [MOVABLE] * n, [colour_id(c) for c in colours], [category_id("box")] * n,
        np.tile(BOX, (n, 1)), [f"obj_{j}" for j in range(n)],
        {"rule": f"restock_{rule_colour}", "eligible": [f"obj_{j}" for j in order],
         "physical": n, "eligible_count": len(order)})


# --- the evaluator-side reader (model-visible fields only) ---------------------------

def _movables(table: Table) -> list[int]:
    return [c for c in range(len(table.kind)) if table.kind[c] == MOVABLE]


def moved_columns(table: Table) -> list[int]:
    return [c for c in _movables(table)
            if np.linalg.norm(table.pose[-1, c, :3] - table.pose[0, c, :3]) > MOVED]


def demonstrated_colour(support: Table) -> int | None:
    """The colour id every moved object shares and no unmoved object has; None if the
    support does not single one out."""
    moved = moved_columns(support)
    kept = [c for c in _movables(support) if c not in moved]
    colours = {int(support.colour[c]) for c in moved}
    if len(colours) != 1:
        return None
    (c,) = colours
    return None if any(int(support.colour[k]) == c for k in kept) else c


def eligible_first_picks(query: Table, colour: int | None) -> frozenset[str]:
    """Tracks of query objects still on the lower shelf with the demonstrated colour."""
    if colour is None:
        return frozenset()
    return frozenset(query.tracks[c] for c in _movables(query)
                     if int(query.colour[c]) == colour and query.pose[-1, c, 2] < LOWER_BAND)


def colour_fixture(seed: int = 0) -> dict:
    """The two supports, the query, and every check's outcome."""
    rng = np.random.default_rng(seed)
    supports = {
        "red": restock_episode([-0.32, -0.14, 0.04, 0.22], ["red", "blue", "red", "red"],
                               "red", seed=seed + 1),
        "blue": restock_episode([-0.28, -0.08, 0.10, 0.27], ["blue", "blue", "red", "blue"],
                                "blue", seed=seed + 2),
    }
    q_colours = ["blue", "red", "red", "blue"]
    query = restock_episode([-0.30, -0.10, 0.08, 0.26], q_colours, "red", seed=seed + 3)
    qt = query_table(query, t=0, history=16)
    sets, shuffled, shared, independent = {}, {}, {}, {}
    for name, ep in supports.items():
        st = support_table(ep)
        sets[name] = eligible_first_picks(qt, demonstrated_colour(st))
        st_s, _ = shuffle_columns(st, rng)
        qt_s, _ = shuffle_columns(qt, rng)
        shuffled[name] = eligible_first_picks(qt_s, demonstrated_colour(st_s))
        m = semantic_relabeling(rng)
        shared[name] = eligible_first_picks(relabel(qt, m), demonstrated_colour(relabel(st, m)))
        # Negative control: the query relabelled independently. Draw until the two
        # mappings disagree on the tested colours, so the control cannot pass by luck.
        while True:
            m2 = semantic_relabeling(rng)
            if any(m2["colour"][colour_id(c)] != m["colour"][colour_id(c)] for c in ("red", "blue")):
                break
        independent[name] = eligible_first_picks(relabel(qt, m2),
                                                 demonstrated_colour(relabel(st, m)))
    expected = {name: frozenset(f"obj_{j}" for j, c in enumerate(q_colours) if c == name)
                for name in supports}
    return {"supports": supports, "query": query, "query_colours": q_colours,
            "expected": expected, "sets": sets, "shuffled": shuffled, "shared": shared,
            "independent": independent,
            "checks": {
                "support_reads_its_colour": all(sets[n] == expected[n] for n in supports),
                "swap_changes_set": sets["red"].isdisjoint(sets["blue"]) and all(sets.values()),
                "column_shuffle_invariant": shuffled == sets,
                "shared_relabel_recovers": shared == sets,
                "independent_relabel_breaks": all(independent[n] != sets[n] for n in supports),
            }}


__all__ = ["COLOURS", "colour_fixture", "demonstrated_colour", "eligible_first_picks",
           "grip_point", "restock_episode"]


def acceptance_report(out: Path, seed: int = 0) -> dict:
    """Runs both first-slice fixtures, writes their figures and ``fixtures.json`` to ``out``."""
    from oct_vla.bpfn.benchmark_tools import render
    from oct_vla.bpfn.data.syn_data import branches

    colour = colour_fixture(seed=seed)
    report = {"colour": {"checks": colour["checks"],
                         "first_picks": {k: sorted(v) for k, v in colour["sets"].items()}},
              "handover": {}}
    written = render.colour_fixture_figure(colour, out / "colour_fixture")
    for giver, name in ((0, "left_gives"), (1, "right_gives")):
        episodes = branches.family(giver=giver)
        checks = branches.handover_checks(episodes)
        report["handover"][name] = {"cue_row": checks["cue"], "extra_hold_s": checks["extra_hold_s"],
                                    "checks": checks["checks"]}
        written += render.handover_fixture_figure(episodes, checks, out / f"handover_fixture_{name}")
    handover_ok = all(all(v["checks"].values()) for v in report["handover"].values())
    report["passed"] = all(colour["checks"].values()) and handover_ok
    report["figures"] = [str(p) for p in written]
    out.mkdir(parents=True, exist_ok=True)
    (out / "fixtures.json").write_text(json.dumps(report, indent=1) + "\n")
    return report
