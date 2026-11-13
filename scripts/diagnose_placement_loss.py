#!/usr/bin/env python
"""Why do placed objects end up off the shelf? Loss attribution from rollout traces.

The rollout outcome `placed_then_lost` (tasks/shelf_restock/events.py) only says the
object entered the upper-shelf band at some point (that happens during lowering, while
still held) and is not there at the end. From the per-step traces
(`eval_shelf_restock_policy.py --trace-dir`) this script splits it into four cases:

- (a) **never released:** the object never rested in the band with both arms beyond
  `HOLD_RADIUS` of it;
- (b) **fell after release:** released, then it left the band with no arm within
  `CONTACT_RADIUS`;
- (c) **displaced by the compactor:** released, and at the step it started moving the
  right arm (the compactor) was the nearest arm, within `CONTACT_RADIUS`;
- (d) **disturbed by the mover:** released, and at that step the left arm was nearest,
  within `CONTACT_RADIUS`. Reported with and without the left arm carrying another
  object.

Unexplained cases are counted, not guessed.

It also reports **goal validity per transfer index** from the future rows actually fed
(PFN arms): the goal row on the upper shelf, and clear of every other object resting
there.

    diagnose_placement_loss.py --eval $OCTVLA_OUTPUT_ROOT/eval/G4F-kv-s1000-four-trace.json \\
        --traces $OCTVLA_OUTPUT_ROOT/traces/G4F-kv-s1000-four-trace [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: An object within this of an end effector is in hand (events.HOLD_RADIUS, measured).
HOLD_RADIUS = 0.17
#: An arm this close to a resting object at the step it starts to move is in contact.
CONTACT_RADIUS = 0.20
#: A resting object has moved once it is this far from where it came to rest.
MOVED = 0.02
#: Clear space a goal needs from another resting object's centre (two ~5 cm objects).
GOAL_CLEARANCE = 0.05


def classify(objects: np.ndarray, eef: np.ndarray, j: int, contains) -> tuple[str, dict]:
    """Case (a)-(d) of object `j`. objects [T, M, 3], eef [T, 2, 3]."""
    T = len(objects)
    pos = objects[:, j]
    in_band = np.array([contains(tuple(p)) for p in pos])
    dist = np.linalg.norm(eef - pos[:, None, :], axis=-1)  # [T, 2]
    free = (dist > HOLD_RADIUS).all(1)
    rested = np.flatnonzero(in_band & free)
    if not len(rested):
        return "a_never_released", {}
    t_rest = int(rested[0])
    rest = pos[t_rest]
    moved = np.flatnonzero(np.linalg.norm(pos[t_rest:] - rest, axis=-1) > MOVED)
    if not len(moved):
        return "unexplained_no_motion", {"t_rest": t_rest}
    t_move = t_rest + int(moved[0])
    t0 = max(t_move - 1, 0)
    near = int(np.argmin(dist[t0]))
    info = {"t_rest": t_rest, "t_move": t_move, "nearest_arm": ("left", "right")[near],
            "nearest_distance": float(dist[t0, near])}
    if dist[t0, near] > CONTACT_RADIUS:
        return "b_fell_after_release", info
    if near == 1:
        return "c_compactor", info
    others = [k for k in range(objects.shape[1]) if k != j]
    carrying = any(np.linalg.norm(objects[t0, k] - eef[t0, 0]) < HOLD_RADIUS for k in others)
    info["mover_carrying"] = carrying
    return "d_mover_carrying" if carrying else "d_mover_empty", info


def lift_order(events: dict) -> list[str]:
    return [t for _, t in sorted((e["lifted_step"], t) for t, e in events.items()
                                 if e.get("lifted_step") is not None)]


def goal_validity(trace, tracks, events, contains) -> dict[int, list[bool]]:
    """Per transfer index k: is the fed goal (row 2) on the shelf and clear of the others?

    The fed rows are read at the step each k-th object is first lifted."""
    if "future" not in trace.files:
        return {}
    future, mask, objects = trace["future"], trace["future_mask"], trace["objects"]
    out: dict[int, list[bool]] = {}
    for k, track in enumerate(lift_order(events), 1):
        t = events[track]["lifted_step"]
        if t is None or t >= len(future) or not bool(mask[t][2]) or np.isnan(future[t]).any():
            continue
        goal = future[t][2, :3]
        j = tracks.index(track)
        resting = [objects[t, i] for i in range(len(tracks))
                   if i != j and contains(tuple(objects[t, i]))]
        clear = all(np.linalg.norm(goal[:2] - o[:2]) > GOAL_CLEARANCE for o in resting)
        out.setdefault(k, []).append(bool(contains(tuple(goal)) and clear))
    return out


def main() -> int:
    from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, nargs="+", required=True)
    ap.add_argument("--traces", type=Path, nargs="+", required=True,
                    help="Trace directory per --eval file, in the same order")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if len(args.eval) != len(args.traces):
        raise SystemExit("one --traces directory per --eval file")
    contains = DEFAULT_SPEC.upper_shelf.contains

    cases, by_index, validity = Counter(), Counter(), {}
    lost = episodes = mismatched = 0
    for ev_path, tr_dir in zip(args.eval, args.traces):
        for ep in json.loads(ev_path.read_text())["episodes"]:
            path = tr_dir / f"{ep['profile']}_seed{ep['seed']}.npz"
            if not path.exists():
                mismatched += 1
                continue
            episodes += 1
            trace = np.load(path)
            tracks = trace["tracks"].tolist()
            order = lift_order(ep["events"])
            for track, e in ep["events"].items():
                if e["outcome"] != "placed_then_lost":
                    continue
                lost += 1
                case, _ = classify(trace["objects"], trace["eef"], tracks.index(track), contains)
                cases[case] += 1
                k = order.index(track) + 1 if track in order else 0
                by_index[(k, case[0])] += 1
            for k, v in goal_validity(trace, tracks, ep["events"], contains).items():
                validity.setdefault(k, []).extend(v)

    print(f"{episodes} traced episodes ({mismatched} without a trace), {lost} placed-then-lost "
          "objects")
    for case, n in sorted(cases.items()):
        print(f"  {case:24s} {n:4d}  ({n / max(lost, 1):.0%})")
    ks = sorted({k for k, _ in by_index})
    print("by transfer index (a/b/c/d/u):",
          "  ".join(f"#{k}: " + "/".join(str(by_index[(k, c)]) for c in "abcdu") for k in ks))
    if validity:
        print("fed goal valid, by transfer index:",
              "  ".join(f"#{k}: {sum(v)}/{len(v)}" for k, v in sorted(validity.items())))
    if args.json:
        args.json.write_text(json.dumps({
            "episodes": episodes, "lost": lost, "cases": dict(cases),
            "by_index": {f"{k}{c}": n for (k, c), n in by_index.items()},
            "goal_validity": {k: [sum(v), len(v)] for k, v in validity.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
