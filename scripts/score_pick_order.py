#!/usr/bin/env python
"""Which order did the robot take the objects in? Scored against each transfer rule.

For the PFN-VLA interventions (``--oracle-rule``, ``--future-support-pool``), task success
alone cannot say whether the policy *followed* the keyframe or the demonstrations, since
any order restocks the shelf. This reads rollout JSONs from ``eval_shelf_restock_policy.py``.
It needs two fields:
- ``initial_x``: where each object started;
- ``events[track].lifted_step``: when each object was first lifted.

From these it reports, per file and per rule, how often the first object lifted is
that rule's first object, and how often the whole lift order is that rule's order.

Rules, as `ShelfRestockManager` applies them to the objects still on the lower shelf:
- leftmost: ascending x;
- rightmost: descending x;
- middle: the median of three first, then leftmost of the rest.

Episodes recorded before ``initial_x`` existed are counted and skipped, never guessed.

    score_pick_order.py $OCTVLA_OUTPUT_ROOT/eval/G5F-kv-s100?-seen*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def rule_order(initial_x: dict[str, float], rule: str) -> list[str]:
    ordered = sorted(initial_x, key=initial_x.get)
    if rule == "leftmost":
        return ordered
    if rule == "rightmost":
        return ordered[::-1]
    if rule == "middle":
        out, rest = [], list(ordered)
        while rest:
            pick = rest[len(rest) // 2] if len(rest) % 2 else rest[0]
            out.append(pick)
            rest.remove(pick)
        return out
    raise ValueError(f"unknown rule {rule!r}")


def lift_order(events: dict) -> list[str]:
    lifted = [(e["lifted_step"], track) for track, e in events.items()
              if e.get("lifted_step") is not None]
    return [track for _, track in sorted(lifted)]


def score(episodes: list[dict], rules=("leftmost", "rightmost", "middle")) -> dict:
    usable = [e for e in episodes if e.get("initial_x") and e.get("events")]
    out = {"episodes": len(episodes), "scored": len(usable),
           "skipped_no_initial_x": sum(1 for e in episodes if not e.get("initial_x"))}
    lifted_any = [e for e in usable if lift_order(e["events"])]
    out["lifted_any"] = len(lifted_any)
    for rule in rules:
        first = full = 0
        for e in lifted_any:
            want, got = rule_order(e["initial_x"], rule), lift_order(e["events"])
            first += got[0] == want[0]
            full += got == want[:len(got)] and len(got) == len(want)
        out[rule] = {"first": first, "full_order": full}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    args = ap.parse_args()
    results = {}
    for path in args.files:
        data = json.loads(path.read_text())
        results[path.name] = score(data["episodes"])
    if args.json:
        print(json.dumps(results, indent=1))
        return 0
    print(f"{'file':42s} {'scored':>7s} {'lifted':>7s}  first pick (left/right/middle)"
          "   full order (left/right/middle)")
    for name, r in results.items():
        firsts = "/".join(str(r[k]["first"]) for k in ("leftmost", "rightmost", "middle"))
        fulls = "/".join(str(r[k]["full_order"]) for k in ("leftmost", "rightmost", "middle"))
        note = f"  ({r['skipped_no_initial_x']} without initial_x)" if r["skipped_no_initial_x"] else ""
        print(f"{name:42s} {r['scored']:7d} {r['lifted_any']:7d}  {firsts:>30s}   {fulls:>28s}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
