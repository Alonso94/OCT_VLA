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

``--served`` scores each file against the rule its effect source served instead
(``served``: S_pick over all scenes, S_order, S_joint = success in that order).

``--paired A.json B.json`` scores counterfactual control: the same checkpoint on the
same scenes, served two different rules. ``S_paired`` counts a scene only if the first
object lifted is the served rule's under *both* (a scene with no lift is a miss); also
reported, how often the swap changes the first pick at all.

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


def served(episodes: list[dict], rule: str) -> dict:
    """Against the rule the effect source served: the gate metrics of plan rev. 11.

    - ``S_pick``: the served rule's first object is the first lifted, over *all* scored
      scenes (a scene with no lift counts as a miss), and conditional on a lift;
    - ``lift_rate``; ``S_task``: every object restocked (the shelf's own success);
    - ``S_order``: every object lifted, in the served rule's order;
    - ``S_joint`` = S_task and S_order: the task done the way the effect asked.
    """
    usable = [e for e in episodes if e.get("initial_x") and e.get("events")]
    n = len(usable)
    pick = order = joint = lifted = task = 0
    for e in usable:
        want, got = rule_order(e["initial_x"], rule), lift_order(e["events"])
        lifted += bool(got)
        pick += bool(got) and got[0] == want[0]
        in_order = got == want
        order += in_order
        task += bool(e.get("success"))
        joint += in_order and bool(e.get("success"))
    return {"rule": rule, "scenes": n, "lift_rate": lifted / n if n else None,
            "S_pick": pick / n if n else None,
            "S_pick_given_lift": pick / lifted if lifted else None,
            "S_task": task / n if n else None, "S_order": order / n if n else None,
            "S_joint": joint / n if n else None,
            "counts": {"pick": pick, "lifted": lifted, "task": task, "order": order,
                       "joint": joint}}


def paired(a: dict, b: dict, rule_a: str | None = None, rule_b: str | None = None) -> dict:
    """Two reports of one checkpoint on the same scenes, each served its own rule."""
    rule_a = rule_a or served_rule(a)
    rule_b = rule_b or served_rule(b)
    if rule_a is None or rule_b is None or rule_a == rule_b:
        raise ValueError(f"paired scoring needs two different served rules, got {rule_a}, {rule_b}")
    if a.get("checkpoint") != b.get("checkpoint"):
        raise ValueError("paired scoring compares one checkpoint under two effects")
    ea = {e["seed"]: e for e in a["episodes"] if e.get("initial_x") and e.get("events")}
    eb = {e["seed"]: e for e in b["episodes"] if e.get("initial_x") and e.get("events")}
    if set(ea) != set(eb):
        raise ValueError(f"the two reports cover different scenes: {sorted(set(ea) ^ set(eb))}")
    both = changed = 0
    for seed, x in ea.items():
        y = eb[seed]
        if x["initial_x"] != y["initial_x"]:
            raise ValueError(f"scene {seed} differs between the two reports")
        gx, gy = lift_order(x["events"]), lift_order(y["events"])
        ok_x = bool(gx) and gx[0] == rule_order(x["initial_x"], rule_a)[0]
        ok_y = bool(gy) and gy[0] == rule_order(y["initial_x"], rule_b)[0]
        both += ok_x and ok_y
        changed += bool(gx) and bool(gy) and gx[0] != gy[0]
    n = len(ea)
    return {"rules": [rule_a, rule_b], "scenes": n,
            "S_paired": both / n if n else None,
            "first_pick_changed": changed / n if n else None,
            "a": served(a["episodes"], rule_a), "b": served(b["episodes"], rule_b),
            "counts": {"paired": both, "changed": changed}}


def served_rule(report: dict) -> str | None:
    """The rule an evaluation served: --oracle-rule, else None (the scene's own)."""
    return (report.get("future_intervention") or {}).get("oracle_rule")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    ap.add_argument("--served", nargs="?", const="auto", default=None,
                    help="Score against the served rule (S_pick, S_order, S_joint): the "
                         "file's --oracle-rule with 'auto', or the rule named here")
    ap.add_argument("--paired", action="store_true",
                    help="Two files: one checkpoint served two rules on the same scenes")
    args = ap.parse_args()
    if args.paired:
        if len(args.files) != 2:
            raise SystemExit("--paired takes exactly two files")
        a, b = (json.loads(p.read_text()) for p in args.files)
        r = paired(a, b)
        if args.json:
            print(json.dumps(r, indent=1))
            return 0
        fmt = lambda v: "   -" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"{r['rules'][0]} vs {r['rules'][1]} on {r['scenes']} scenes: "
              f"S_paired {fmt(r['S_paired'])}, first pick changed {fmt(r['first_pick_changed'])}")
        for side in ("a", "b"):
            x = r[side]
            print(f"  {x['rule']:>9s}: lift {fmt(x['lift_rate'])} S_pick {fmt(x['S_pick'])} "
                  f"S_task {fmt(x['S_task'])} S_order {fmt(x['S_order'])} S_joint {fmt(x['S_joint'])}")
        return 0
    if args.served is not None:
        rows = {}
        for path in args.files:
            data = json.loads(path.read_text())
            rule = served_rule(data) if args.served == "auto" else args.served
            if rule is None:
                raise SystemExit(f"{path.name} served no other rule; name one: --served RULE")
            rows[path.name] = served(data["episodes"], rule)
        if args.json:
            print(json.dumps(rows, indent=1))
            return 0
        print(f"{'file':44s} {'rule':>9s} {'n':>3s} {'lift':>5s} {'S_pick':>7s} "
              f"{'|lift':>6s} {'S_task':>7s} {'S_order':>8s} {'S_joint':>8s}")
        fmt = lambda v: "   -" if v is None else f"{v:.2f}"  # noqa: E731
        for name, r in rows.items():
            print(f"{name:44s} {r['rule']:>9s} {r['scenes']:3d} {fmt(r['lift_rate']):>5s} "
                  f"{fmt(r['S_pick']):>7s} {fmt(r['S_pick_given_lift']):>6s} "
                  f"{fmt(r['S_task']):>7s} {fmt(r['S_order']):>8s} {fmt(r['S_joint']):>8s}")
        return 0
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
