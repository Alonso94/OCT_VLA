#!/usr/bin/env python
"""Audit the interaction corpus the effect-conditioned VLA would train on (read-only).

For each rule's full-run collection it reports episodes, transfers, per-arm grasps and
releases, compaction pushes, arm roles, object sizes, pick orders and seeds shared
across rules (whose two runs must land on the same split side). It also writes a
per-transfer **interaction timeline** (``oct_vla.data.effects.object_timeline``: grasp,
lift, place, release, let_go, push, settled; object poses, EEF positions and the measured
aperture only) next to the frame at which ``ShelfKeyframeOracle`` switches its target
away from the object, and the recorded phase labels (reference only). For both label
contracts it reports how often the hindsight effect's object agrees with the oracle's
target, frame by frame. Nothing is written but the report.

    audit_interaction_corpus.py --rule leftmost=$HPCVAULT/octvla-collection-v4/leftmost/full_run/three_object \\
        --rule rightmost=$HPCVAULT/octvla-collection-v4/rightmost/full_run/three_object \\
        --output $OCTVLA_OUTPUT_ROOT/diagnostics/interaction_audit.json
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


def audited_timeline(p, line):
    """The shared timeline (``data/effects.object_timeline``) plus audit-only measures:
    poses at release and at the end of the object's push, drift before its push, and
    how far later neighbours' pushes moved it again."""
    out = dict(line)
    r, T = out.get("release"), len(p)
    if r is None:
        return out
    end = out.get("push_end")
    if end is not None:
        out["pre_push_drift_m"] = round(float(np.linalg.norm(p[out["push_start"] - 1] - p[r])), 4)
        out["later_nudge_m"] = round(float(np.linalg.norm(p[T - 1] - p[end])), 4)
        out["push_end_pose"] = p[end].round(4).tolist()
    else:
        out["pre_push_drift_m"] = round(float(np.linalg.norm(p[T - 1] - p[r])), 4)
    out["release_pose"] = p[r].round(4).tolist()
    out["final_pose"] = p[T - 1].round(4).tolist()
    out["push_shift_m"] = round(float(np.linalg.norm(p[end if end is not None else T - 1]
                                                    - p[r])), 4)
    return out


def oracle_switches(episode, spec):
    """Frame at which the keyframe oracle's target first moves away from each object."""
    from oct_vla.tasks.shelf_restock.keyframes import ShelfKeyframeOracle

    oracle = ShelfKeyframeOracle(spec)
    for s in episode.samples:
        oracle(s.scene, s.observation.eef)
    targets = [t for _, t in oracle.history]
    oracle_switches.targets = targets
    out = {}
    for t in range(1, len(targets)):
        if targets[t - 1] is not None and targets[t] != targets[t - 1] \
                and targets[t - 1] not in out:
            out[targets[t - 1]] = t
    return out, oracle.unplannable_steps


def phase_spans(episode, track):
    """First and last frame of each recorded phase while `track` is the target."""
    spans = {}
    for i, s in enumerate(episode.samples):
        if s.context.target_track_id != track:
            continue
        a, _ = spans.get(s.phase, (i, i))
        spans[s.phase] = (a, i)
    return spans


def audit_episode(episode, spec):
    from oct_vla.data.effects import episode_tracks, timelines

    tracks = episode_tracks(episode, spec)
    ids, pos, sizes = tracks.ids, tracks.pos, [tuple(x) for x in tracks.sizes.tolist()]
    lines = {k: audited_timeline(pos[:, ids.index(k)], v)
             for k, v in timelines(tracks).items()}
    switches, unplannable = oracle_switches(episode, spec)
    from oct_vla.data.effects import effect_rows

    agreement = {}
    for contract in ("A", "B"):
        _, _, target = effect_rows(tracks, timelines(tracks), contract)
        agreement[contract] = float(np.mean([a == b for a, b in
                                             zip(target, oracle_switches.targets)]))
    initial_x = {ids[j]: float(pos[0, j, 0]) for j in range(len(ids))}
    order = sorted((v["lift"], k) for k, v in lines.items() if v.get("lift") is not None)
    for k, v in lines.items():
        v["oracle_switch"] = switches.get(k)
        spans = phase_spans(episode, k)
        v["phases"] = {p: list(spans[p]) for p in ("close", "release", "retreat", "park mover",
                                                   "compact push", "park compactor")
                       if p in spans}
    return {"seed": episode.seed, "frames": len(episode.samples),
            "rule": episode.metadata.get("rule"), "pick_order": [k for _, k in order],
            "initial_x": initial_x, "sizes": dict(zip(ids, sizes)), "objects": lines,
            "oracle_unplannable_steps": unplannable, "oracle_agreement": agreement}


def order_name(pick, initial_x):
    by_x = sorted(initial_x, key=initial_x.get)
    if pick == by_x:
        return "leftmost"
    if pick == by_x[::-1]:
        return "rightmost"
    return "other"


def summarise(rows):
    s = {"episodes": len(rows), "transfers": 0, "pushes": 0, "orders": Counter(),
         "movers": Counter(), "pushers": Counter(), "push_shift_cm": [],
         "oracle_lag_vs_release": [], "oracle_lag_vs_clear": [], "oracle_lag_vs_let_go": [], "release_to_push": [], "drift_cm": [], "displaced_unpushed": [],
         "nudged_after_push": [],
         "sizes": [], "frames": [], "missing": Counter(), "unplannable_episodes": 0}
    for r in rows:
        s["frames"].append(r["frames"])
        for c, a in r["oracle_agreement"].items():
            s.setdefault(f"oracle_agreement_{c}", []).append(a)
        s["orders"][order_name(r["pick_order"], r["initial_x"])] += 1
        s["unplannable_episodes"] += r["oracle_unplannable_steps"] > 0
        s["sizes"] += [list(v) for v in r["sizes"].values()]
        for k, v in r["objects"].items():
            for key in ("grasp", "lift", "place", "release", "clear"):
                if v.get(key) is None:
                    s["missing"][key] += 1
            if v.get("release") is None:
                continue
            s["transfers"] += 1
            s["movers"][v["mover"]] += 1
            s["drift_cm"].append(100 * v["pre_push_drift_m"])
            if v.get("push_start") is None and v["pre_push_drift_m"] > 0.02:
                s["displaced_unpushed"].append([r["seed"], k, round(100 * v["pre_push_drift_m"], 1)])
            if (v.get("later_nudge_m") or 0) > 0.02:
                s["nudged_after_push"].append([r["seed"], k, round(100 * v["later_nudge_m"], 1)])
            if v.get("push_start") is not None:
                s["pushes"] += 1
                s["pushers"][v.get("pusher")] += 1
                s["push_shift_cm"].append(100 * v["push_shift_m"])
                s["release_to_push"].append(v["push_start"] - v["release"])
            if v.get("oracle_switch") is not None:
                s["oracle_lag_vs_release"].append(v["oracle_switch"] - v["release"])
                if v.get("let_go") is not None:
                    s["oracle_lag_vs_let_go"].append(v["oracle_switch"] - v["let_go"])
                if v.get("clear") is not None:
                    s["oracle_lag_vs_clear"].append(v["oracle_switch"] - v["clear"])

    def stats(x):
        x = np.asarray(x, float)
        return None if not len(x) else {"n": int(len(x)), "min": float(x.min()),
                                        "median": float(np.median(x)), "max": float(x.max())}

    sizes = np.asarray(s.pop("sizes"))
    return {**{k: (dict(v) if isinstance(v, Counter) else v) for k, v in s.items()
               if k not in ("push_shift_cm", "oracle_lag_vs_release", "oracle_lag_vs_clear",
                            "oracle_lag_vs_let_go",
                            "release_to_push", "frames", "drift_cm")},
            "release_to_push_start_drift_cm": stats(s["drift_cm"]),
            "frames": stats(s["frames"]),
            **{f"oracle_agreement_{c}": stats(s.pop(f"oracle_agreement_{c}", []))
               for c in ("A", "B")}, "push_shift_cm": stats(s["push_shift_cm"]),
            "release_to_push_frames": stats(s["release_to_push"]),
            "oracle_switch_minus_release": stats(s["oracle_lag_vs_release"]),
            "oracle_switch_minus_clear": stats(s["oracle_lag_vs_clear"]),
            "oracle_switch_minus_let_go": stats(s["oracle_lag_vs_let_go"]),
            "size_min": sizes.min(0).round(4).tolist(), "size_max": sizes.max(0).round(4).tolist()}


def main() -> int:
    from oct_vla.data.store import read_episode
    from oct_vla.tasks.shelf_restock.spec import spec_for_rule

    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", action="append", required=True, help="RULE=COLLECTION_ROOT")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--timelines", type=int, default=4, help="Episodes per rule in full")
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_shelf_restock_splits import SPLIT_BLOCKS

    from oct_vla.data import effects

    report = {"rules": {}, "timelines": {}, "constants": {
        k: getattr(effects, k) for k in ("LIFT_TOL", "STEP_TOL", "CLEAR_DIST", "FULL_OPEN_TOL",
                                         "PUSH_GAP", "REST_FRAMES", "PLACE_DZ", "OPENING",
                                         "LEAVING", "RELEASE_DISTANCE")}}
    seeds_by_rule = {}
    for item in args.rule:
        rule, root = item.split("=", 1)
        spec = spec_for_rule(rule)
        rows = []
        for path in sorted(Path(root).glob("seed_*/episode_*")):
            if not (path / "episode.json").exists():
                continue
            episode = read_episode(path)
            if episode.metadata.get("rule", rule) != rule:
                raise SystemExit(f"{path}: rule {episode.metadata.get('rule')} != {rule}")
            rows.append(audit_episode(episode, spec))
            print(f"{rule} seed {episode.seed}: order {rows[-1]['pick_order']}", flush=True)
        seeds_by_rule[rule] = {r["seed"] for r in rows}
        report["rules"][rule] = summarise(rows)
        with_push = [r for r in rows if any(v.get("push_start") is not None
                                            for v in r["objects"].values())]
        report["timelines"][rule] = (with_push + rows)[:args.timelines]
        report["rules"][rule]["seeds"] = sorted(seeds_by_rule[rule])

    rules = list(seeds_by_rule)
    shared = set.intersection(*seeds_by_rule.values()) if len(rules) > 1 else set()
    blocks = SPLIT_BLOCKS.get("three_object", {})

    def side(seed):
        for name, ranges in blocks.items():
            if any(seed in r for r in ranges):
                return name
        return None

    report["shared_seeds"] = {
        "count": len(shared), "seeds": sorted(shared),
        "split_sides": dict(Counter(side(s) for s in shared)),
        "outside_blocks": sorted(s for s in set().union(*seeds_by_rule.values())
                                 if side(s) is None),
        "note": "a shared seed has the same initial scene under both rules, and a different "
                "first object: the contrast Z must resolve; both runs share a split side "
                "because the split is by seed block"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("rules", "shared_seeds")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
