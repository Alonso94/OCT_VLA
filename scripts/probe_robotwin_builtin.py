#!/usr/bin/env python
"""Collect a few seeds of a RoboTwin built-in and report what came out.

The first thing to run against a new task adapter. It answers, in order: does
the recording override fire at all, does the oracle succeed, and is the
canonical scene what we declared it to be. Each is a separate failure with a
separate cause, so each is reported separately rather than as one success rate.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from oct_vla.robots.robotwin.native import RoboTwinNativePort
    from oct_vla.tasks.robotwin_builtin.adapter import spec_for
    from oct_vla.tasks.robotwin_builtin.collect import BuiltinCollectionError, collect_episode

    lo, _, hi = args.seeds.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    spec = spec_for(args.task)
    # need_plan: the built-in's own oracle relies on RoboTwin's planner.
    port = RoboTwinNativePort(args.robotwin_root, task_name=args.task, need_plan=True)
    assets = args.robotwin_root / "assets"

    ok, reasons, rows = 0, Counter(), []
    for seed in seeds:
        try:
            episode = collect_episode(port, spec, seed=seed, assets_root=assets)
        except BuiltinCollectionError as error:
            reasons[str(error).split(":", 1)[-1].strip()[:60]] += 1
            continue
        except Exception as error:  # noqa: BLE001 - reported, the probe continues
            # The full traceback for the first unexpected failure. A truncated
            # one-line summary is enough for a known discard reason and useless
            # for a bug: "IndexError: list index out of range" names neither the
            # file nor the list, and each guess costs a cluster round trip.
            if not reasons:
                traceback.print_exc()
            reasons[f"{type(error).__name__}: {error}"[:70]] += 1
            continue
        ok += 1
        objects = episode.samples[0].scene.objects
        rows.append({
            "seed": seed,
            "samples": len(episode.samples),
            "seconds": round(episode.samples[-1].timestamp, 2),
            "objects": {o.track_id: {"category": o.category,
                                     "size": [round(v, 4) for v in o.size_xyz]} for o in objects},
        })

    print(f"\ncollected {ok}/{len(seeds)} seeds of {args.task}")
    for reason, count in reasons.most_common():
        print(f"  {count:>3}x  {reason}")
    if rows:
        first = rows[0]
        print(f"\nfirst episode: {first['samples']} samples over {first['seconds']}s")
        for track_id, info in first["objects"].items():
            print(f"  {track_id:12} {info['category']:14} size {info['size']}")
        cats = Counter(o["category"] for r in rows for o in r["objects"].values())
        print(f"\ncategories across {ok} episodes: {dict(cats)}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"collected": ok, "attempted": len(seeds),
                                           "reasons": dict(reasons), "episodes": rows}, indent=2))
        print(f"\nwrote {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
