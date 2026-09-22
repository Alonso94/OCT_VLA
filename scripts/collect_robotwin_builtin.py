#!/usr/bin/env python
"""Collect demonstrations of a RoboTwin built-in into the canonical store.

One seed per invocation, so a Slurm array is the parallelism and a failed seed
costs only itself. A discarded seed is not retried: the reason it failed does
not depend on how many times it is asked, which is the same rule the
shelf-restock collector follows.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Collection root")
    args = parser.parse_args()

    from oct_vla.data.store import write_episode
    from oct_vla.robots.robotwin.native import RoboTwinNativePort
    from oct_vla.tasks.robotwin_builtin.adapter import spec_for
    from oct_vla.tasks.robotwin_builtin.collect import BuiltinCollectionError, collect_episode

    spec = spec_for(args.task)
    destination = args.output / args.task / f"seed_{args.seed}"
    report = {"seed": args.seed, "task": args.task}

    # need_plan: the built-in's own oracle relies on RoboTwin's planner, unlike
    # ours, which plans every motion itself and replays the result.
    port = RoboTwinNativePort(args.robotwin_root, task_name=args.task, need_plan=True)
    try:
        episode = collect_episode(
            port, spec, seed=args.seed, assets_root=args.robotwin_root / "assets"
        )
    except BuiltinCollectionError as error:
        report |= {"status": "discarded", "detail": str(error)}
    except Exception as error:  # noqa: BLE001 - recorded, the array continues
        traceback.print_exc()
        report |= {"status": "error", "detail": f"{type(error).__name__}: {error}"}
    else:
        path = write_episode(episode, destination / f"episode_{args.seed:04d}")
        categories = sorted(
            {o.category for o in episode.samples[0].scene.objects if o.category}
        )
        report |= {
            "status": "ok",
            "samples": len(episode.samples),
            "seconds": round(episode.samples[-1].timestamp, 2),
            "categories": categories,
            "path": str(path),
        }

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "collection_report.json").write_text(json.dumps([report], indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "ok" else 0  # a discard is not a job failure


if __name__ == "__main__":
    raise SystemExit(main())
