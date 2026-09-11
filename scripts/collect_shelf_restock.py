#!/usr/bin/env python
"""Collect one or more canonical shelf-restock scenes from RoboTwin."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from oct_vla.robots.robotwin.native import RoboTwinNativePort
    from oct_vla.tasks.shelf_restock.collect import collect_dataset
    from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC

    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--profile", choices=("two_object", "three_object", "four_object"), required=True
    )
    parser.add_argument("--seeds", required=True, help="Comma-separated scene seeds")
    args = parser.parse_args()
    # Every profile shares DEFAULT_SPEC and differs only in object_count, so
    # the spec cannot disagree with the scene the task class builds. It is
    # still passed explicitly rather than left to collect_dataset's default:
    # when the profiles did carry their own geometry, omitting it here meant
    # collect_dataset silently fell back to the narrower default, and an
    # object spawned outside that region was invisible to
    # objects_on_lower_shelf() -- the run reported a clean success while never
    # restocking it (confirmed live at seed 1000).
    task_class, spec = {
        "two_object": ("ShelfRestockTwoObjectTask", DEFAULT_SPEC),
        "three_object": ("ShelfRestockTask", DEFAULT_SPEC),
        "four_object": ("ShelfRestockFourObjectTask", DEFAULT_SPEC),
    }[args.profile]
    port = RoboTwinNativePort(
        args.robotwin_root,
        task_name=f"oct_vla.tasks.shelf_restock.robotwin_env:{task_class}",
        task_config="demo_clean",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    reports = collect_dataset(port, args.output, seeds, spec=spec)
    (args.output / "collection_report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
