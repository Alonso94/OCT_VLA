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
    from oct_vla.robots.robotwin.task_config import require_franka_planner_limits
    from oct_vla.tasks.shelf_restock.collect import collect_dataset
    from oct_vla.tasks.shelf_restock.spec import RULE_INSTRUCTIONS, spec_for_rule

    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--profile", choices=("two_object", "three_object", "four_object"), required=True
    )
    parser.add_argument("--seeds", required=True, help="Comma-separated scene seeds")
    parser.add_argument(
        "--episode-kind", choices=("atomic", "full_run", "paired"), default="atomic",
        help="Store transfer clips (default), the original continuous shelf-emptying "
        "run, or both from the same oracle execution (paired).",
    )
    parser.add_argument(
        "--rule", choices=tuple(RULE_INSTRUCTIONS), default="leftmost",
        help="Transfer order; also selects the instruction recorded with every clip.",
    )
    args = parser.parse_args()
    spec_ = spec_for_rule(args.rule)
    # Refuse before starting the simulator: a corpus planned under other limits
    # collects at a different yield and with slower motions, and nothing downstream
    # would notice. Recorded in every episode so a corpus states what it used.
    limits = require_franka_planner_limits(args.robotwin_root)
    planner_metadata = {f"curobo_{key}": str(value) for key, value in limits.items()}
    # Every profile shares DEFAULT_SPEC and differs only in object_count, so
    # the spec cannot disagree with the scene the task class builds. It is
    # still passed explicitly rather than left to collect_dataset's default:
    # when the profiles did carry their own geometry, omitting it here meant
    # collect_dataset silently fell back to the narrower default, and an
    # object spawned outside that region was invisible to
    # objects_on_lower_shelf() -- the run reported a clean success while never
    # restocking it (confirmed live at seed 1000).
    task_class, spec = {
        "two_object": ("ShelfRestockTwoObjectTask", spec_),
        "three_object": ("ShelfRestockTask", spec_),
        "four_object": ("ShelfRestockFourObjectTask", spec_),
    }[args.profile]
    port = RoboTwinNativePort(
        args.robotwin_root,
        task_name=f"oct_vla.tasks.shelf_restock.robotwin_env:{task_class}",
        task_config="demo_clean",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    reports = collect_dataset(
        port, args.output, seeds, spec=spec, episode_kind=args.episode_kind,
        metadata=planner_metadata,
    )
    (args.output / "collection_report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
