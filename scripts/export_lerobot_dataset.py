#!/usr/bin/env python
"""Export canonical recordings to a local LeRobot dataset.

Run this with the LeRobot policy environment.  ``--object-tokens`` produces a
separate dataset variant rather than mutating the RGB-only baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from oct_vla.data.lerobot_export import export_episodes
    from oct_vla.data.object_tokens import ObjectTokenSpec

    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path, nargs="+", help="Canonical episode directories")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--object-tokens", action="store_true")
    parser.add_argument("--max-objects", type=int, default=8)
    args = parser.parse_args()
    spec = ObjectTokenSpec(max_objects=args.max_objects) if args.object_tokens else None
    report = export_episodes(
        args.sources, args.output, repo_id=args.repo_id, object_token_spec=spec
    )
    print(f"exported {len(report.exported)} episodes to {report.output}")
    for source, reason in report.skipped:
        print(f"skipped {source}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
