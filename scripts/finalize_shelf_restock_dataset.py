#!/usr/bin/env python
"""Export all canonical clips below one collection root as a LeRobot dataset."""

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
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--object-tokens", action="store_true")
    parser.add_argument(
        "--control-space",
        default="cartesian",
        choices=["cartesian", "joint"],
        help="'joint' makes observation.state the measured joint configuration and "
        "action the next one, so the policy is proprioceptive in the space it "
        "commands and no inverse kinematics sits between them.",
    )
    args = parser.parse_args()
    sources = sorted(path.parent for path in args.canonical_root.rglob("episode.json"))
    spec = ObjectTokenSpec() if args.object_tokens else None
    report = export_episodes(
        sources,
        args.output,
        repo_id=args.repo_id,
        object_token_spec=spec,
        control_space=args.control_space,
    )
    print(f"exported {len(report.exported)} episodes to {report.output}")
    for source, reason in report.skipped:
        print(f"skipped {source}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
