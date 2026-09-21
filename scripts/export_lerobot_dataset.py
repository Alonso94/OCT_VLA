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
    parser.add_argument(
        "--entity-tokens",
        action="store_true",
        help="Export raw schema-v2 entity tokens and their validity mask.",
    )
    parser.add_argument("--max-entities", type=int, default=16)
    parser.add_argument(
        "--entity-training-sources",
        nargs="+",
        type=Path,
        help="Explicit training-only episodes used to fit entity statistics.",
    )
    args = parser.parse_args()
    normalizer = None
    if args.entity_tokens:
        if not args.entity_training_sources:
            parser.error("--entity-tokens requires --entity-training-sources; never fit validation")
        from oct_vla.data.entity_tokens import (
            EntityTokenNormalizer,
            build_entity_tokens,
            shelf_support_entities,
        )
        from oct_vla.data.store import read_episode
        from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC

        supports = shelf_support_entities(DEFAULT_SPEC)

        def training_entities():
            for source in args.entity_training_sources:
                episode = read_episode(source)
                for sample in episode.samples:
                    yield build_entity_tokens(
                        sample.scene, sample.observation.eef, supports, args.max_entities
                    )

        normalizer = EntityTokenNormalizer.fit(training_entities())
    spec = ObjectTokenSpec(max_objects=args.max_objects) if args.object_tokens else None
    report = export_episodes(
        args.sources,
        args.output,
        repo_id=args.repo_id,
        object_token_spec=spec,
        entity_token_max_entities=args.max_entities if args.entity_tokens else None,
        entity_normalizer=normalizer,
    )
    print(f"exported {len(report.exported)} episodes to {report.output}")
    for source, reason in report.skipped:
        print(f"skipped {source}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
