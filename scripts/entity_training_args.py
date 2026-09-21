#!/usr/bin/env python
"""Validate exported entity metadata and emit one policy argument per line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oct_vla.data.entity_tokens import ENTITY_TOKEN_DIM, ENTITY_TOKEN_SCHEMA, EntityTokenNormalizer


def entity_training_args(dataset: Path) -> list[str]:
    info = json.loads((dataset / "meta/info.json").read_text())
    metadata = info.get("entity_tokens", {})
    if (
        metadata.get("schema") != ENTITY_TOKEN_SCHEMA
        or metadata.get("token_dim") != ENTITY_TOKEN_DIM
    ):
        raise ValueError("Training requires an exported entity-v2 dataset")
    if metadata.get("raw_on_disk") is not True:
        raise ValueError("Entity inputs must be raw on disk; policy applies stored normalization")
    if not metadata.get("normalization"):
        raise ValueError("Export training-only entity normalization before launching training")
    stats = EntityTokenNormalizer.from_dict(metadata["normalization"])
    capacity = metadata.get("capacity")
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise ValueError("Entity capacity must be a positive integer")
    for key, expected in (
        ("observation.entity_tokens", [capacity, ENTITY_TOKEN_DIM]),
        ("observation.entity_mask", [capacity]),
    ):
        if list(info.get("features", {}).get(key, {}).get("shape", [])) != expected:
            raise ValueError(f"Dataset feature {key} disagrees with entity metadata")
    return [
        "--policy.object_representation=entity_v2",
        "--policy.object_injection_mode=layerwise",
        f"--policy.object_max_entities={capacity}",
        "--policy.object_entity_normalizer=" + json.dumps(stats.to_dict(), separators=(",", ":")),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print("\n".join(entity_training_args(args.dataset)))


if __name__ == "__main__":
    main()
