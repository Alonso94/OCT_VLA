#!/usr/bin/env python
"""Validate a dataset's entity metadata and emit the policy arguments it implies.

One argument per line: the entity capacity, and the training-split
normalisation statistics the policy standardises entities with. The arm itself
(`--policy.object_conditioning`) is the launcher's choice, not the dataset's.
"""

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
    args = [
        f"--policy.object_max_entities={capacity}",
        "--policy.object_entity_normalizer=" + json.dumps(stats.to_dict(), separators=(",", ":")),
    ]
    # A view with per-object visual features (scripts/add_entity_visual.py).
    visual = info.get("features", {}).get("observation.entity_visual")
    if visual is not None:
        shape = list(visual.get("shape", []))
        if len(shape) != 2 or shape[0] != capacity:
            raise ValueError(f"observation.entity_visual is {shape}; expected [{capacity}, D]")
        args.append(f"--policy.object_visual_dim={shape[1]}")
    # A view with the next-interaction keyframe (scripts/add_future_keyframe.py):
    # the dataset carries the branch's input, so it switches the branch on --
    # a kv run on this view without it would ignore the very column it exists for.
    future = info.get("features", {}).get("observation.future_keyframe")
    if future is not None:
        from oct_vla.data.future_keyframes import FUTURE_DIM

        shape = list(future.get("shape", []))
        mask = list(info["features"].get("observation.future_keyframe_mask", {}).get("shape", []))
        # The view says what it holds (add_future_keyframe.py); a latent view
        # carries the prior's cells, whose width is the prior's, not FUTURE_DIM.
        kind = info.get("future_keyframe", {}).get("kind", "keyframe")
        if kind not in ("keyframe", "latent"):
            raise ValueError(f"unknown future_keyframe kind {kind!r}")
        width = FUTURE_DIM if kind == "keyframe" else (shape[1] if len(shape) == 2 else None)
        if len(shape) != 2 or shape[1] != width or mask != shape[:1]:
            raise ValueError(f"observation.future_keyframe is {shape} with mask {mask}; "
                             f"expected [rows, {width}] and [rows] for a {kind} view")
        args += ["--policy.object_future=true", f"--policy.object_future_rows={shape[0]}"]
        if kind == "latent":
            args += ["--policy.object_future_kind=latent", f"--policy.object_future_dim={width}"]
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print("\n".join(entity_training_args(args.dataset)))


if __name__ == "__main__":
    main()
