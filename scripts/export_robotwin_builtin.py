#!/usr/bin/env python
"""Export a collected RoboTwin built-in task as a LeRobot dataset.

Mirrors the shelf-restock protocol so the two corpora are comparable: episodes
ordered train-then-validation for LeRobot's positional `eval_split`, entity
normalisation fitted on the training partition alone, and whole object
*categories* withheld so an evaluation can separate learning the task from
learning the object.

Here the holdout is a genuine category rather than a size. `place_container_plate`
draws its container from `002_bowl` or `021_cup`, which can share a bounding
size -- unlike our shelf corpus, where one asset's variants differ only in
geometry and a semantic channel therefore cannot carry anything geometry does not.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


#: The rate every dataset in this project is declared at. The built-in
#: collector captures every 17 control steps at dt = 1/250 s (14.7 Hz), the
#: nearest it can get; LeRobot needs an integer.
CAPTURE_FPS_DECLARED = 15


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--task", default="place_container_plate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--max-entities", type=int, default=16)
    parser.add_argument(
        "--holdout-category",
        default="021_cup",
        help="Withheld from the dataset entirely, so it reaches neither split.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    from oct_vla.data.lerobot_export import export_episodes
    from oct_vla.data.store import read_episode
    from oct_vla.tasks.robotwin_builtin.adapter import spec_for

    spec = spec_for(args.task)
    root = args.collection_root / args.task
    clips = sorted(root.glob("seed_*/episode_*"), key=lambda p: int(p.parent.name.split("_")[1]))
    if not clips:
        raise SystemExit(f"no episodes under {root}")

    # Read from what was recorded, not assumed: a collection mixing capture
    # rates would give frames of different durations one declared rate.
    rates = {
        json.loads((clip / "episode.json").read_text())["metadata"].get("save_freq")
        for clip in clips
    }
    if len(rates) != 1 or None in rates:
        raise SystemExit(f"episodes disagree on save_freq (or lack it): {sorted(map(str, rates))}")
    spec_save_freq = int(rates.pop())

    def category(clip: Path) -> str:
        scene = json.loads((clip / "episode.json").read_text())["samples"][0]["scene"]
        for obj in scene["objects"]:
            if obj["track_id"] == spec.target_track_id:
                return obj.get("category") or "unknown"
        return "unknown"

    kept, held = [], []
    counts = Counter()
    for clip in clips:
        name = category(clip)
        counts[name] += 1
        (held if name == args.holdout_category else kept).append(clip)
    if not kept:
        raise SystemExit(f"--holdout-category {args.holdout_category!r} withheld every episode")

    # Deterministic and positional, as LeRobot's eval_split requires: the last
    # `val` episodes of the kept set, taken in seed order.
    n_val = max(1, round(len(kept) * args.val_fraction))
    train, val = kept[: len(kept) - n_val], kept[len(kept) - n_val :]
    ordered = train + val
    eval_split = len(val) / len(ordered)

    print(f"categories collected: {dict(counts)}")
    print(f"held out entirely   : {len(held)} episodes of {args.holdout_category}")
    print(f"train {len(train)} | val {len(val)} | eval_split {eval_split:.4f}")

    # Fitted on the training partition alone, like the shelf corpus: statistics
    # drawn over both splits would leak validation geometry into every sample.
    from oct_vla.data.entity_tokens import EntityTokenNormalizer, build_entity_tokens

    def training_entities():
        for clip in train:
            for sample in read_episode(clip).samples:
                yield build_entity_tokens(
                    sample.scene, sample.observation.eef, spec.supports(), args.max_entities
                )

    normalizer = EntityTokenNormalizer.fit(training_entities())
    # Paths, not episodes: `export_episodes` reads and validates each one itself.
    # Unified with binary gripper commands, as the shelf corpus is, so the same
    # view projection yields the absolute-EE training view.
    report = export_episodes(
        ordered,
        args.output,
        repo_id=args.repo_id,
        control_space="unified",
        gripper_encoding="binary_command",
        fps=CAPTURE_FPS_DECLARED,
        entity_token_max_entities=args.max_entities,
        entity_supports=spec.supports(),
        entity_normalizer=normalizer,
    )
    (args.output / "split_manifest.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "repo_id": args.repo_id,
                "total_episodes": len(ordered),
                "eval_split": eval_split,
                "train": {"count": len(train), "episodes": [0, len(train) - 1]},
                "val": {"count": len(val), "episodes": [len(train), len(ordered) - 1]},
                "identity_holdout": {
                    "count": len(held),
                    "categories": [args.holdout_category],
                },
                "train_identities": sorted({category(c) for c in train}),
                "categories_collected": dict(counts),
                # The declared rate is a label; the rollout harness must execute
                # one action per `save_freq` control steps, as recorded.
                "declared_fps": CAPTURE_FPS_DECLARED,
                "capture_save_freq": spec_save_freq,
                "capture_hz": 250.0 / spec_save_freq,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.output}: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
