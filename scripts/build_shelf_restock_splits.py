#!/usr/bin/env python
"""Export one LeRobot dataset per variant whose episode order encodes the split.

Collection is split-agnostic (docs/slurm_collection.md): every seed of a profile
lands under one canonical root, and the split is recovered afterwards from the
seed recorded for each episode. This script is that recovery step.

It exports train episodes first, then validation, each block in ascending seed
order, because LeRobot's own offline-validation path
(`lerobot.datasets.factory.make_train_eval_datasets`) does not read seeds: it
holds out the *last* ``ceil(n_episodes * eval_split)`` episodes per task. Laying
the episodes out in that order makes that positional rule select exactly the
reserved validation seeds, so `lerobot-train --dataset.eval_split=...` reproduces
the protocol split rather than an arbitrary tail.

The resulting fraction is therefore a derived quantity, not a tuning knob. It is
computed and verified here, and written to split_manifest.json next to the
dataset. A boundary that does not land exactly on the train/validation seed
division is a hard error: silently training on validation clips is precisely the
failure this script exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Reserved seed blocks from docs/dataset_protocol.md. A seed belongs to exactly
#: one split, fixed before collection, so a scene can never move between splits.
#:
#: A split may own several ranges. Budgeting by *run* rather than by clip needs
#: far more scenes than the original blocks were sized for, so the train split
#: was extended into 350-399 (unreserved, below the two-object block at 400).
#: Ranges are listed in the order they were reserved and clips are taken in
#: ascending seed, so a later range only ever adds scenes behind the earlier
#: ones and a budget of N runs stays a subset of a budget of N+1.
SPLIT_BLOCKS: dict[str, dict[str, tuple[range, ...]]] = {
    "three_object": {
        "train": (range(100, 200), range(350, 400)),
        "val": (range(200, 250),),
    },
    "two_object": {"val": (range(400, 450),)},
    "four_object": {"val": (range(600, 650),)},
}


def seed_of(clip_dir: Path) -> int:
    """Seed owning a clip, read from its ``seed_<n>/`` parent directory."""
    for part in clip_dir.parts[::-1]:
        if part.startswith("seed_"):
            return int(part.removeprefix("seed_"))
    raise ValueError(f"No seed_<n> component in {clip_dir}")


def collect_clips(
    canonical_root: Path, blocks: dict[str, tuple[range, ...]]
) -> dict[str, list[Path]]:
    """Group canonical clip directories by the split their seed is reserved for."""
    grouped: dict[str, list[Path]] = {name: [] for name in blocks}
    unreserved: list[int] = []
    for episode in canonical_root.rglob("episode.json"):
        clip = episode.parent
        seed = seed_of(clip)
        for name, ranges in blocks.items():
            if any(seed in block for block in ranges):
                grouped[name].append(clip)
                break
        else:
            unreserved.append(seed)
    if unreserved:
        print(f"note: ignoring {len(unreserved)} clip(s) from unreserved seeds "
              f"{sorted(set(unreserved))}", file=sys.stderr)
    # Ascending seed, then clip name, so transfer order within a scene is kept.
    for name in grouped:
        grouped[name].sort(key=lambda p: (seed_of(p), p.name))
    return grouped


def take_runs(clips: list[Path], limit: int | None) -> list[Path]:
    """Keep the clips belonging to the first `limit` complete runs.

    A run is one scene played to the end -- every object moved -- and is stored
    as its atomic transfer clips, not concatenated. The budget is counted in
    runs because that is the unit of data collection: asking for 25 runs means
    25 scenes and all ~75 clips they contain, not 25 clips.

    Ascending seed, matching the protocol's fixed selection rule, so growing
    the budget only ever adds scenes and never reshuffles which ones a smaller
    budget used.
    """
    if limit is None:
        return clips
    kept: list[Path] = []
    seen: list[int] = []
    for clip in clips:
        seed = seed_of(clip)
        if seed not in seen:
            if len(seen) == limit:
                break
            seen.append(seed)
        kept.append(clip)
    if len(seen) < limit:
        raise SystemExit(
            f"Asked for {limit} runs but only {len(seen)} are available in this split. "
            "Collect more seeds, or lower the budget."
        )
    return kept


def main() -> int:
    from oct_vla.data.lerobot_export import export_episodes
    from oct_vla.data.object_tokens import ObjectTokenSpec

    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Dataset root to create")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--profile", default="three_object", choices=sorted(SPLIT_BLOCKS))
    parser.add_argument("--object-tokens", action="store_true")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Train on the first N complete runs (scenes), including all of each "
        "run's atomic transfer clips. The unit is runs, not clips: --max-runs 25 "
        "means 25 scenes and the ~75 clips they contain. Applies to the train "
        "split only; validation always uses every reserved run it has.",
    )
    parser.add_argument(
        "--privileged",
        action="store_true",
        help="Export object tokens as observation.environment_state and omit the "
        "cameras, for a vision-free upper bound on what privileged scene state alone "
        "can achieve. Implies --object-tokens.",
    )
    parser.add_argument(
        "--control-space",
        default="cartesian",
        choices=["cartesian", "joint", "joint_delta"],
        help="'joint' makes observation.state the measured joint configuration and "
        "action the next one, so the policy is proprioceptive in the space it "
        "commands and no inverse kinematics sits between them.",
    )
    args = parser.parse_args()

    blocks = SPLIT_BLOCKS[args.profile]
    grouped = collect_clips(args.canonical_root, blocks)
    train, val = grouped.get("train", []), grouped.get("val", [])
    if not train:
        raise SystemExit(f"No train clips found under {args.canonical_root}")
    # Budget the TRAIN split only. Validation keeps every reserved run it has,
    # so a data-scaling curve varies one thing -- how much the policy saw --
    # and every point is scored against the same held-out scenes.
    train = take_runs(train, args.max_runs)
    train_runs = len({seed_of(c) for c in train})
    val_runs = len({seed_of(c) for c in val})
    print(
        f"train: {train_runs} runs / {len(train)} clips  |  "
        f"val: {val_runs} runs / {len(val)} clips"
    )

    ordered = train + val
    total, n_val = len(ordered), len(val)

    # Invert LeRobot's rule: it holds out ceil(total * eval_split) episodes. Take
    # the midpoint of the interval of fractions that yield exactly n_val so the
    # value is robust to float representation at either end.
    if n_val:
        lo, hi = (n_val - 1) / total, n_val / total
        eval_split = (lo + hi) / 2
        if math.ceil(total * eval_split) != n_val:
            raise SystemExit(f"Could not derive an eval_split selecting exactly {n_val} episodes")
    else:
        eval_split = 0.0

    spec = ObjectTokenSpec() if (args.object_tokens or args.privileged) else None
    report = export_episodes(
        ordered,
        args.output,
        repo_id=args.repo_id,
        object_token_spec=spec,
        control_space=args.control_space,
        privileged=args.privileged,
    )
    print(f"exported {len(report.exported)} episodes to {report.output}")
    for source, reason in report.skipped:
        print(f"skipped {source}: {reason}")
    if len(report.exported) != total:
        raise SystemExit(
            f"{total - len(report.exported)} clip(s) were skipped; the positional "
            f"eval_split boundary would no longer match the seed boundary"
        )

    # The boundary must fall exactly between the last train seed and the first
    # validation seed, or the held-out tail is not the reserved block.
    boundary = total - n_val
    if n_val:
        last_train, first_val = seed_of(ordered[boundary - 1]), seed_of(ordered[boundary])
        if last_train >= first_val:
            raise SystemExit(f"Split boundary is not ordered: {last_train} >= {first_val}")

    manifest = {
        "profile": args.profile,
        "repo_id": args.repo_id,
        "total_episodes": total,
        "eval_split": eval_split,
        "train": {
            "episodes": [0, boundary - 1],
            "count": boundary,
            "seeds": sorted({seed_of(p) for p in train}),
        },
        "val": {
            "episodes": [boundary, total - 1] if n_val else None,
            "count": n_val,
            "seeds": sorted({seed_of(p) for p in val}),
        },
    }
    (args.output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\ntrain: {boundary} episodes from {len(manifest['train']['seeds'])} seeds")
    print(f"val:   {n_val} episodes from {len(manifest['val']['seeds'])} seeds")
    if n_val:
        print(f"boundary: episode {boundary - 1} (seed {last_train}) | "
              f"episode {boundary} (seed {first_val})")
    print(f"\nlerobot-train --dataset.eval_split={eval_split!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
