#!/usr/bin/env python
"""Pick one representative rollout video per reported row, for the report.

Every final-matrix rollout records all of its episodes to a staging area
(`slurm/submit_final_experiments.sh`). This picks, for each row the results
table reports -- an arm on three objects (per tier, or pooled if the tiers are
indistinguishable, by the same rule the table uses), and on two and four
objects -- the one episode that best stands for that row, copies it to a
report directory, and writes a table saying exactly what each one shows.

"Representative" is defined, not chosen by eye, because the tempting choice is
the best episode and that is the one this project has already been misled by:

1. the **median training seed** of the row, by mean transfers (the lower of
   the two middle seeds on a tie), so a lucky seed is not the face of the arm;
2. within it, the episode whose transfers are **closest to that seed's mean**,
   then whose lifts are closest to its mean lifts, then the lowest scene seed.

So a row averaging 0.4 transfers shows an episode with no transfer, and one
averaging 1.05 shows a single transfer -- what a viewer should expect to see.

    scripts/select_rollout_videos.py $OCTVLA_OUTPUT_ROOT/eval --prefix F \
        --dest docs/videos/F --table /tmp/F.md

docs/rollout_videos.md holds the report's selection (docs/videos/<prefix>/).
Mind the home quota when choosing --dest: the home filesystem allocates 32 MB
per file, so 45 half-megabyte videos take ~1.5 GB of it.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import statistics
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "collect_final_results", Path(__file__).resolve().parent / "collect_final_results.py"
)
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)

SETTING = {"two_object": "2 objects", "three_object": "3 objects", "four_object": "4 objects"}


def options(rows: list[dict], arms: list[str]) -> list[tuple[str, str, str, dict]]:
    """(arm, setting slug, setting label, row filter) for every reported row."""
    split = any(collect.tier_shifts(rows, arms).values())
    out = []
    for arm in arms:
        if split:
            for tier in collect.TIERS:
                out.append((arm, f"three_{tier}", f"3 objects, {tier} identities",
                            dict(arm=arm, profile="three_object", tier=tier)))
        else:
            out.append((arm, "three_all", "3 objects, all identities",
                        dict(arm=arm, profile="three_object")))
        for profile in ("two_object", "four_object"):
            out.append((arm, profile.replace("_object", ""), f"{SETTING[profile]}, seen",
                        dict(arm=arm, profile=profile, tier="seen")))
    return out


def representative(episodes: list[dict]) -> tuple[dict, dict]:
    """The chosen episode, and the per-seed summary it was chosen from."""
    by_seed: dict[int, list[dict]] = {}
    for episode in episodes:
        by_seed.setdefault(episode["train_seed"], []).append(episode)
    means = {s: statistics.mean(e["transfers"] for e in eps) for s, eps in by_seed.items()}
    ordered = sorted(by_seed, key=lambda s: (means[s], s))
    seed = ordered[(len(ordered) - 1) // 2]
    pool = by_seed[seed]
    mean_t = means[seed]
    mean_l = statistics.mean(e["lifted"] for e in pool)
    chosen = min(pool, key=lambda e: (abs(e["transfers"] - mean_t),
                                      abs(e["lifted"] - mean_l), e["eval_seed"]))
    return chosen, {"seed": seed, "seed_mean_transfers": mean_t, "per_seed": means}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--prefix", default="F")
    parser.add_argument("--dest", type=Path, required=True, help="Where the videos go")
    parser.add_argument("--table", type=Path, help="Markdown table path (default: DEST/table.md)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows, rejected = collect.load(args.eval_dir, args.prefix)
    if rejected:
        print(f"ignoring {len(rejected)} rejected rollout(s); run the collector for reasons")
    arms = [a for a in collect.ARMS if collect.select(rows, arm=a)]
    if not arms:
        raise SystemExit(f"no admissible {args.prefix}-matrix rollouts under {args.eval_dir}")

    lines = [
        "| arm | setting | training seed | scene seed | meshes | transfers | lifted "
        "| success | row mean transfers (per seed) | video |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    missing = []
    if not args.dry_run:
        args.dest.mkdir(parents=True, exist_ok=True)
    for arm, slug, label, where in options(rows, arms):
        episodes = collect.select(rows, **where)
        if not episodes:
            continue
        chosen, summary = representative(episodes)
        source = Path(chosen["video"]) if chosen["video"] else None
        name = f"{arm}__{slug}.mp4"
        if source is None or not source.is_file():
            missing.append(f"{arm} / {label}: {source or 'no video recorded'}")
            link = "missing"
        else:
            if not args.dry_run:
                shutil.copyfile(source, args.dest / name)
            link = f"`{args.dest.resolve() / name}`"
        per_seed = ", ".join(f"{v:.2f}" for _, v in sorted(summary["per_seed"].items()))
        lines.append(
            f"| {arm} | {label} | {chosen['train_seed']} | {chosen['eval_seed']} "
            f"| {chosen['tier']} | {chosen['transfers']:g} | {chosen['lifted']} "
            f"| {'yes' if chosen['success'] else 'no'} | {per_seed} | {link} |"
        )
    table = "\n".join(lines)
    print(table)
    if missing:
        print("\nno video for:\n  " + "\n  ".join(missing))
    if not args.dry_run:
        target = args.table or args.dest / "table.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(table + "\n")
        size = sum(p.stat().st_size for p in args.dest.glob("*.mp4")) / 1e6
        print(f"\nwrote {args.dest} ({size:.1f} MB of video)")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
