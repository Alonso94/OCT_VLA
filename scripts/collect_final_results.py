#!/usr/bin/env python
"""Collect the final matrix into one table, with the caveats attached.

Reads every rollout the final experiments wrote and reports across training
seeds rather than from one. That last part is the point: a single-seed cell is
what produced this project's 6/20 result and its retraction, so nothing here
prints a number without the spread beside it.

Three axes, each reported only as finely as the data supports:

* **Identity tier** (seen / held-out / novel). Reported per tier only when some
  arm's tier differs from its seen score *in the same direction on every
  training seed*. Otherwise the tiers are statistically one population, and
  splitting them would just print three noisier copies of the same number --
  so they are pooled into one row and the table says so.
* **Object count** (2 / 3 / 4, seen identities). Always reported: it is the
  compositional-generalisation question, and a flat line is itself the answer.
* **Arm against rgb**, paired on (training seed, evaluation seed): every
  conditioned arm fine-tunes from the same seed's rgb checkpoint and faces the
  same scenes, so exact McNemar over the discordant episodes is the right test.

A rollout is admitted only if it proves its tier: the report must name the
variants it requested and every episode must name the variants it spawned,
inside that set. The first run of this matrix wrote no such record, and all 44
of its tier rollouts turned out to be the same unpinned scenes three times.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_eval import paired_difference  # noqa: E402

#: `<prefix>-<arm>-s<seed>-<job>`, where job is a tier, or `count` for the
#: two- and four-object rollouts on seen identities.
CELL = re.compile(r"^(?P<prefix>[A-Z])-(?P<arm>[a-z]+)-s(?P<seed>\d+)-(?P<job>[a-z]+)$")
#: Ordered so the table reads as the ControlVLA recipe does: the stage-one
#: policy, then what conditioning adds to it, then the ablation that omits
#: stage one entirely. No `semantic`: its cells trained the entity model under
#: another name (the flag selecting it was never read).
ARMS = ("rgb", "entity", "adaln", "incontext", "scratch")
TIERS = ("seen", "heldout", "novel")
PROFILES = ("two_object", "three_object", "four_object")
ARM_NOTE = {
    "rgb": "stage 1: images and proprioception, no conditioning",
    "entity": "stage 2: layerwise conditioning on geometry, from the rgb checkpoint",
    "adaln": "stage 2: entity + AdaLN-Zero on every encoder/decoder block (LPWM)",
    "incontext": "stage 2: entity + entity tokens in the encoder sequence (LPWM)",
    "scratch": "w/o pretrain: conditioning with no stage 1 -- ControlVLA's failing ablation",
}
#: The variants each tier must be pinned to, per matrix. Checked against what a
#: rollout requested, so a file whose name and pin disagree is refused rather
#: than filed under the tier its name claims. Must match the submitter's TIER.
TIER_IDS = {
    "F": {"seen": [1, 2, 3, 4], "heldout": [0, 6], "novel": [5]},
}
TIER_NOTE = {
    "seen": "identities the policy trained on",
    "heldout": "identities excluded from the dataset",
    "novel": "a variant present in zero collected runs",
    "all": "tiers pooled: no arm differed consistently across seeds",
}


class Rejected(ValueError):
    """A rollout that cannot prove which scenes it scored."""


def read(path: Path, prefix: str) -> list[dict]:
    """One row per episode, tagged with its cell. Raises Rejected if unpinned."""
    match = CELL.match(path.stem)
    if not match or match["prefix"] != prefix or match["arm"] not in ARMS:
        return []
    report = json.loads(path.read_text())
    requested = report.get("model_ids_requested")
    if not requested:
        raise Rejected(f"{path.name}: records no requested variants, so its tier is unproven")
    job = match["job"]
    tier = "seen" if job == "count" else job
    if tier not in TIERS:
        raise Rejected(f"{path.name}: unknown job {job!r}")
    expected = TIER_IDS.get(prefix, {}).get(tier)
    if expected is not None and sorted(requested) != expected:
        raise Rejected(
            f"{path.name}: named {tier!r} but pinned to {sorted(requested)}, not {expected}"
        )
    rows = []
    for episode in report.get("episodes") or []:
        spawned = episode.get("model_ids")
        if not spawned or not set(spawned) <= set(requested):
            raise Rejected(
                f"{path.name}: seed {episode.get('seed')} spawned {spawned}, "
                f"outside the requested {requested}"
            )
        rows.append({
            "arm": match["arm"], "train_seed": int(match["seed"]), "tier": tier,
            "profile": episode["profile"], "eval_seed": int(episode["seed"]),
            "success": bool(episode.get("success")),
            "transfer": episode.get("transfers_completed", 0) > 0,
            "transfers": float(episode.get("transfers_completed", 0)),
            "lift": episode.get("objects_lifted", 0) > 0,
            "lifted": int(episode.get("objects_lifted", 0)),
            "video": episode.get("video") or "",
        })
    return rows


def per_seed(rows: list[dict]) -> dict[int, dict]:
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_seed[row["train_seed"]].append(row)
    return {
        seed: {
            "episodes": len(r),
            "success": sum(x["success"] for x in r),
            "transfer": sum(x["transfer"] for x in r),
            "lift": sum(x["lift"] for x in r),
            "mean_transfers": sum(x["transfers"] for x in r) / len(r),
        }
        for seed, r in sorted(by_seed.items())
    }


def consistent_shift(base: dict[int, dict], other: dict[int, dict]) -> int:
    """+1 / -1 if `other` beats / trails `base` on every shared seed, else 0.

    Judged on mean transfers, the finest-grained score. Requires at least two
    shared seeds, since one seed agreeing with itself establishes nothing.
    """
    seeds = sorted(set(base) & set(other))
    if len(seeds) < 2:
        return 0
    deltas = [other[s]["mean_transfers"] - base[s]["mean_transfers"] for s in seeds]
    if all(d > 0 for d in deltas):
        return 1
    if all(d < 0 for d in deltas):
        return -1
    return 0


def select(rows: list[dict], **where) -> list[dict]:
    return [r for r in rows if all(r[k] == v for k, v in where.items())]


def tier_shifts(rows: list[dict], arms: list[str]) -> dict[tuple[str, str], int]:
    """Each arm's held-out and novel tier against its own seen tier."""
    stats = {
        (arm, tier): per_seed(select(rows, arm=arm, tier=tier, profile="three_object"))
        for arm in arms for tier in TIERS
    }
    return {
        (arm, tier): consistent_shift(stats[(arm, "seen")], stats[(arm, tier)])
        for arm in arms for tier in TIERS[1:]
        if stats[(arm, "seen")] and stats[(arm, tier)]
    }


def load(eval_dir: Path, prefix: str) -> tuple[list[dict], list[str]]:
    """Every admissible episode row, and a reason for each rejected file."""
    rows, rejected = [], []
    for path in sorted(eval_dir.glob(f"{prefix}-*.json")):
        try:
            rows += read(path, prefix)
        except Rejected as error:
            rejected.append(str(error))
    return rows, rejected


def spread(values: list[float], denom: int | None = None) -> str:
    lo, hi = min(values), max(values)
    mid = statistics.mean(values)
    if denom is not None:
        return f"{mid:.1f}/{denom} [{lo:g}-{hi:g}]"
    return f"{mid:.2f} [{lo:.2f}-{hi:.2f}]"


def row_line(label: str, sub: str, stats: dict[int, dict]) -> tuple[str, dict]:
    n = min(s["episodes"] for s in stats.values())
    succ = [s["success"] for s in stats.values()]
    tr = [s["transfer"] for s in stats.values()]
    mt = [s["mean_transfers"] for s in stats.values()]
    line = (f"{label:10} {sub:13} {len(stats):>5} {spread(succ, n):>18} "
            f"{spread(tr, n):>18} {spread(mt):>18}")
    return line, {
        "seeds": sorted(stats), "episodes_per_seed": n,
        "success": succ, "transfer": tr, "mean_transfers": mt,
    }


def header(sub: str) -> str:
    return (f"{'arm':10} {sub:13} {'seeds':>5} {'success':>18} {'>=1 transfer':>18} "
            f"{'mean transfers':>18}\n" + "-" * 86)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", type=Path, help="Directory of rollout JSONs")
    parser.add_argument("--prefix", default="F", help="Matrix prefix: F shelf, E easy task")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows, rejected = load(args.eval_dir, args.prefix)
    if rejected:
        print(f"REJECTED {len(rejected)} rollout(s) that cannot prove their tier:")
        for line in rejected:
            print(f"  {line}")
        print()
    if not rows:
        raise SystemExit(f"no admissible {args.prefix}-matrix rollouts under {args.eval_dir}")

    def pick(**where) -> list[dict]:
        return select(rows, **where)

    arms = [a for a in ARMS if pick(arm=a)]
    table: dict = {"identity": {}, "object_count": {}, "paired_vs_rgb": {}}

    # ---- identity tiers, three objects -----------------------------------
    stats = {
        (arm, tier): per_seed(pick(arm=arm, tier=tier, profile="three_object"))
        for arm in arms for tier in TIERS
    }
    shifts = tier_shifts(rows, arms)
    informative = [k for k, v in shifts.items() if v]
    split = bool(informative)
    print("IDENTITY (three objects)")
    if split:
        print("  tiers reported separately; consistent shift vs seen for: "
              + ", ".join(f"{a}/{t} ({'+' if shifts[(a, t)] > 0 else '-'})"
                          for a, t in informative))
    else:
        print("  no arm's held-out or novel tier differs from seen consistently across "
              "seeds;\n  tiers pooled into one row (per-tier rows are in --output)")
    print(header("tier"))
    for arm in arms:
        if split:
            for tier in TIERS:
                if stats[(arm, tier)]:
                    line, record = row_line(arm, tier, stats[(arm, tier)])
                    print(line)
                    table["identity"][f"{arm}/{tier}"] = record
        else:
            pooled = per_seed(pick(arm=arm, profile="three_object"))
            line, record = row_line(arm, "all", pooled)
            print(line)
            table["identity"][f"{arm}/all"] = record
            for tier in TIERS:
                if stats[(arm, tier)]:
                    table["identity"][f"{arm}/{tier}"] = row_line(arm, tier, stats[(arm, tier)])[1]
    table["identity_split"] = split

    # ---- object count, seen identities -----------------------------------
    print("\nOBJECT COUNT (seen identities; trained on three)")
    print(header("objects"))
    for arm in arms:
        for profile in PROFILES:
            s = per_seed(pick(arm=arm, tier="seen", profile=profile))
            if s:
                line, record = row_line(arm, profile.replace("_object", ""), s)
                print(line)
                table["object_count"][f"{arm}/{profile}"] = record
        print()

    # ---- paired against rgb ----------------------------------------------
    if "rgb" in arms:
        print("PAIRED vs rgb over matched (training seed, evaluation seed, scene) episodes")
        print(f"{'arm':10} {'scope':13} {'pairs':>5} {'success rgb->arm':>18} {'p':>7} "
              f"{'transfer rgb->arm':>18} {'p':>7}")
        print("-" * 86)
        scopes = {"three_object": dict(profile="three_object")}
        scopes |= {p: dict(profile=p, tier="seen") for p in ("two_object", "four_object")}
        for arm in arms:
            if arm == "rgb":
                continue
            for scope, where in scopes.items():
                def keyed(a):
                    return {(r["train_seed"], r["tier"], r["profile"], r["eval_seed"]): r
                            for r in pick(arm=a, **where)}
                base, treat = keyed("rgb"), keyed(arm)
                shared = sorted(set(base) & set(treat))
                if not shared:
                    continue
                result = {}
                for metric in ("success", "transfer"):
                    result[metric] = paired_difference(
                        [(base[k][metric], treat[k][metric]) for k in shared]
                    )
                s, t = result["success"], result["transfer"]
                print(f"{arm:10} {scope.replace('_object', ' obj'):13} {len(shared):>5} "
                      f"{s['baseline_rate']:>8.2f} -> {s['treatment_rate']:<6.2f} "
                      f"{s['mcnemar_p']:>7.3f} "
                      f"{t['baseline_rate']:>8.2f} -> {t['treatment_rate']:<6.2f} "
                      f"{t['mcnemar_p']:>7.3f}")
                table["paired_vs_rgb"][f"{arm}/{scope}"] = result
        print()

    single = [k for section in ("identity", "object_count")
              for k, v in table[section].items() if len(v["seeds"]) < 2]
    if single:
        print("WARNING: single-seed cells, not to be reported as results: " + ", ".join(single))
    print("Ranges are min-max across training seeds, not confidence intervals.\n"
          "A cell whose range spans zero has not established anything. p is two-sided\n"
          "exact McNemar; with 5 arms x 3 scopes, read p < 0.05 as a lead, not a finding.")
    if args.output:
        table["arm_notes"] = {a: ARM_NOTE[a] for a in arms}
        table["tier_notes"] = TIER_NOTE
        table["rejected"] = rejected
        args.output.write_text(json.dumps(table, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
