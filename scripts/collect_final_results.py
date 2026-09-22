#!/usr/bin/env python
"""Collect the final matrix into one table, with the caveats attached.

Reads every rollout the final experiments wrote, groups by arm and identity
tier, and reports across seeds rather than from one. That last part is the
point: a single-seed cell is what produced this project's 6/20 result and its
retraction, so nothing here prints a number without the spread beside it.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

CELL = re.compile(r"F-(?P<arm>rgb|entity|semantic)-s(?P<seed>\d+)-(?P<tier>seen|heldout|novel)")
ARMS = ("rgb", "entity", "semantic")
TIERS = ("seen", "heldout", "novel")
TIER_NOTE = {
    "seen": "identities the policy trained on",
    "heldout": "identities excluded from the dataset",
    "novel": "a mesh present in zero collected runs",
}


def read(path: Path) -> dict | None:
    match = CELL.search(path.stem)
    if not match:
        return None
    report = json.loads(path.read_text())
    episodes = report.get("episodes") or []
    if not episodes:
        return None
    return {
        **match.groupdict(),
        "episodes": len(episodes),
        "success": sum(1 for e in episodes if e.get("success")),
        "transfer": sum(1 for e in episodes if e.get("transfers_completed", 0) > 0),
        "lift": sum(1 for e in episodes if e.get("objects_lifted", 0) > 0),
        "mean_transfers": sum(e.get("transfers_completed", 0) for e in episodes) / len(episodes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", type=Path, help="Directory of rollout JSONs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cells = [r for r in (read(p) for p in sorted(args.eval_dir.glob("*.json"))) if r]
    if not cells:
        raise SystemExit(f"no final-matrix rollouts found under {args.eval_dir}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for cell in cells:
        grouped[(cell["arm"], cell["tier"])].append(cell)

    print(f"{'arm':10} {'tier':9} {'seeds':>5} {'success':>18} {'>=1 transfer':>18} "
          f"{'mean transfers':>18}")
    print("-" * 82)
    table = {}
    for arm in ARMS:
        for tier in TIERS:
            rows = grouped.get((arm, tier), [])
            if not rows:
                continue
            n = rows[0]["episodes"]
            succ = [r["success"] for r in rows]
            tr = [r["transfer"] for r in rows]
            mt = [r["mean_transfers"] for r in rows]

            def spread(values, denom=None):
                lo, hi = min(values), max(values)
                mid = statistics.mean(values)
                body = f"{mid:.2f}" if denom is None else f"{mid:.1f}/{denom}"
                return f"{body} [{lo}-{hi}]" if denom else f"{body} [{lo:.2f}-{hi:.2f}]"

            print(f"{arm:10} {tier:9} {len(rows):>5} {spread(succ, n):>18} "
                  f"{spread(tr, n):>18} {spread(mt):>18}")
            table[f"{arm}/{tier}"] = {
                "seeds": sorted(int(r["seed"]) for r in rows),
                "episodes_per_seed": n,
                "success": succ,
                "transfer": tr,
                "mean_transfers": mt,
                "note": TIER_NOTE[tier],
            }
        print()

    single = [k for k, v in table.items() if len(v["seeds"]) < 2]
    if single:
        print("WARNING: single-seed cells, not to be reported as results: " + ", ".join(single))
    print(
        "\nRanges are min-max across training seeds, not confidence intervals.\n"
        "A cell whose range spans zero has not established anything."
    )
    if args.output:
        args.output.write_text(json.dumps(table, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
