#!/usr/bin/env python
"""Rank rollout-evaluated checkpoints for three-object development only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oct_vla.experiments import load_eval_report, select_rollout_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="Rollout evaluation JSON reports")
    parser.add_argument("--output", type=Path, help="Write the selection report here")
    args = parser.parse_args()

    reports = [(path, load_eval_report(path)) for path in args.reports]
    selection = select_rollout_checkpoint(reports)
    print(
        f"selected {selection['selected']['checkpoint']} "
        f"from {selection['selected']['report_path']}"
    )
    print(f"{'checkpoint':<48} {'ok':>4} {'xfers':>6} {'inf':>4} {'n':>4}")
    print("-" * 72)
    for row in selection["ranked"]:
        print(
            f"{row['checkpoint']:<48} {row['successes']:>4} "
            f"{row['transfers_completed']:>6} {row['infeasible']:>4} "
            f"{row['episodes']:>4}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
