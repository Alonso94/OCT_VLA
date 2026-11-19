#!/usr/bin/env python
"""BimanualPFN entry point (plan: stage 0). One subcommand per stage; more are added as
they are built.

    bimanual_pfn.py fixtures --out reports/fixtures     # first-slice acceptance + figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def cmd_fixtures(args) -> int:
    from oct_vla.prior.pose_window import branches as B
    from oct_vla.prior.pose_window import fixtures as F
    from oct_vla.prior.pose_window import render as R

    colour = F.colour_fixture(seed=args.seed)
    report = {"colour": {"checks": colour["checks"],
                         "first_picks": {k: sorted(v) for k, v in colour["sets"].items()}},
              "handover": {}}
    written = R.colour_fixture_figure(colour, args.out / "colour_fixture")
    for giver, name in ((0, "left_gives"), (1, "right_gives")):
        eps = B.family(giver=giver)
        checks = B.handover_checks(eps)
        report["handover"][name] = {"cue_row": checks["cue"], "extra_hold_s": checks["extra_hold_s"],
                                    "checks": checks["checks"]}
        written += R.handover_fixture_figure(eps, checks, args.out / f"handover_fixture_{name}")
    passed = all(colour["checks"].values()) and all(
        all(v["checks"].values()) for v in report["handover"].values())
    report["passed"] = passed
    (args.out / "fixtures.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    print("\n".join(str(p) for p in written))
    return 0 if passed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fixtures", help="first-slice CPU fixtures and their figures")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_fixtures)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
