#!/usr/bin/env python
"""BimanualPFN tools not yet folded into the stage runners (scripts/bpfn/).

    bimanual_pfn.py fixtures --out reports/fixtures     # first-slice acceptance + figures
    bimanual_pfn.py eval     --run RUN --pack PACKS/dev.pkl --out RUN/eval_dev.json
    bimanual_pfn.py scan     --run RUN --out reports/scan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def cmd_fixtures(args) -> int:
    from oct_vla.bpfn.data.syn_data import branches
    from oct_vla.prior.pose_window import fixtures as F
    from oct_vla.prior.pose_window import render as R

    colour = F.colour_fixture(seed=args.seed)
    report = {"colour": {"checks": colour["checks"],
                         "first_picks": {k: sorted(v) for k, v in colour["sets"].items()}},
              "handover": {}}
    written = R.colour_fixture_figure(colour, args.out / "colour_fixture")
    for giver, name in ((0, "left_gives"), (1, "right_gives")):
        eps = branches.family(giver=giver)
        checks = branches.handover_checks(eps)
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


def cmd_eval(args) -> int:
    from oct_vla.bpfn.benchmark_tools.evaluator import Evaluator
    from oct_vla.bpfn.data import packs
    from oct_vla.bpfn.models.pose_window_pfn import PoseWindowPFN

    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    items = packs.load(args.pack)
    model = PoseWindowPFN.from_run(args.run, args.device)
    if items[0]["target"].shape[0] != model.horizon:
        raise SystemExit(f"{args.pack} has {items[0]['target'].shape[0]}-step targets but the "
                         f"checkpoint predicts {model.horizon}; regenerate the pack")
    results = {"run": str(args.run), "pack": str(args.pack), "items": len(items)}
    results.update(Evaluator(items, args.device).run(model))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps({k: (v.get("both_correct"), v.get("trans_mm")) for k, v in results.items()
                      if isinstance(v, dict) and "both_correct" in v}))
    return 0


def cmd_scan(args) -> int:
    from oct_vla.prior.pose_window import atlas as A
    from oct_vla.bpfn.benchmark_tools.evaluator import predict as predict_windows
    from oct_vla.bpfn.models.pose_window_pfn import PoseWindowPFN

    if args.run is None:
        predict, title = A.expert_predict, "Marker scan: analytic expert windows (scorer check)"
    else:
        model = PoseWindowPFN.from_run(args.run, args.device)
        predict = lambda items: predict_windows(model, items, max(len(i["support"].time) for i in items),  # noqa: E731
                                                args.device)
        title = f"Marker scan: {args.run.name}"
    rows = A.marker_scan(predict)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "marker_scan.json").write_text(json.dumps(rows, indent=1, default=float) + "\n")
    print("\n".join(map(str, A.render(rows, args.out / "marker_scan", title))))
    acc = {n: [r["correct"] for r in rows if r["support"] == n and r["correct"] is not None]
           for n in ("nearest", "farthest")}
    print({n: sum(v) / len(v) for n, v in acc.items()})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fixtures", help="first-slice CPU fixtures and their figures")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_fixtures)
    p = sub.add_parser("eval", help="score a checkpoint on a pack (correct / shuffled / none)")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser("scan", help="atlas marker scan for a checkpoint (or the expert)")
    p.add_argument("--run", type=Path, help="checkpoint; omit for the analytic expert")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_scan)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
