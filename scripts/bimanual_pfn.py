#!/usr/bin/env python
"""BimanualPFN entry point (plan: stage 0). One subcommand per stage; more are added as
they are built.

    bimanual_pfn.py fixtures --out reports/fixtures     # first-slice acceptance + figures
    bimanual_pfn.py generate --out PACKS --packs dev heldout_handover_colour --items 512
    bimanual_pfn.py overfit  --out reports/overfit.json  # CPU memorisation gate
    bimanual_pfn.py train    --out RUN --arm full --seed 0 --dev PACKS/dev.pkl
    bimanual_pfn.py eval     --run RUN --pack PACKS/dev.pkl --out RUN/eval_dev.json
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


def cmd_generate(args) -> int:
    from oct_vla.prior.pose_window import packs

    for name in args.packs:
        path = packs.build(name, args.out, args.items, s_rows=args.s_rows)
        print(path, (path.with_suffix(".json")).read_text())
    return 0


def cmd_overfit(args) -> int:
    import torch

    from oct_vla.prior.pose_window import train as Tr

    torch.set_num_threads(args.threads)
    items = Tr.conflict_items(args.pairs, seed=args.seed, s_rows=args.s_rows)
    base = Tr.baselines(items)
    _, res = Tr.overfit(items, steps=args.steps, s_rows=args.s_rows, seed=args.seed)
    gate = (res["both_correct"] >= 0.9 and base["hold"]["both_correct"] < 0.5
            and base["replay"]["both_correct"] < 0.5)
    report = {"pairs": args.pairs, "model": res, "baselines": base, "gate_passed": gate}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"model_both_correct": res["both_correct"],
                      "hold": base["hold"]["both_correct"],
                      "replay": base["replay"]["both_correct"], "gate_passed": gate}))
    return 0 if gate else 1


def cmd_train(args) -> int:
    from oct_vla.prior.pose_window import packs, train as Tr

    dev = packs.load(args.dev) if args.dev else None
    Tr.train(args.out, arm=args.arm, steps=args.steps, batch=args.batch, width=args.width,
             layers=args.layers, lr=args.lr, workers=args.workers, seed=args.seed,
             s_rows=args.s_rows, dev_items=dev, eval_every=args.eval_every, device=args.device)
    return 0


def cmd_eval(args) -> int:
    import numpy as np

    from oct_vla.prior.pose_window import packs, train as Tr

    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    items = packs.load(args.pack)
    model = Tr.load(args.run, args.device)
    rng = np.random.default_rng(0)
    results = {"run": str(args.run), "pack": str(args.pack), "items": len(items)}
    for cond in ("correct", "shuffled", "none"):
        if cond == "correct":
            use = items
        elif cond == "shuffled":
            perm = rng.permutation(len(items))
            use = [dict(it, support=items[perm[k]]["support"]) for k, it in enumerate(items)]
        else:
            use = [dict(it, support=_empty_support(it["support"])) for it in items]
        s_rows = max(len(i["support"].time) for i in items)
        results[cond] = Tr.evaluate(Tr.predict(model, use, s_rows, args.device), use)
    results["baselines"] = Tr.baselines(items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps({k: (v.get("both_correct"), v.get("trans_mm")) for k, v in results.items()
                      if isinstance(v, dict) and "both_correct" in v}))
    return 0


def _empty_support(table):
    """The same support with every row marked invalid: the 'none' condition."""
    from dataclasses import replace

    return replace(table, time=table.time[:1], pose=table.pose[:1], observed=table.observed[:1] & False,
                   age=table.age[:1], goal=table.goal[:1], goal_valid=table.goal_valid[:1] & False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fixtures", help="first-slice CPU fixtures and their figures")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_fixtures)
    p = sub.add_parser("generate", help="fixed evaluation packs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--packs", nargs="+", default=["dev"])
    p.add_argument("--items", type=int, default=512)
    p.add_argument("--s-rows", type=int, default=128)
    p.set_defaults(func=cmd_generate)
    p = sub.add_parser("overfit", help="the CPU memorisation gate on fixed conflict pairs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=32)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--s-rows", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.set_defaults(func=cmd_overfit)
    p = sub.add_parser("train", help="train one arm x seed")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", default="full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--s-rows", type=int, default=128)
    p.add_argument("--dev", type=Path)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("eval", help="score a checkpoint on a pack (correct / shuffled / none)")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_eval)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
