#!/usr/bin/env python
"""Stage 2, offline evaluation of a trained PFN.

    stage2_evaluate.py --data synthetic --run RUN --pack PACKS/dev.pkl --out RUN/eval_dev.json
    stage2_evaluate.py --data real --run RUN --corpus leftmost=ROOT --corpus rightmost=ROOT \\
        --out RUN/eval_real_v4.json [--support RULE=ROOT] [--queries 20]
    stage2_evaluate.py --data atlas [--run RUN] --out reports/scan    # no --run: the expert

``--corpus RULE=ROOT`` names the rule its runs were collected under (checked against the
runs' metadata); supports come from the same corpora, never the query's own run, unless
``--support RULE=ROOT`` names a separate bank.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def rule_roots(values: list[str] | None) -> dict[str, list[Path]]:
    roots: dict[str, list[Path]] = {}
    for value in values or []:
        rule, path = value.split("=", 1)
        roots.setdefault(rule, []).append(Path(path))
    return roots


def evaluate_synthetic(args, model) -> dict:
    from oct_vla.bpfn.benchmark_tools.evaluator import Evaluator
    from oct_vla.bpfn.data import packs

    items = packs.load(args.pack)
    if items[0]["target"].shape[0] != model.horizon:
        raise SystemExit(f"{args.pack} has {items[0]['target'].shape[0]}-step targets but the "
                         f"checkpoint predicts {model.horizon}; regenerate the pack")
    results = {"run": str(args.run), "pack": str(args.pack), "items": len(items)}
    results.update(Evaluator(items, args.device).run(model))
    return results


def evaluate_real(args, model) -> dict:
    from oct_vla.bpfn.benchmark_tools.real_evaluator import RealEvaluator
    from oct_vla.bpfn.data.real import RealData

    queries = RealData(rule_roots(args.corpus))
    supports = RealData(rule_roots(args.support)) if args.support else None
    evaluator = RealEvaluator(queries, supports, queries_per_rule=args.queries, seed=args.seed,
                              device=args.device)
    results = evaluator.run(model)
    results.update(run=str(args.run),
                   corpora={rule: len(runs) for rule, runs in queries.runs.items()},
                   target_source="next_measured_pose")
    return results


def scan_atlas(args, model) -> list[dict]:
    from oct_vla.bpfn.benchmark_tools import atlas
    from oct_vla.bpfn.benchmark_tools.evaluator import predict

    if model is None:
        predict_windows = atlas.expert_predict
        title = "Marker scan: analytic expert windows (scorer check)"
    else:
        def predict_windows(items):
            s_rows = max(len(i["support"].time) for i in items)
            return predict(model, items, s_rows, args.device)
        title = f"Marker scan: {args.run.name}"
    rows = atlas.marker_scan(predict_windows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "marker_scan.json").write_text(json.dumps(rows, indent=1, default=float) + "\n")
    for path in atlas.render(rows, args.out / "marker_scan", title):
        print(path)
    for name in ("nearest", "farthest"):
        correct = [r["correct"] for r in rows if r["support"] == name and r["correct"] is not None]
        print(name, sum(correct) / len(correct))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=("synthetic", "real", "atlas"), required=True)
    parser.add_argument("--run", type=Path, help="training run (atlas: omit for the expert)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pack", type=Path, help="synthetic: the evaluation pack")
    parser.add_argument("--corpus", action="append", help="real: RULE=ROOT of query runs")
    parser.add_argument("--support", action="append", help="real: RULE=ROOT of support runs")
    parser.add_argument("--queries", type=int, help="real: query runs per rule")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from oct_vla.bpfn import models

    if args.data != "atlas" and args.run is None:
        parser.error("--run is required")
    if args.data == "synthetic" and args.pack is None:
        parser.error("--data synthetic needs --pack")
    if args.data == "real" and not args.corpus:
        parser.error("--data real needs --corpus")
    model = models.load(args.run, args.device) if args.run is not None else None

    if args.data == "atlas":
        scan_atlas(args, model)
        return 0
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    if args.data == "synthetic":
        results = evaluate_synthetic(args, model)
    else:
        results = evaluate_real(args, model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, default=float) + "\n")
    summary = {k: v for k, v in results.items() if k != "records"}
    print(json.dumps(summary, indent=1, default=float)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
