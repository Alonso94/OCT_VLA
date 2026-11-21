#!/usr/bin/env python
"""Offline evaluation of a BimanualPFN checkpoint on recorded runs (plan stage 2a).

    bpfn_eval_real.py --run RUN --corpus leftmost=ROOT --corpus rightmost=ROOT \\
        --out RUN/eval_real_v4.json [--queries 20] [--device cuda]

Each ``--corpus RULE=ROOT`` names the rule its runs were collected under (the run's own
metadata must agree). Supports for a query come from the same corpora, never the query's
own run; ``--support RULE=ROOT`` draws them from a separate bank instead. See
``oct_vla.prior.pose_window.real_eval`` for what is scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _pairs(values):
    out = {}
    for v in values or []:
        rule, path = v.split("=", 1)
        out.setdefault(rule, []).append(Path(path))
    return out


def main() -> int:
    from oct_vla.data.store import read_states
    from oct_vla.prior.pose_window import real_eval as RE
    from oct_vla.bpfn.models.pose_window_pfn import PoseWindowPFN

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--corpus", action="append", required=True, help="RULE=ROOT (queries)")
    ap.add_argument("--support", action="append", help="RULE=ROOT (support bank)")
    ap.add_argument("--queries", type=int, default=None, help="per rule")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")

    def runs_for(spec: dict) -> dict:
        out = {}
        for rule, roots in spec.items():
            runs = [p for root in roots for p in RE.find_runs(root)]
            for p in runs[:3]:
                got = read_states(p).metadata.get("rule")
                if got != rule:
                    raise SystemExit(f"{p} was collected under rule {got!r}, not {rule!r}")
            out[rule] = runs
        return out

    corpora = {r: RE.Corpus(r, runs) for r, runs in runs_for(_pairs(args.corpus)).items()}
    pool = runs_for(_pairs(args.support)) if args.support else None
    model = PoseWindowPFN.from_run(args.run, args.device)
    result = RE.evaluate(model, corpora, queries_per_rule=args.queries, seed=args.seed,
                         device=args.device, support_pool=pool)
    result.update(run=str(args.run), corpora={r: len(c.runs) for r, c in corpora.items()},
                  target_source="next_measured_pose")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=1, default=float)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
