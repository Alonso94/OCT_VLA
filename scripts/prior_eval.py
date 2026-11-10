#!/usr/bin/env python
"""Gate G1 of the PFN-VLA plan: does the prior infer the next keyframe from its context?

    prior_eval.py --eval-dir $HPCVAULT/octvla-prior/eval/v1 --runs RUN [RUN ...] \
        --lang $HPCVAULT/octvla-prior/language/minilm.pt --out results.json

For every split (train / composition / shelf_family), every predictor (each run, and the
copy-current and retrieval baselines), and every support × language condition
(interventions.py), it reports:
- accuracy with Wilson intervals;
- median position errors;
- contrast accuracy.

Under `conflicting` support it also scores the predictions against the partner's targets:
`follows_support` vs `follows_language`.

Read it per regime. With no or ambiguous language, a context-using model collapses under
`shuffled` and follows the support under `conflicting`. With explicit language, following
the language is correct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oct_vla.prior import baselines as BL
from oct_vla.prior import interventions as I
from oct_vla.prior import metrics as MT
from oct_vla.prior import model as M
from oct_vla.prior.data import load_eval_set, load_language

SPLITS = ("train", "composition", "shelf_family")


def load_run(path: Path, device):
    blob = torch.load(path / "model.pt", map_location="cpu")
    model = M.KeyframePFN(**blob["config"])
    model.load_state_dict(blob["state"])
    return model.to(device).eval()


def compact(s: dict) -> dict:
    return {
        k: (
            round(v["acc"], 3)
            if isinstance(v, dict)
            else round(v, 2)
            if isinstance(v, float)
            else v
        )
        for k, v in s.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="*", default=[])
    ap.add_argument("--lang", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    lang = load_language(args.lang)

    predictors = {"copy_current": None, "retrieval": None}
    for r in args.runs:
        predictors[r.name] = load_run(r, args.device)
    results: dict = {}
    for split in args.splits:
        base = load_eval_set(args.eval_dir / f"{split}.npz")
        for name, model in predictors.items():
            has_lang = model is not None and model.lang is not None
            langs = ("none", "ambiguous", "explicit") if has_lang else ("none",)
            for sup in I.SUPPORTS:
                for lg in langs:
                    rng = np.random.default_rng(
                        [args.seed, I.SUPPORTS.index(sup), I.LANGUAGES.index(lg)]
                    )
                    b, mask = I.support(base, sup, rng)
                    b = I.language(b, lg, rng)
                    if model is None:
                        pred = BL.copy_current(b) if name == "copy_current" else BL.retrieval(b)
                    else:
                        pred = M.predict(model, b, lang, args.device)
                    key = f"{split}/{name}/{sup}/{lg}"
                    res = {"all": MT.summarise(pred, b, mask)}
                    if sup in ("correct", "shuffled"):
                        k = base["demo_valid"].sum(1)
                        for kk in np.unique(k):
                            res[f"K={kk}"] = MT.summarise(pred, b, mask & (k == kk))
                    if sup == "conflicting":
                        res["follows_support"] = MT.summarise(pred, I.partner_targets(b), mask)
                    results[key] = res
                    print(
                        key,
                        compact(res["all"]),
                        "| follows_support",
                        compact(res["follows_support"]) if "follows_support" in res else "",
                        flush=True,
                    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"args": {k: str(v) for k, v in vars(args).items()}, "results": results},
            indent=1,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
