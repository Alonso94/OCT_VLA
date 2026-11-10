#!/usr/bin/env python
"""Generate the interaction prior's fixed evaluation sets (no simulator; CPU only).

    prior_generate.py --out $HPCVAULT/octvla-prior/eval/v1 --items 4000

Writes one set per program split:
- train         training programs on new scenes (in distribution);
- composition   held-out selector x arrangement / arm conjunctions;
- shelf_family  the shelf-restock programs, never seen in pretraining.

Each set is `<split>.npz` plus a `<split>.json` manifest. Existing sets are never overwritten.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from oct_vla.prior.data import build_eval_set

SPLITS = ("train", "composition", "shelf_family")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--items", type=int, default=4000)
    ap.add_argument(
        "--name", default="prior-eval-v1", help="seeds every chunk with (name, split, chunk)"
    )
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    args = ap.parse_args()
    for split in args.splits:
        path = build_eval_set(args.out, args.name, split, args.items, workers=args.workers)
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
