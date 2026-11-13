#!/usr/bin/env python
"""Fixed time-row evaluation packs from the unchanged v1 generator.

Writes `<out>/<split>.npz`, `<split>.meta.jsonl` and `<split>.json` (manifest) for each of
v1's program splits (train, composition, shelf_family). Each pack is seeded per chunk
like v1's `prior_generate.py`, so every model is scored on the same items. Packs live in
their own namespace and are never overwritten.

    time_row_generate.py --out $HPCVAULT/octvla-prior/eval/time_row_v1 --items 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from oct_vla.prior import programs as P
    from oct_vla.prior.batches import ItemConfig
    from oct_vla.prior.time_row_batches import TimeRowConfig, build_time_row_eval_set

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default="time_row_v1")
    ap.add_argument("--items", type=int, default=4000)
    ap.add_argument("--splits", nargs="+", default=["train", "composition", "shelf_family"])
    ap.add_argument("--rows-per-segment", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grammar", choices=("v1", "v2"), default="v1",
                    help="v2: the push prior (programs_v2, generator_v2)")
    args = ap.parse_args()
    for split in args.splits:
        if args.grammar == "v2":
            from oct_vla.prior.programs_v2 import program_split_v2

            programs = program_split_v2(split)
        else:
            programs = P.program_split(split)
        cfg = TimeRowConfig(item=ItemConfig(programs=programs),
                            rows_per_segment=args.rows_per_segment)
        path = build_time_row_eval_set(args.out, args.name, split, args.items,
                                       workers=args.workers, cfg=cfg)
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
