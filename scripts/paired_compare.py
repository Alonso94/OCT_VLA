#!/usr/bin/env python
"""Pair two cells that may sit in different matrices, episode by episode.

`collect_final_results.py` pairs arms within one matrix prefix. Comparing
across prefixes -- atomic clips (AF) against full runs (CF), or a Stage A cell
against its full-run counterpart -- needs the same matching: the same training
seed, tier, profile and scene seed, scored by exact McNemar.

    paired_compare.py <eval dir> AF-rgb CF-rgb [CF-rgb_cont ...]

Each argument after the directory is PREFIX-ARM, optionally under a
subdirectory of it (`exec8/GF-rgb`: the same checkpoints rolled out with a
different execution horizon); every later cell is paired against the first.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "collect_final_results", Path(__file__).resolve().parent / "collect_final_results.py"
)
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)

SCOPES = (
    ("three_seen", "three_object", "seen"),
    ("three_heldout", "three_object", "heldout"),
    ("three_novel", "three_object", "novel"),
    ("two", "two_object", "seen"),
    ("four", "four_object", "seen"),
)


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def cell_rows(eval_dir: Path, cell: str) -> dict:
    subdir, _, cell = cell.rpartition("/")
    prefix, arm = cell.split("-", 1)
    rows, _ = collect.load(eval_dir / subdir if subdir else eval_dir, prefix)
    return {(r["train_seed"], r["tier"], r["profile"], r["eval_seed"]): r
            for r in rows if r["arm"] == arm}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("cells", nargs="+", help="PREFIX-ARM, the first is the baseline")
    args = parser.parse_args()
    base = cell_rows(args.eval_dir, args.cells[0])
    for cell in args.cells[1:]:
        other = cell_rows(args.eval_dir, cell)
        for scope, profile, tier in SCOPES:
            keys = [k for k in base if k in other and k[2] == profile and k[1] == tier]
            if not keys:
                continue
            parts = []
            for measure in ("success", "transfer"):
                b = sum(base[k][measure] and not other[k][measure] for k in keys)
                c = sum(other[k][measure] and not base[k][measure] for k in keys)
                parts.append(f"{measure} {sum(base[k][measure] for k in keys)}->"
                             f"{sum(other[k][measure] for k in keys)} b/c {b}/{c} "
                             f"p={mcnemar(b, c):.3f}")
            mean = lambda rows: sum(rows[k]["transfers"] for k in keys) / len(keys)  # noqa: E731
            print(f"{args.cells[0]} -> {cell:20} {scope:14} n={len(keys):3}  "
                  + "  ".join(parts) + f"  mean {mean(base):.2f}->{mean(other):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
