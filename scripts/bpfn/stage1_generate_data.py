#!/usr/bin/env python
"""Stage 1, synthetic data: the training episode bank and the fixed evaluation packs.

    stage1_generate_data.py --bank $WORK/octvla-scratch/bpfn-bank/v1
    stage1_generate_data.py --packs-root $HPCVAULT/octvla-bpfn/packs --packs dev test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from oct_vla.bpfn.data.bank import BankSpec  # noqa: E402
from oct_vla.bpfn.data.synthetic import SyntheticData  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, help="episode bank directory")
    parser.add_argument("--bank-name", default="v1")
    parser.add_argument("--pool", type=int, default=24, help="episodes per (program, count)")
    parser.add_argument("--conflict", type=int, default=24, help="conflict families per (program, count)")
    parser.add_argument("--handover", type=int, default=48, help="handover families per (program, count)")
    parser.add_argument("--shards", type=int, default=32)
    parser.add_argument("--shard-ids", type=int, nargs="+", help="build only these shards")
    parser.add_argument("--workers", type=int, help="processes (default: the CPUs this job owns)")
    parser.add_argument("--packs-root", type=Path, help="evaluation pack directory")
    parser.add_argument("--packs", nargs="+", default=[], help="pack names (data/packs.py)")
    parser.add_argument("--items", type=int, default=512, help="items per pack")
    parser.add_argument("--s-rows", type=int, default=128)
    args = parser.parse_args()
    if args.bank is None and not args.packs:
        parser.error("nothing to do: pass --bank and/or --packs")

    spec = BankSpec(name=args.bank_name, pool=args.pool, conflict=args.conflict,
                    handover=args.handover, shards=args.shards)
    data = SyntheticData(bank_root=args.bank, bank_spec=spec, packs_root=args.packs_root,
                         pack_names=tuple(args.packs), pack_items=args.items, s_rows=args.s_rows,
                         workers=args.workers)
    data.generate_data(shard_ids=args.shard_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
