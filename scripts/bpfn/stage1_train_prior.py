#!/usr/bin/env python
"""Stage 1, prior training: one arm x seed of the pose-window PFN on synthetic items.

    stage1_train_prior.py --out RUN --arm full --seed 0 --bank BANK --dev PACKS/dev.pkl
    stage1_train_prior.py --overfit --out reports/overfit.json     # CPU memorisation gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def run_overfit(args) -> int:
    import torch

    from oct_vla.bpfn.benchmark_tools.evaluator import baselines
    from oct_vla.bpfn.training_tools.trainer import conflict_items, overfit

    torch.set_num_threads(args.threads)
    items = conflict_items(args.pairs, seed=args.seed, s_rows=args.overfit_s_rows)
    base = baselines(items)
    _, scores = overfit(items, steps=args.overfit_steps, s_rows=args.overfit_s_rows, seed=args.seed)
    passed = (scores["both_correct"] >= 0.9 and base["hold"]["both_correct"] < 0.5
              and base["replay"]["both_correct"] < 0.5)
    report = {"pairs": args.pairs, "model": scores, "baselines": base, "gate_passed": passed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"model_both_correct": scores["both_correct"],
                      "hold": base["hold"]["both_correct"],
                      "replay": base["replay"]["both_correct"], "gate_passed": passed}))
    return 0 if passed else 1


def run_training(args) -> int:
    from oct_vla.bpfn.data import packs
    from oct_vla.bpfn.training_tools.trainer import PFNTrainer, TrainConfig

    cfg = TrainConfig(out=args.out, arm=args.arm, seed=args.seed, steps=args.steps,
                      batch=args.batch, width=args.width, layers=args.layers, lr=args.lr,
                      workers=args.workers, s_rows=args.s_rows, eval_every=args.eval_every,
                      device=args.device, bank=args.bank, reuse=args.reuse,
                      pair_extra=args.pair_extra, conflict_fraction=args.conflict_fraction,
                      per_support=args.per_support, tf32=args.tf32)
    dev_items = packs.load(args.dev) if args.dev else None
    PFNTrainer(cfg, dev_items).train()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="run directory (or the gate's report)")
    parser.add_argument("--arm", default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--s-rows", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dev", type=Path, help="dev pack (logged, never used for selection)")
    parser.add_argument("--bank", type=Path, help="episode bank (stage1_generate_data.py)")
    parser.add_argument("--reuse", type=int, help="items per support/query pair")
    parser.add_argument("--pair-extra", type=int, help="extra items per conflict member")
    parser.add_argument("--conflict-fraction", type=float)
    parser.add_argument("--per-support", type=int, default=1,
                        help="queries per support in a batch (bank only; supports encoded once)")
    parser.add_argument("--tf32", action="store_true", help="TF32 matmuls on CUDA")
    parser.add_argument("--overfit", action="store_true", help="run the CPU memorisation gate")
    parser.add_argument("--pairs", type=int, default=32, help="overfit: conflict pairs")
    parser.add_argument("--overfit-steps", type=int, default=1500)
    parser.add_argument("--overfit-s-rows", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    return run_overfit(args) if args.overfit else run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
