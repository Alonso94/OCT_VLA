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
                      per_support=args.per_support, tf32=args.tf32,
                      fresh_supports=args.fresh_supports, compile=args.compile,
                      lr_floor=args.lr_floor, warmup=args.warmup, arch=args.arch,
                      task_read=args.task_read, output=args.output, task_tokens=args.task_tokens,
                      clip=args.clip, frame=args.frame, hypotheses=args.hypotheses,
                      support_patch=args.support_patch, bf16_trunk=args.bf16_trunk,
                      aux_factors=args.aux_factors, ema=args.ema)
    val_packs = {}
    if args.dev:
        val_packs["dev"] = packs.load(args.dev)
    for value in args.val or []:
        name, path = value.split("=", 1)
        val_packs[name] = packs.load(Path(path))
    PFNTrainer(cfg, val_packs).train()
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
    parser.add_argument("--lr-floor", type=float, default=0.0, help="final lr as a fraction of --lr")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--compile", action="store_true", help="torch.compile the training forward")
    parser.add_argument("--arch", choices=("axial", "table"), default="axial")
    parser.add_argument("--clip", type=float, default=1.0, help="gradient-norm clip")
    parser.add_argument("--frame", choices=("workcell", "relative"), default="workcell",
                        help="table: positions around the workcell or each table's object centroid")
    parser.add_argument("--hypotheses", type=int, default=1, help="table rows: futures predicted")
    parser.add_argument("--support-patch", type=int, default=1, help="table: demonstration rows per token")
    parser.add_argument("--bf16-trunk", action="store_true", help="table: axial blocks in bfloat16")
    parser.add_argument("--aux-factors", type=float, default=0.0, help="table: rule-factor loss weight")
    parser.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0: off)")
    parser.add_argument("--task-read", choices=("cell", "film", "cross", "cross_all"), default="cell",
                        help="table: how the query reads the demonstration")
    parser.add_argument("--output", choices=("head", "rows"), default="head", help="table: output")
    parser.add_argument("--task-tokens", type=int, default=1, help="table: pooled demonstration tokens")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--s-rows", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dev", type=Path, help="dev pack (logged, never used for selection)")
    parser.add_argument("--val", action="append", help="NAME=PACK, more packs logged in training")
    parser.add_argument("--bank", type=Path, help="episode bank (stage1_generate_data.py)")
    parser.add_argument("--reuse", type=int, help="items per support/query pair")
    parser.add_argument("--pair-extra", type=int, help="extra items per conflict member")
    parser.add_argument("--conflict-fraction", type=float)
    parser.add_argument("--per-support", type=int, default=1,
                        help="queries per support in a batch (bank only; supports encoded once)")
    parser.add_argument("--tf32", action="store_true", help="TF32 matmuls on CUDA")
    parser.add_argument("--fresh-supports", action="store_true",
                        help="bank queries, every support generated anew")
    parser.add_argument("--overfit", action="store_true", help="run the CPU memorisation gate")
    parser.add_argument("--pairs", type=int, default=32, help="overfit: conflict pairs")
    parser.add_argument("--overfit-steps", type=int, default=1500)
    parser.add_argument("--overfit-s-rows", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    return run_overfit(args) if args.overfit else run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
