#!/usr/bin/env python
"""Train a Phase-2 time-row prior: TimeRowPFN (B1) or the KeyframePFN-encoder control (A1).

Both stream fresh time-row items (`time_row_batches.sample_time_row_batch`) and train
on the same paired loss (`time_row_model.paired_loss`). A1 sees the items converted to
v1's keyframe tables (`time_row_v1_bridge`), so the comparison isolates encoder from
target. Runs go to a new namespace and are never overwritten.

    time_row_prior_train.py --model timerow --grammar v1 --rows-per-segment 1 \\
        --lang $P/language/minilm.pt --out $P/runs_timerow/B1-v1-r1-s0 --steps 10000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import torch

    from oct_vla.prior import model as M
    from oct_vla.prior import programs as P
    from oct_vla.prior.batches import ItemConfig
    from oct_vla.prior.data import load_language
    from oct_vla.prior.time_row_batches import TimeRowConfig, sample_time_row_batch
    from oct_vla.prior.time_row_model import TimeRowPFN, paired_loss, to_torch_time_row
    from oct_vla.prior.time_row_v1_bridge import KeyframePairedPFN, to_keyframe_batch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("timerow", "keyframe_paired"), required=True)
    ap.add_argument("--grammar", choices=("v1", "v2"), default="v1")
    ap.add_argument("--rows-per-segment", type=int, default=1)
    ap.add_argument("--no-query-initial", action="store_true",
                    help="Ablation: the query is its current row only (TimeRowPFN)")
    ap.add_argument("--lang", type=Path, default=None,
                    help="Instruction embeddings (v1 grammar only; v2 trains without language)")
    ap.add_argument("--anchor", choices=("gripper", "object"), default="gripper",
                    help="Where each arm's position residual starts (time_row_model.PairedHead)")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; runs are never overwritten")
    if args.model == "keyframe_paired" and args.grammar != "v1":
        raise SystemExit("the A1 control reads v1 keyframe tables: --grammar v1 only")
    lang = load_language(args.lang) if args.lang and args.grammar == "v1" else None
    if args.grammar == "v2":
        from oct_vla.prior.programs_v2 import TRAIN_PROGRAMS_V2

        programs = TRAIN_PROGRAMS_V2
    else:
        programs = P.TRAIN_PROGRAMS
    language = (0.2, 0.4, 0.4) if lang is not None else (1.0, 0.0, 0.0)
    cfg = TimeRowConfig(item=ItemConfig(programs=programs, language=language,
                                        support_dropout=0.25 if lang is not None else 0.0),
                        rows_per_segment=args.rows_per_segment)
    torch.manual_seed(args.seed)
    lang_dim = 0 if lang is None else lang[0].shape[-1]
    if args.model == "timerow":
        model = TimeRowPFN(width=args.width, heads=args.heads, layers=args.layers,
                           grammar=args.grammar, lang_dim=lang_dim,
                           query_initial=not args.no_query_initial, anchor=args.anchor)
    else:
        model = KeyframePairedPFN(width=args.width, heads=args.heads, layers=args.layers,
                                  lang_dim=lang_dim, use_edges=False, anchor=args.anchor)
    model = model.to(args.device).train()
    rot = args.grammar == "v2"

    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            rng = np.random.default_rng([args.seed, info.id if info else 0])
            while True:
                inputs, targets, _ = sample_time_row_batch(rng, cfg, args.batch)
                if args.model == "keyframe_paired":
                    inputs = to_keyframe_batch(inputs, targets)
                yield inputs, targets

    loader = torch.utils.data.DataLoader(Stream(), batch_size=None, num_workers=args.workers,
                                         persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup)
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    args.out.mkdir(parents=True)
    meta = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "model_config": model.config, "schema": cfg.schema,
            "parameters": sum(p.numel() for p in model.parameters()),
            "item_config": {"language": language, "programs": f"TRAIN_PROGRAMS{'_V2' if rot else ''}",
                            "rows_per_segment": args.rows_per_segment}}
    (args.out / "config.json").write_text(json.dumps(meta, indent=1))
    hist = (args.out / "history.jsonl").open("a")
    t0, running = time.time(), {}
    for step, (inputs, targets) in enumerate(loader):
        if step >= args.steps:
            break
        inputs = {k: np.asarray(v) for k, v in inputs.items()}
        targets = {k: np.asarray(v) for k, v in targets.items()}
        if args.model == "keyframe_paired":
            b = M.to_torch(inputs, lang, args.device)
            b.update({f"t_{k}": torch.as_tensor(v).to(args.device) for k, v in targets.items()})
            out = model.forward_paired(b)
        else:
            b = to_torch_time_row(inputs, targets, lang, args.device)
            out = model(b)
        total, terms = paired_loss({k: v.float() for k, v in out.items()}, b, rot)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in terms.items():
            running[k] = running.get(k, 0.0) + v
        if (step + 1) % 100 == 0:
            rec = {"step": step + 1, "time_s": round(time.time() - t0, 1),
                   "lr": sched.get_last_lr()[0], **{k: round(v / 100, 5) for k, v in running.items()}}
            running = {}
            hist.write(json.dumps(rec) + "\n")
            hist.flush()
            print(json.dumps(rec), flush=True)
    torch.save({"config": model.config, "state": model.state_dict(), "steps": args.steps,
                "kind": args.model}, args.out / "model.pt")
    print(f"saved {args.out / 'model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
