#!/usr/bin/env python
"""Train the interaction prior (KeyframePFN) on the synthetic generator.

    prior_train.py --out $HPCVAULT/octvla-prior/runs/P1-interaction-s0 --head interaction \
        --lang $HPCVAULT/octvla-prior/language/minilm.pt \
        --eval-set $HPCVAULT/octvla-prior/eval/v1/train.npz

Arms (Step 1 of the PFN-VLA plan):
    --head interaction          the method
    --head goal                 goal-only ablation: same contexts and encoder,
                                trained on the pointer and the final pose only
    --no-context                trained without demonstrations (language and query only)
    --freeze-encoder            random-encoder control: only the heads are trained

Data is generated on the fly by the loader workers, seeded by (seed, worker). The run
keeps one final checkpoint (`model.pt`) and a `history.jsonl`. The periodic evaluation on
`--eval-set` is a training monitor; Step 1's gate is read with `prior_eval.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from oct_vla.prior import metrics as MT
from oct_vla.prior import model as M
from oct_vla.prior.batches import ItemConfig
from oct_vla.prior.data import Stream, load_eval_set, load_language

HEAD_PREFIXES = ("pointer.", "final.", "grip_head.", "event_head.", "obj.", "grip.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--head", choices=M.HEADS, default="interaction")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--no-edges", action="store_true",
                    help="relational ablation: no difference edges in row attention")
    ap.add_argument(
        "--lang", type=Path, default=None, help="instruction embeddings; omit for no language"
    )
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--conflict-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--eval-set", type=Path, default=None)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} exists; runs are never overwritten")
    args.out.mkdir(parents=True)
    torch.manual_seed(args.seed)
    lang = load_language(args.lang)
    # Without a language encoder no item may carry an instruction.
    language = (0.2, 0.4, 0.4) if lang is not None else (1.0, 0.0, 0.0)
    cfg = ItemConfig(
        conflict_fraction=args.conflict_fraction,
        language=language,
        support_dropout=0.25 if lang is not None else 0.0,
    )
    if args.no_context and lang is None:
        raise SystemExit("--no-context without --lang leaves the model only the query")
    model = M.KeyframePFN(
        width=args.width,
        heads=args.heads,
        layers=args.layers,
        lang_dim=0 if lang is None else lang[0].shape[-1],
        head=args.head,
        use_context=not args.no_context,
        use_edges=not args.no_edges,
    ).to(args.device)
    params = [
        p
        for n, p in model.named_parameters()
        if not args.freeze_encoder or n.startswith(HEAD_PREFIXES)
    ]
    if args.freeze_encoder:
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith(HEAD_PREFIXES))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / args.warmup)
            * 0.5
            * (1 + math.cos(math.pi * min(1.0, s / args.steps)))
        ),
    )
    meta = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_config": model.config,
        "item_config": {
            "language": language,
            "conflict_fraction": args.conflict_fraction,
            "programs": "TRAIN_PROGRAMS",
        },
        "trainable_parameters": sum(p.numel() for p in params),
    }
    (args.out / "config.json").write_text(json.dumps(meta, indent=1))
    loader = torch.utils.data.DataLoader(
        Stream(cfg, args.batch, args.seed),
        batch_size=None,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    eval_set = load_eval_set(args.eval_set) if args.eval_set else None
    if eval_set is not None and lang is None:
        eval_set = {
            **eval_set,
            "lang_kind": eval_set["lang_kind"] * 0,
            "lang_index": eval_set["lang_index"] * 0 - 1,
        }
    hist = (args.out / "history.jsonl").open("a")
    t0, running = time.time(), {}
    model.train()
    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        b = M.to_torch(batch, lang, args.device)
        with torch.autocast(
            args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.device != "cpu"
        ):
            out = model(b)
        total, terms = M.loss(model, {k: v.float() for k, v in out.items()}, b)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        for k, v in terms.items():
            running[k] = running.get(k, 0.0) + v
        if (step + 1) % 200 == 0:
            rec = {
                "step": step + 1,
                "time_s": round(time.time() - t0, 1),
                "lr": sched.get_last_lr()[0],
                **{k: round(v / 200, 5) for k, v in running.items()},
            }
            running = {}
            if eval_set is not None and (step + 1) % args.eval_every == 0:
                pred = M.predict(model, eval_set, lang, args.device)
                s = MT.summarise(pred, eval_set)
                rec["monitor"] = {k: (v["acc"] if isinstance(v, dict) else v) for k, v in s.items()}
                model.train()
            hist.write(json.dumps(rec) + "\n")
            hist.flush()
            print(json.dumps(rec), flush=True)
    torch.save(
        {"config": model.config, "state": model.state_dict(), "steps": args.steps},
        args.out / "model.pt",
    )
    print(f"saved {args.out / 'model.pt'}")


if __name__ == "__main__":
    main()
