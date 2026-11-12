#!/usr/bin/env python
"""PFN-VLA stage B: fine-tune the pretrained prior on real demonstrations (prior/finetune.py).

Starts from a pretrained run (``--init``), or from random weights of the same
architecture (``--scratch``, the no-pretraining control, trained on real data only).
Each batch mixes real items with synthetic replay (``--synthetic-fraction``).
Real items come from the training seed blocks of every rule given; validation seeds
are G1b's (``prior_shelf_g1b.py --query-split val``).

    prior_finetune.py --init $P/runs/P1-interaction-s0 \\
        --collection leftmost=$C/leftmost/full_run/three_object \\
        --collection rightmost=$C/rightmost/full_run/three_object \\
        --lang $P/language/minilm.pt --out $P/runs/F1-finetune-s0
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

    from oct_vla.data.entity_tokens import shelf_support_entities
    from oct_vla.data.store import read_episode
    from oct_vla.prior import finetune as FT
    from oct_vla.prior import model as M
    from oct_vla.prior.batches import ItemConfig
    from oct_vla.prior.data import Stream, load_language
    from oct_vla.tasks.shelf_restock.spec import spec_for_rule

    ap = argparse.ArgumentParser()
    start = ap.add_mutually_exclusive_group(required=True)
    start.add_argument("--init", type=Path, help="Pretrained prior run to fine-tune")
    start.add_argument("--scratch", type=Path,
                       help="Random init with this run's architecture (no pretraining)")
    ap.add_argument("--collection", action="append", required=True, help="RULE=PATH")
    ap.add_argument("--lang", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--synthetic-fraction", type=float, default=None,
                    help="Share of each batch from the generator (default 0.5; 0 for --scratch)")
    ap.add_argument("--pair-fraction", type=float, default=0.25,
                    help="Share of the real items that are frame-0 conflict pairs")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None,
                    help="Default 1e-4 fine-tuning, 3e-4 from scratch")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12, help="Synthetic generator workers")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    synthetic = args.synthetic_fraction
    if synthetic is None:
        synthetic = 0.0 if args.scratch else 0.5
    if args.scratch and synthetic > 0:
        raise SystemExit("--scratch is the no-pretraining control: it sees no synthetic data")
    lr = args.lr or (3e-4 if args.scratch else 1e-4)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; runs are never overwritten")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    lang = load_language(args.lang)
    source = args.init or args.scratch
    blob = torch.load(source / "model.pt", map_location="cpu")
    model = M.KeyframePFN(**blob["config"])
    if args.init:
        model.load_state_dict(blob["state"])
    model = model.to(args.device).train()

    episodes, supports = {}, {}
    for spec in args.collection:
        rule, path = spec.split("=", 1)
        dirs = sorted(p for p in Path(path).glob("seed_*/*") if p.is_dir())
        eps = []
        for d in dirs:
            seed = int(d.parent.name.split("_")[1])
            try:
                if FT.split_of(seed) != "train":
                    continue
            except ValueError:
                continue  # outside every split, or reserved: never read
            ep = read_episode(d)
            if ep.success:
                eps.append(ep)
        episodes[rule] = eps
        supports[rule] = shelf_support_entities(spec_for_rule(rule))
        print(f"{rule}: {len(eps)} training episodes", flush=True)
    data = FT.build(episodes, supports, stride=args.stride)
    print({r: len(v) for r, v in data.items.items()}, f"{len(data.pairs)} conflict pairs",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup)
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    n_syn = int(round(args.batch * synthetic))
    # Generation is the slow part (the pretraining needed 12 workers): replay is
    # streamed from background workers, as in prior_train.py.
    synthetic_batches = iter(torch.utils.data.DataLoader(
        Stream(ItemConfig(), n_syn, args.seed), batch_size=None, num_workers=args.workers,
        persistent_workers=args.workers > 0)) if n_syn else None
    args.out.mkdir(parents=True)
    meta = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "synthetic_fraction": synthetic, "lr": lr, "model_config": model.config,
            "episodes": {r: [int(e.seed) for e in eps] for r, eps in episodes.items()},
            "real_items": {r: len(v) for r, v in data.items.items()},
            "conflict_pairs": len(data.pairs)}
    (args.out / "config.json").write_text(json.dumps(meta, indent=1))
    hist = (args.out / "history.jsonl").open("a")
    t0, running = time.time(), {}
    for step in range(args.steps):
        parts = [FT.sample(data, rng, args.batch - n_syn, pair_fraction=args.pair_fraction)]
        if n_syn:
            parts.append(FT.with_masks({k: np.asarray(v) for k, v in
                                        next(synthetic_batches).items()}))
        b = M.to_torch(FT.concat(*parts), lang, args.device)
        with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16,
                            enabled=args.device != "cpu"):
            out = model(b)
        total, terms = M.loss(model, {k: v.float() for k, v in out.items()}, b)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in terms.items():
            running[k] = running.get(k, 0.0) + v
        if (step + 1) % 100 == 0:
            rec = {"step": step + 1, "time_s": round(time.time() - t0, 1),
                   "lr": sched.get_last_lr()[0],
                   **{k: round(v / 100, 5) for k, v in running.items()}}
            running = {}
            hist.write(json.dumps(rec) + "\n")
            hist.flush()
            print(json.dumps(rec), flush=True)
    torch.save({"config": model.config, "state": model.state_dict(), "steps": args.steps,
                "init": str(source), "pretrained": bool(args.init)}, args.out / "model.pt")
    print(f"saved {args.out / 'model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
