"""Training and evaluation loops for PoseWindowPFN (plan 0.3).

- ``overfit``: a fixed set of conflict pairs, trained to memorisation on CPU. The
  engineering gate: >= 90 % paired both-correct, with hold and replay failing.
- ``train``: items drawn on the fly from the arm's programs and count matrix (one stream
  per data-loader worker), AdamW + warmup/cosine, scalar history every ``log_every`` steps,
  one checkpoint at the end (``model.pt``), evaluation on a fixed dev pack at milestones.
"""

from __future__ import annotations

import collections
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import batches as Bt
from . import generator as Gn
from . import metrics as Me
from . import splits as Sp
from .model import PoseWindowPFN, to_torch, window_loss


def conflict_items(n_pairs: int, seed: int, programs=None, counts=None,
                   s_rows: int = 64) -> list[dict]:
    cfg = Bt.ItemConfig(programs=programs or Sp.programs_for("full"),
                        counts=counts or Sp.CountMatrix(), s_rows=s_rows)
    rng = np.random.default_rng(seed)
    items: list[dict] = []
    while len(items) < 2 * n_pairs:
        try:
            items.extend(Bt.sample_pair(rng, cfg))
        except Gn.Infeasible:
            continue
    return items


@torch.no_grad()
def predict(model: PoseWindowPFN, items: list[dict], s_rows: int, device="cpu",
            batch: int = 64) -> list[np.ndarray]:
    model.eval()
    out = []
    for k in range(0, len(items), batch):
        b = to_torch(Bt.pack(items[k:k + batch], s_rows=s_rows), device)
        out.extend(model(b)["window"].cpu().numpy())
    return out


def evaluate(windows: list[np.ndarray], items: list[dict]) -> dict:
    pairs = collections.defaultdict(list)
    for w, it in zip(windows, items):
        if it["meta"].get("pair") is not None:
            pairs[it["meta"]["pair"]].append((Me.decision(w, it["query"]),
                                              it["meta"]["first_pick"]))
    errs = Me.errors(np.stack(windows), np.stack([i["target"] for i in items]),
                     np.stack([i["target_valid"] for i in items]))
    return {**errs, **Me.paired_both_correct(pairs)}


def baselines(items: list[dict], horizon: int = Bt.HORIZON) -> dict:
    return {"hold": evaluate([Me.hold(i["query"], horizon) for i in items], items),
            "replay": evaluate([Me.replay(i["support"], i["query"], horizon) for i in items], items)}


def overfit(items: list[dict], *, steps: int = 1500, width: int = 64, layers: int = 2,
            s_rows: int = 64, lr: float = 1e-3, seed: int = 0, log=print) -> tuple[PoseWindowPFN, dict]:
    torch.manual_seed(seed)
    model = PoseWindowPFN(width=width, layers=layers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    b = to_torch(Bt.pack(items, s_rows=s_rows))
    for step in range(1, steps + 1):
        model.train()
        loss, stats = window_loss(model(b)["window"], b["target"], b["target_valid"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps:
            res = evaluate(predict(model, items, s_rows), items)
            log(f"step {step}: loss {stats['loss']:.4f} trans {res['trans_mm']:.1f} mm "
                f"rot {res['rot_deg']:.2f} deg both-correct {res['both_correct']:.3f}")
            if res["both_correct"] >= 0.95 and res["trans_mm"] < 5:
                break
    return model, evaluate(predict(model, items, s_rows), items)


class _Stream(torch.utils.data.IterableDataset):
    def __init__(self, cfg: Bt.ItemConfig, batch: int, seed: int, s_rows: int):
        self.cfg, self.batch, self.seed, self.s_rows = cfg, batch, seed, s_rows

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, 0 if info is None else info.id])
        while True:
            items = Bt.sample_items(rng, self.cfg, self.batch)[:self.batch + 1]
            yield {k: torch.as_tensor(v) for k, v in Bt.pack(items, s_rows=self.s_rows).items()}


def train(out: Path, *, arm: str = "full", steps: int = 20000, batch: int = 64,
          width: int = 128, heads: int = 4, layers: int = 4, lr: float = 3e-4,
          warmup: int = 1000, workers: int = 8, seed: int = 0, s_rows: int = Bt.S_ROWS,
          dev_items: list[dict] | None = None, eval_every: int = 2000, log_every: int = 100,
          device: str = "cuda", faults=None) -> dict:
    out = Path(out)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    from .observation import CLEAN

    cfg = Bt.ItemConfig(programs=Sp.programs_for(arm), counts=Sp.count_matrix_for(arm),
                        branch_fraction=0.0 if arm == "nominal_only" else 0.3,
                        faults=faults or CLEAN, s_rows=s_rows)
    model = PoseWindowPFN(width=width, heads=heads, layers=layers, horizon=Bt.HORIZON,
                          use_support=arm != "scene_only").to(device)
    config = {"arm": arm, "steps": steps, "batch": batch, "lr": lr, "warmup": warmup,
              "seed": seed, "s_rows": s_rows, "model": model.config,
              "programs": len(cfg.programs), "counts": cfg.counts.kind,
              "branch_fraction": cfg.branch_fraction, "schema": "pose-window-v1"}
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, s / steps))))
    loader = torch.utils.data.DataLoader(_Stream(cfg, batch, seed, s_rows), batch_size=None,
                                         num_workers=workers, persistent_workers=workers > 0)
    history = (out / "history.jsonl").open("a")
    t0 = time.time()
    for step, b in enumerate(loader, start=1):
        b = {k: v.to(device, torch.float32 if v.is_floating_point() else None) for k, v in b.items()}
        model.train()
        loss, stats = window_loss(model(b)["window"], b["target"], b["target_valid"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0:
            history.write(json.dumps({"step": step, **stats, "lr": sched.get_last_lr()[0],
                                      "elapsed_s": round(time.time() - t0, 1)}) + "\n")
            history.flush()
        if dev_items is not None and (step % eval_every == 0 or step == steps):
            res = evaluate(predict(model, dev_items, s_rows, device), dev_items)
            history.write(json.dumps({"step": step, "dev": res}) + "\n")
            history.flush()
        if step >= steps:
            break
    torch.save({"config": model.config, "state": model.state_dict(), "steps": steps,
                "train_config": config}, out / "model.pt")
    (out / "DONE").write_text(json.dumps({"steps": steps, "elapsed_s": round(time.time() - t0)}) + "\n")
    return config


def load(run_dir: Path, device="cpu") -> PoseWindowPFN:
    ck = torch.load(Path(run_dir) / "model.pt", map_location=device, weights_only=False)
    model = PoseWindowPFN(**ck["config"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model
