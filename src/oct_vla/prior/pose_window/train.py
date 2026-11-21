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
    """Packed batches from ``sampler`` (a ``bank.BankSampler``) or, without one, from
    episodes generated on the fly."""

    def __init__(self, cfg: Bt.ItemConfig, batch: int, seed: int, s_rows: int, sampler=None):
        self.cfg, self.batch, self.seed, self.s_rows = cfg, batch, seed, s_rows
        self.sampler = sampler

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, 0 if info is None else info.id])
        pending: list[dict] = []
        while True:
            # Reused groups overshoot a batch; the rest starts the next one.
            if len(pending) < self.batch:
                pending += (self.sampler.sample_items(rng, self.batch - len(pending))
                            if self.sampler is not None
                            else Bt.sample_items(rng, self.cfg, self.batch - len(pending)))
            items, pending = pending[:self.batch], pending[self.batch:]
            yield {k: torch.as_tensor(v) for k, v in Bt.pack(items, s_rows=self.s_rows).items()}


def train(out: Path, *, arm: str = "full", steps: int = 80000, batch: int = 64,
          width: int = 128, heads: int = 4, layers: int = 4, lr: float = 3e-4,
          warmup: int = 1000, workers: int = 8, seed: int = 0, s_rows: int = Bt.S_ROWS,
          dev_items: list[dict] | None = None, eval_every: int = 5000, log_every: int = 100,
          device: str = "cuda", faults=None, reuse: int | None = None,
          pair_extra: int | None = None, conflict_fraction: float | None = None,
          bank: Path | None = None) -> dict:
    """``bank``: a pre-generated episode bank (``bank.py``); items are then drawn from it
    and the defaults are recipe 1's item mix (one item per support/query pair, conflict
    pairs half the draws), since generation is no longer the cost. Without a bank,
    episodes are generated on the fly and reused 8 times."""
    out = Path(out)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    from .observation import CLEAN

    if reuse is None:
        reuse = 1 if bank is not None else 8
    store = None
    if bank is not None:
        from . import bank as Bk

        store = Bk.Bank(bank)       # in RAM before the workers fork; they share it
    out.mkdir(parents=True)

    cfg = Bt.ItemConfig(programs=Sp.programs_for(arm), counts=Sp.count_matrix_for(arm),
                        branch_fraction=0.0 if arm == "nominal_only" else 0.3,
                        faults=faults or CLEAN, s_rows=s_rows, reuse=reuse,
                        pair_extra=(1 if reuse > 1 else 0) if pair_extra is None else pair_extra,
                        conflict_fraction=((0.7 if reuse > 1 else 0.5) if conflict_fraction is None
                                           else conflict_fraction))
    model = PoseWindowPFN(width=width, heads=heads, layers=layers, horizon=Bt.HORIZON,
                          use_support=arm != "scene_only").to(device)
    config = {"arm": arm, "steps": steps, "batch": batch, "lr": lr, "warmup": warmup,
              "seed": seed, "s_rows": s_rows, "model": model.config,
              "programs": len(cfg.programs), "counts": cfg.counts.kind,
              "branch_fraction": cfg.branch_fraction, "schema": "pose-window-v1",
              "reuse": reuse, "pair_extra": cfg.pair_extra,
              "conflict_fraction": cfg.conflict_fraction, "workers": workers,
              "bank": None if store is None else {"path": str(bank), "episodes": store.episodes,
                                                  "spec": store.manifest["spec"]},
              "lambda_rot": __import__("oct_vla.prior.pose_window.model", fromlist=["LAMBDA_ROT"]).LAMBDA_ROT}
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, s / steps))))
    sampler = None if store is None else Bk.BankSampler(store, cfg)
    loader = torch.utils.data.DataLoader(_Stream(cfg, batch, seed, s_rows, sampler), batch_size=None,
                                         num_workers=workers, persistent_workers=workers > 0)
    history = (out / "history.jsonl").open("a")
    t0 = time.time()
    waited = computed = 0.0
    mark = time.time()
    for step, b in enumerate(loader, start=1):
        got = time.time()
        waited += got - mark                       # time blocked on the data loader
        b = {k: v.to(device, torch.float32 if v.is_floating_point() else None) for k, v in b.items()}
        model.train()
        loss, stats = window_loss(model(b)["window"], b["target"], b["target_valid"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if device != "cpu":
            torch.cuda.synchronize()
        mark = time.time()
        computed += mark - got
        if step % log_every == 0:
            history.write(json.dumps({"step": step, **stats, "lr": sched.get_last_lr()[0],
                                      "elapsed_s": round(time.time() - t0, 1),
                                      "data_wait_s": round(waited / log_every, 4),
                                      "compute_s": round(computed / log_every, 4)}) + "\n")
            history.flush()
            waited = computed = 0.0
        if dev_items is not None and (step % eval_every == 0 or step == steps):
            dev_rows = max(len(i["support"].time) for i in dev_items)   # the pack's own size
            res = evaluate(predict(model, dev_items, dev_rows, device), dev_items)
            history.write(json.dumps({"step": step, "dev": res}) + "\n")
            history.flush()
        if step % eval_every == 0 and step < steps:
            # One checkpoint, overwritten: a timeout keeps the latest, never several.
            torch.save({"config": model.config, "state": model.state_dict(), "steps": step,
                        "train_config": config}, out / "model.pt")
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
