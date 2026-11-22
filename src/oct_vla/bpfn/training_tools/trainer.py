"""Stage-1 training of the pose-window PFN on synthetic items.

- ``PFNTrainer``: items from the episode bank (or generated on the fly), AdamW with warmup
  and cosine decay, a scalar log every ``log_every`` steps, the dev pack every
  ``eval_every`` steps, one checkpoint (``model.pt``) overwritten at each evaluation.
- ``overfit``: the CPU memorisation gate on fixed conflict pairs (>= 90 % both-correct).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from oct_vla.bpfn.benchmark_tools.evaluator import evaluate, predict
from oct_vla.bpfn.data import batches, splits
from oct_vla.bpfn.data.bank import Bank
from oct_vla.bpfn.data.syn_data import generator
from oct_vla.bpfn.data.syn_data.observation import CLEAN
from oct_vla.bpfn import models
from oct_vla.bpfn.models.pose_window_pfn import PoseWindowPFN, to_torch
from oct_vla.bpfn.training_tools.data import BankSampler, OnTheFlySampler, build_dataloader
from oct_vla.bpfn.training_tools.losses.window import WindowLoss
from oct_vla.bpfn.training_tools.validator import TrainingValidator


@dataclass
class TrainConfig:
    """One training run (an arm x seed).

    ``reuse``, ``pair_extra`` and ``conflict_fraction`` default by data source: from a bank
    one item per support/query pair and conflict pairs in half the draws; on the fly each
    generated pair is reused 8 times, with one extra item per conflict member and conflict
    pairs in 70 % of the draws.
    """

    out: Path
    arm: str = "full"
    seed: int = 0
    steps: int = 80000
    batch: int = 64
    width: int = 128
    heads: int = 4
    layers: int = 4
    lr: float = 3e-4
    warmup: int = 1000
    workers: int = 8
    s_rows: int = batches.S_ROWS
    eval_every: int = 5000
    log_every: int = 100
    device: str = "cuda"
    bank: Path | None = None
    reuse: int | None = None
    pair_extra: int | None = None
    conflict_fraction: float | None = None
    per_support: int = 1        # queries per support in a batch (bank only)
    fresh_supports: bool = False  # bank queries, but every support generated anew
    tf32: bool = False          # TF32 matmuls on CUDA
    compile: bool = False       # torch.compile the training forward (fuses small kernels)
    arch: str = "axial"         # "axial" (PoseWindowPFN) or "table" (TablePFN)
    task_read: str = "cell"     # table: how the query reads the demonstration (cell / film / cross)
    output: str = "head"        # table: "head" or "rows"
    task_tokens: int = 1        # table: pooled demonstration tokens
    lr_floor: float = 0.0       # the cosine decays to lr * lr_floor, not to zero
    clip: float = 1.0           # gradient-norm clip


class PFNTrainer:
    """Builds data, model, loss, optimiser, scheduler and validator for one run.

    Args:
        cfg: the run.
        val_packs: fixed packs scored during training, by name (dev, held-out, count);
            logged only, never used to pick a checkpoint. None: no validation.
    """

    def __init__(self, cfg: TrainConfig, val_packs: dict[str, list[dict]] | None = None) -> None:
        self.cfg = cfg
        self.out = Path(cfg.out)
        if self.out.exists():
            raise SystemExit(f"Refusing to overwrite {self.out}")
        if cfg.per_support > 1 and (cfg.bank is None or cfg.batch % cfg.per_support):
            raise SystemExit("per_support > 1 needs a bank and a batch divisible by it")
        if cfg.fresh_supports and cfg.bank is None:
            raise SystemExit("fresh_supports draws queries from a bank: pass one")
        if cfg.tf32 and cfg.device != "cpu":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Data. The bank is loaded before the loader workers fork, so they share it.
        self.bank = Bank(cfg.bank) if cfg.bank is not None else None
        self.out.mkdir(parents=True)
        self.item_cfg = self._item_config()
        if self.bank is not None:
            sampler = BankSampler(self.bank, self.item_cfg, fresh_supports=cfg.fresh_supports)
        else:
            sampler = OnTheFlySampler(self.item_cfg)

        # Model, loss, optimisation. Seeded first: torch's own start-up seed is random, so
        # without this the initial weights would differ on every launch of the same seed.
        torch.manual_seed(cfg.seed)
        self.model = models.build(self._model_config()).to(cfg.device)
        self.loss_fn = WindowLoss()
        self.config = self._config_record()
        (self.out / "config.json").write_text(json.dumps(self.config, indent=1) + "\n")
        # The training forward may be compiled; validation and checkpoints use self.model
        # (the same parameters).
        self.train_model = torch.compile(self.model) if cfg.compile else self.model
        fused = cfg.device != "cpu"          # one fused kernel for the whole update
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=0.01,
                                           fused=fused)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_factor)
        self.dataloader = build_dataloader(sampler, cfg.batch, cfg.seed, cfg.s_rows, cfg.workers,
                                           cfg.per_support)

        # Validation (logged only; never used to pick a checkpoint).
        self.validators = {}
        for name, items in (val_packs or {}).items():
            self.validators[name] = TrainingValidator(items, cfg.device, shuffled=name == "dev")

    def _model_config(self) -> dict:
        cfg = self.cfg
        use_support = cfg.arm != "scene_only"
        if cfg.arch == "axial":
            return dict(arch="axial", width=cfg.width, heads=cfg.heads, layers=cfg.layers,
                        horizon=batches.HORIZON, use_support=use_support)
        return dict(arch="table", width=cfg.width, heads=cfg.heads, support_layers=cfg.layers,
                    query_layers=cfg.layers, horizon=batches.HORIZON, task_read=cfg.task_read,
                    output=cfg.output, task_tokens=cfg.task_tokens, use_support=use_support)

    def _item_config(self) -> batches.ItemConfig:
        cfg = self.cfg
        reuse = cfg.reuse
        if reuse is None:
            reuse = 1 if self.bank is not None else 8
        pair_extra = cfg.pair_extra
        if pair_extra is None:
            pair_extra = 1 if reuse > 1 else 0
        conflict_fraction = cfg.conflict_fraction
        if conflict_fraction is None:
            conflict_fraction = 0.7 if reuse > 1 else 0.5
        return batches.ItemConfig(programs=splits.programs_for(cfg.arm),
                                  counts=splits.count_matrix_for(cfg.arm),
                                  branch_fraction=0.0 if cfg.arm == "nominal_only" else 0.3,
                                  faults=CLEAN, s_rows=cfg.s_rows, reuse=reuse,
                                  pair_extra=pair_extra, conflict_fraction=conflict_fraction)

    def _config_record(self) -> dict:
        cfg, item_cfg = self.cfg, self.item_cfg
        bank = None
        if self.bank is not None:
            bank = {"path": str(cfg.bank), "episodes": self.bank.episodes,
                    "spec": self.bank.manifest["spec"]}
        return {"arm": cfg.arm, "steps": cfg.steps, "batch": cfg.batch, "lr": cfg.lr,
                "warmup": cfg.warmup, "seed": cfg.seed, "s_rows": cfg.s_rows,
                "model": self.model.config, "programs": len(item_cfg.programs),
                "counts": item_cfg.counts.kind, "branch_fraction": item_cfg.branch_fraction,
                "schema": "pose-window-v1", "reuse": item_cfg.reuse,
                "pair_extra": item_cfg.pair_extra, "conflict_fraction": item_cfg.conflict_fraction,
                "workers": cfg.workers, "bank": bank, "per_support": cfg.per_support,
                "fresh_supports": cfg.fresh_supports, "lambda_rot": self.loss_fn.lambda_rot,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "run": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}}

    def _lr_factor(self, step: int) -> float:
        """Linear warmup, then cosine decay to ``lr_floor`` (a fraction of lr) at ``steps``."""
        warmup = min(1.0, (step + 1) / self.cfg.warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, step / self.cfg.steps)))
        floor = self.cfg.lr_floor
        return warmup * (floor + (1 - floor) * cosine)

    def _save_checkpoint(self, step: int) -> None:
        torch.save({"config": self.model.config, "state": self.model.state_dict(), "steps": step,
                    "train_config": self.config}, self.out / "model.pt")

    def _log(self, record: dict) -> None:
        self.history.write(json.dumps(record) + "\n")
        self.history.flush()

    def _to_device(self, batch: dict) -> dict:
        moved = {}
        for key, value in batch.items():
            dtype = torch.float32 if value.is_floating_point() else None
            moved[key] = value.to(self.cfg.device, dtype)
        return moved

    def _log_window(self, step: int, sums: dict, steps: int, waited: float, elapsed: float,
                    start: float) -> None:
        """Means over the last ``steps`` training steps (one GPU sync per window)."""
        record = {"step": step}
        for key, total in sums.items():
            record[key] = round(float(total) / steps, 5)
        record["clipped"] = record.pop("clipped_sum")          # share of steps clipped
        record["lr"] = self.scheduler.get_last_lr()[0]
        record["elapsed_s"] = round(time.time() - start, 1)
        record["data_wait_s"] = round(waited / steps, 4)
        record["compute_s"] = round((elapsed - waited) / steps, 4)
        record["items_per_s"] = round(steps * self.cfg.batch / elapsed, 1)
        if self.cfg.device != "cpu":
            record["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        self._log(record)

    def train(self) -> dict:
        cfg = self.cfg
        self.history = (self.out / "history.jsonl").open("a")
        start = time.time()
        sums: dict = {}                  # per-window sums of the step statistics (tensors)
        waited = 0.0                     # seconds blocked on the loader in this window
        window_start = mark = time.time()
        for step, batch in enumerate(self.dataloader, start=1):
            waited += time.time() - mark
            batch = self._to_device(batch)

            self.train_model.train()
            loss, stats = self.loss_fn(self.train_model(batch)["window"], batch["target"],
                                       batch["target_valid"])
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip)
            self.optimizer.step()
            self.scheduler.step()
            stats["grad_norm"] = grad_norm.detach()
            stats["clipped_sum"] = (grad_norm > cfg.clip).float()
            for key, value in stats.items():
                sums[key] = sums.get(key, 0.0) + value
            mark = time.time()

            if step % cfg.log_every == 0:
                if cfg.device != "cpu":
                    torch.cuda.synchronize()
                elapsed = time.time() - window_start
                self._log_window(step, sums, cfg.log_every, waited, elapsed, start)
                sums, waited = {}, 0.0
                window_start = mark = time.time()
            if self.validators and (step % cfg.eval_every == 0 or step == cfg.steps):
                record = {"step": step}
                for name, validator in self.validators.items():
                    record[name] = validator.run_validation(self.model)
                self._log(record)
                window_start = mark = time.time()       # validation is not training time
            if step % cfg.eval_every == 0 and step < cfg.steps:
                self._save_checkpoint(step)        # one file, overwritten: a timeout keeps the latest
            if step >= cfg.steps:
                break

        self._save_checkpoint(cfg.steps)
        done = {"steps": cfg.steps, "elapsed_s": round(time.time() - start)}
        (self.out / "DONE").write_text(json.dumps(done) + "\n")
        return self.config


def conflict_items(n_pairs: int, seed: int, programs=None, counts=None,
                   s_rows: int = 64) -> list[dict]:
    """``n_pairs`` conflict pairs generated on the fly (the overfit gate's data)."""
    cfg = batches.ItemConfig(programs=programs or splits.programs_for("full"),
                             counts=counts or splits.CountMatrix(), s_rows=s_rows)
    rng = np.random.default_rng(seed)
    items: list[dict] = []
    while len(items) < 2 * n_pairs:
        try:
            items.extend(batches.sample_pair(rng, cfg))
        except generator.Infeasible:
            continue
    return items


def overfit(items: list[dict], *, steps: int = 1500, width: int = 64, layers: int = 2,
            s_rows: int = 64, lr: float = 1e-3, seed: int = 0, log=print) -> tuple[PoseWindowPFN, dict]:
    """Trains on fixed items until memorised; returns the model and its scores on them."""
    torch.manual_seed(seed)
    model = PoseWindowPFN(width=width, layers=layers)
    loss_fn = WindowLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    batch = to_torch(batches.pack(items, s_rows=s_rows))
    for step in range(1, steps + 1):
        model.train()
        loss, stats = loss_fn(model(batch)["window"], batch["target"], batch["target_valid"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps:
            scores = evaluate(predict(model, items, s_rows), items)
            log(f"step {step}: loss {float(stats['loss']):.4f} trans {scores['trans_mm']:.1f} mm "
                f"rot {scores['rot_deg']:.2f} deg both-correct {scores['both_correct']:.3f}")
            if scores["both_correct"] >= 0.95 and scores["trans_mm"] < 5:
                break
    return model, evaluate(predict(model, items, s_rows), items)
