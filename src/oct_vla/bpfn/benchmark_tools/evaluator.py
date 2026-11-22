"""Scoring a PFN on a fixed set of items (a synthetic pack), under three support conditions."""

from __future__ import annotations

import collections

import numpy as np
import torch

from oct_vla.bpfn.benchmark_tools import metrics
from oct_vla.bpfn.data import batches
from oct_vla.bpfn.models.pose_window_pfn import PoseWindowPFN, to_torch
from oct_vla.bpfn.schema import COLOURS, empty_support


@torch.no_grad()
def predict(model: PoseWindowPFN, items: list[dict], s_rows: int, device: str = "cpu",
            batch: int = 64) -> list[np.ndarray]:
    """The predicted window [H, 2, 7] of every item."""
    model.eval()
    windows = []
    for start in range(0, len(items), batch):
        b = to_torch(batches.pack(items[start:start + batch], s_rows=s_rows), device)
        windows.extend(model(b)["window"].cpu().numpy())
    return windows


def evaluate(windows: list[np.ndarray], items: list[dict]) -> dict:
    """Pose errors over all items plus paired both-correct over the conflict pairs."""
    pairs = collections.defaultdict(list)
    for window, item in zip(windows, items):
        if item["meta"].get("pair") is not None:
            picked = metrics.decision(window, item["query"])
            pairs[item["meta"]["pair"]].append((picked, item["meta"]["first_pick"]))
    errors = metrics.errors(np.stack(windows), np.stack([i["target"] for i in items]),
                            np.stack([i["target_valid"] for i in items]))
    return {**errors, **metrics.paired_both_correct(pairs)}


def decisions(windows: list[np.ndarray], items: list[dict]) -> dict:
    """First-pick accuracy of the conflict-pair items per (manipulation, selection) family,
    and for colour rules the share of picks with the demonstrated colour."""
    correct = collections.Counter()
    total = collections.Counter()
    colour_right = colour_total = 0
    for window, item in zip(windows, items):
        meta = item["meta"]
        if meta.get("pair") is None:
            continue
        manipulation, eligibility = meta["program"].split("|")[:2]
        family = f"{manipulation}/{eligibility.split(':')[0]}"
        picked = metrics.decision(window, item["query"])
        total[family] += 1
        correct[family] += picked == meta["first_pick"]
        if eligibility.startswith("colour:") and picked is not None:
            query = item["query"]
            colour = COLOURS[query.colour[query.tracks.index(picked)]]
            colour_right += colour == eligibility.split(":")[1]
            colour_total += 1
    out = {"family_accuracy": {f: correct[f] / total[f] for f in sorted(total)}}
    if colour_total:
        out["right_colour"] = colour_right / colour_total
    return out


def baselines(items: list[dict], horizon: int = batches.HORIZON) -> dict:
    """Hold (both wrists stay) and replay (the support's motion, time-aligned)."""
    hold = [metrics.hold(i["query"], horizon) for i in items]
    replay = [metrics.replay(i["support"], i["query"], horizon) for i in items]
    return {"hold": evaluate(hold, items), "replay": evaluate(replay, items)}


class Evaluator:
    """Scores a model on fixed items with the correct, a shuffled and no support.

    Args:
        items: the items to score (``data/packs.py``).
        device: where the model runs.
        seed: seed of the support shuffle.
    """

    CONDITIONS = ("correct", "shuffled", "none")

    def __init__(self, items: list[dict], device: str = "cpu", seed: int = 0) -> None:
        self.items = items
        self.device = device
        self.seed = seed
        self.s_rows = max(1, max(len(i["support"].time) for i in items))   # the pack's own size

    def score(self, model: PoseWindowPFN, items: list[dict] | None = None) -> dict:
        items = self.items if items is None else items
        return evaluate(predict(model, items, self.s_rows, self.device), items)

    def run(self, model: PoseWindowPFN) -> dict:
        rng = np.random.default_rng(self.seed)
        results = {}
        for condition in self.CONDITIONS:
            results[condition] = self.score(model, self._condition_items(condition, rng))
        results["baselines"] = baselines(self.items)
        return results

    def _condition_items(self, condition: str, rng: np.random.Generator) -> list[dict]:
        if condition == "correct":
            return self.items
        if condition == "shuffled":
            perm = rng.permutation(len(self.items))
            return [dict(item, support=self.items[perm[k]]["support"])
                    for k, item in enumerate(self.items)]
        return [dict(item, support=empty_support(item["support"])) for item in self.items]
