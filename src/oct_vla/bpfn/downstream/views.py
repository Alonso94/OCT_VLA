"""Stage 3: pose tokens for executor training views.

For every frame of a recorded episode, one token entry per support:
- ``pose_window``: the frozen PFN's window (2H tokens) from the causal query prefix
  ending at that frame;
- ``support_table``: the support's own table (both wrists over time, every object's start
  and end), no PFN: the direct-demonstration baseline.
Supports are other recorded runs of the same rule, never the episode's own run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from oct_vla.bpfn.benchmark_tools.evaluator import predict
from oct_vla.bpfn.data import batches
from oct_vla.bpfn.schema import Episode, query_table, support_table

SUPPORT_ROWS = 32          # rows of a support_table entry
MAX_ENTITIES = 12          # objects and fixtures a support_table entry can hold


def choose_supports(pool: list[Path], own: Path, bank: int, seed: int,
                    episode_index: int) -> list[Path]:
    """``bank`` supports from ``pool``, never ``own``; fixed per (seed, episode)."""
    candidates = [p for p in pool if p.resolve() != own.resolve()]
    if len(candidates) < bank:
        raise SystemExit(f"episode {episode_index}: {len(candidates)} supports in the pool, "
                         f"need {bank}")
    rng = np.random.default_rng([seed, int(episode_index)])
    return [candidates[i] for i in rng.choice(len(candidates), bank, replace=False)]


class PoseTokenBuilder:
    """Per-frame token entries of one kind.

    Args:
        kind: ``pose_window`` or ``support_table``.
        model: the frozen PFN (``pose_window`` only).
        device: where the model runs.
    """

    def __init__(self, kind: str, model=None, device: str = "cpu") -> None:
        if kind == "pose_window" and model is None:
            raise ValueError("pose_window tokens need the PFN")
        self.kind = kind
        self.model = model
        self.device = device

    @property
    def rows(self) -> int:
        """Token rows of one entry."""
        if self.kind == "pose_window":
            return 2 * self.model.horizon
        return 2 * SUPPORT_ROWS + 2 * MAX_ENTITIES

    def frame_entries(self, query: Episode, supports: list[Episode]) -> list[list[np.ndarray]]:
        """``entries[t][k]``: the token entry of frame ``t`` under support ``k``."""
        from oct_vla.policies.conditioning import pose_tokens

        entries = [[None] * len(supports) for _ in range(query.T)]
        for k, support in enumerate(supports):
            if self.kind == "support_table":
                tokens = pose_tokens.support_tokens(support, SUPPORT_ROWS)
                for t in range(query.T):
                    entries[t][k] = tokens
                continue
            for t, window in enumerate(self._windows(query, support)):
                entries[t][k] = pose_tokens.window_tokens(window)
        return entries

    def _windows(self, query: Episode, support: Episode) -> list[np.ndarray]:
        horizon = self.model.horizon
        s_tab = support_table(support, batches.support_rows(support, batches.S_ROWS))
        items = []
        for t in range(query.T):
            items.append({"support": s_tab, "query": query_table(query, t, batches.HISTORY),
                          "target": np.zeros((horizon, 2, 7), np.float32),
                          "target_valid": np.zeros(horizon, bool), "meta": {}})
        return predict(self.model, items, batches.S_ROWS, self.device)
