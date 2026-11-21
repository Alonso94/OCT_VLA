"""Environment-agnostic controllers: they read the measured history as an ``Episode`` and
return both wrists' next poses ``[2, 7]``."""

from __future__ import annotations

import numpy as np

from oct_vla.bpfn.data import batches
from oct_vla.bpfn.schema import Episode, query_table, support_table


class PFNController:
    """The frozen PFN: one support per reset, ``k`` executed steps per prediction."""

    def __init__(self, model, support: Episode, k: int = 1, device: str = "cpu",
                 s_rows: int | None = None, history: int | None = None) -> None:
        self.model, self.k, self.device = model, k, device
        self.s_rows = s_rows or batches.S_ROWS
        self.history = history or batches.HISTORY
        self.support_tab = support_table(support, batches.support_rows(support, self.s_rows))
        self._memory = None
        self._queue: list[np.ndarray] = []
        self.windows: list[np.ndarray] = []

    def reset(self) -> None:
        self._memory, self._queue, self.windows = None, [], []

    def __call__(self, ep: Episode) -> np.ndarray:
        import torch

        from oct_vla.bpfn.models.pose_window_pfn import to_torch

        if not self._queue:
            item = {"support": self.support_tab, "query": query_table(ep, ep.T - 1, self.history),
                    "target": np.zeros((self.model.horizon, 2, 7), np.float32),
                    "target_valid": np.zeros(self.model.horizon, bool), "meta": {}}
            b = to_torch(batches.pack([item], s_rows=self.s_rows), self.device)
            with torch.no_grad():
                if self._memory is None:
                    self._memory = self.model.encode_support(b)
                window = self.model.predict(self._memory, b)["window"][0].cpu().numpy()
            self.windows.append(window)
            self._queue = [window[h] for h in range(min(self.k, len(window)))]
        return self._queue.pop(0)


class HoldController:
    def __call__(self, ep: Episode, obs=None) -> np.ndarray:
        return ep.wrist[-1].copy()
