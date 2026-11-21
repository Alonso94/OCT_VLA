"""The executor's pose tokens at evaluation (stage 3), from the measured RoboTwin history."""

from __future__ import annotations

import numpy as np

from oct_vla.bpfn.downstream import tokens
from oct_vla.bpfn.envs.controllers import PFNController
from oct_vla.bpfn.envs.robotwin.history import History


class PoseTokenFutureSource:
    """The executor's pose tokens at evaluation (stage 3): the frozen PFN's window from the
    measured history (``pose_window``), or the support's own table (``support_table``).
    One support per rollout, in bank entry 0 (the entry the executor reads at evaluation);
    the other entries are masked. Configured from the training view's provenance, and the
    PFN checkpoint must be the very one the view was built with."""

    def __init__(self, provenance: dict, support: str, device: str = "cpu") -> None:
        import hashlib
        from pathlib import Path

        from oct_vla.bpfn.data.real_data.robotwin import read_recording

        self.kind = provenance["kind"]
        self.bank, self.rows = int(provenance["bank"]), int(provenance["rows"])
        self.support_path = str(support)
        self.support = read_recording(support)
        self.model = self.controller = None
        if self.kind == "pose_window":
            from oct_vla.bpfn.models.pose_window_pfn import PoseWindowPFN

            run = Path(provenance["run"])
            sha = hashlib.sha256((run / "model.pt").read_bytes()).hexdigest()[:16]
            if sha != provenance["checkpoint_sha256_16"]:
                raise ValueError(f"{run} is not the PFN checkpoint the view was built with")
            self.model = PoseWindowPFN.from_run(run, device)
            self.controller = PFNController(self.model, self.support, k=1, device=device)
        elif self.kind == "support_table":
            self.static = tokens.support_tokens(self.support, int(provenance["support_rows"]))
        else:
            raise ValueError(f"not a pose-token view: {self.kind!r}")
        self.history: History | None = None
        self._last_time = None

    def reset(self, scene_seed: int, observation) -> list[str]:
        self.history = History()
        self._last_time = None
        if self.controller is not None:
            self.controller.reset()
        return [self.support_path]

    def __call__(self, observation) -> tuple[np.ndarray, np.ndarray, float]:
        if self.history is None:
            raise RuntimeError("PoseTokenFutureSource called before reset()")
        if self._last_time != observation.timestamp:          # one row per measurement
            self.history.add(observation)
            self._last_time = observation.timestamp
        if self.kind == "support_table":
            entry = self.static
        else:
            entry = tokens.window_tokens(self.controller(self.history.episode()))
            self.controller._queue = []                       # a fresh window every step
        empty = entry[:0]
        packed, mask = tokens.bank([entry] + [empty] * (self.bank - 1), self.rows)
        return packed, mask, 1.0
