"""The observed table, built from a physical trajectory (plan stage 0.2).

The physical episode is generated first; this decides what a tracker would have reported.
Object tracks get contiguous **missing** intervals (no measurement: the cell keeps its last
measured pose, ``observed`` is False and ``age`` grows) and **stale** intervals (a
measurement that repeats an old value: same as missing to the model, since a tracker
cannot tell either). Both wrists come from proprioception and are never corrupted.

Faults are placed around the episode's own events (approach, exchange, release) with
probability ``near_event``, otherwise anywhere, and also on tracks that are irrelevant to
the task. Their labels never reach the inputs: the model sees only ``observed`` and
``age``, which a real tracker also reports. Physical targets are unchanged: an
observation-only fault does not alter what the expert does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import Episode


@dataclass(frozen=True)
class FaultConfig:
    rate: float = 0.0            # expected faulted intervals per object track
    length: tuple[int, int] = (3, 15)   # rows per interval
    near_event: float = 0.5


CLEAN = FaultConfig()


def observe(ep: Episode, rng: np.random.Generator, cfg: FaultConfig = CLEAN):
    """``(pose [T, C, 7], observed [T, C], age [T, C])`` over the table's columns
    (both wrists, then the episode's objects)."""
    T, N = ep.T, len(ep.track_ids)
    pose = np.concatenate([ep.wrist, ep.objects], axis=1).copy()
    observed = np.ones((T, N + 2), bool)
    if cfg.rate > 0:
        event_rows = [e["row"] for e in ep.annotations.get("events", [])]
        for j in range(N):
            for _ in range(rng.poisson(cfg.rate)):
                n = int(rng.integers(cfg.length[0], cfg.length[1] + 1))
                if event_rows and rng.random() < cfg.near_event:
                    start = int(np.clip(rng.choice(event_rows) - rng.integers(0, n), 1, T - 1))
                else:
                    start = int(rng.integers(1, T))
                observed[start:start + n, j + 2] = False
    age = np.zeros((T, N + 2), np.float32)
    for c in range(N + 2):
        last = 0
        for t in range(T):
            if observed[t, c]:
                last = t
            else:
                pose[t, c] = pose[last, c]
                age[t, c] = ep.time[t] - ep.time[last]
    return pose, observed, age
