#!/usr/bin/env python
"""Training-split statistics for rel_act, printed as --policy flags.

State statistics, and those of the chunk-relative target -- action[t + k]
minus state[t] in the relative dimensions, action[t + k] elsewhere, for every
k < chunk that stays inside the episode (what the loss sees; padded steps are
masked out of it). Fitted on the training episodes only: fitting on train and
validation together is a bug this project has already had once.

    relative_action_stats.py <dataset root> [--chunk 50]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_RELATIVE = [0, 1, 2, 8, 9, 10]


def statistics(episodes, chunk, relative):
    """(state_mean, state_std, target_mean, target_std) over `episodes`,
    each a (states [L, D], actions [L, D]) pair of numpy arrays. The target
    statistics are per chunk step, shape [chunk, D]."""
    import numpy as np

    states = np.concatenate([s for s, _ in episodes])
    width = states.shape[1]
    mask = np.zeros(width, dtype=bool)
    mask[relative] = True
    count = np.zeros(chunk)
    total = np.zeros((chunk, width))
    squares = np.zeros((chunk, width))
    for state, action in episodes:
        length = len(state)
        for k in range(min(chunk, length)):
            target = action[k:] - np.where(mask, state[: length - k], 0.0)
            count[k] += len(target)
            total[k] += target.sum(axis=0)
            squares[k] += (target**2).sum(axis=0)
    if (count == 0).any():
        raise SystemExit("some chunk step has no sample; the chunk is longer than every episode")
    mean = total / count[:, None]
    std = np.sqrt(np.maximum(squares / count[:, None] - mean**2, 0.0))
    # A floor, so a dimension that never moves (a parked arm's gripper) does
    # not divide by zero; 1 mm / 1e-3 of a quaternion component.
    floor = 1e-3
    return (states.mean(axis=0), np.maximum(states.std(axis=0), floor),
            mean, np.maximum(std, floor))


def main() -> int:
    import numpy as np
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--chunk", type=int, default=50)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "split_manifest.json").read_text())
    first, last = manifest["train"]["episodes"]
    tables = [pq.read_table(f, columns=["observation.state", "action", "episode_index"])
              for f in sorted((args.dataset / "data").glob("*/*.parquet"))]
    episodes = []
    for table in tables:
        index = np.asarray(table["episode_index"])
        state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False))
        action = np.stack(table["action"].to_numpy(zero_copy_only=False))
        for episode in np.unique(index):
            if first <= episode <= last:
                rows = index == episode
                episodes.append((state[rows].astype(np.float64), action[rows].astype(np.float64)))
    if len(episodes) != last - first + 1:
        raise SystemExit(f"found {len(episodes)} training episodes, expected {last - first + 1}")
    names = ("state_mean", "state_std", "target_mean", "target_std")
    for name, values in zip(names, statistics(episodes, args.chunk, DEFAULT_RELATIVE), strict=True):
        flat = np.asarray(values).reshape(-1)
        print(f"--policy.{name}={json.dumps([round(float(v), 8) for v in flat])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
