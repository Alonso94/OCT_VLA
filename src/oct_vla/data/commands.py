"""The commands the expert actually applied, per physics tick, for both arms.

A recorded episode stores *measured* states at 15 Hz. The commands that produced them
(joint references and gripper commands, 250 Hz) were kept in memory by the oracle
(``ExecutedMotion``) and reduced to phase spans. This module keeps them:

- ``q`` / ``qdot`` ``[n_ticks, 2, dof]`` and ``gripper`` ``[n_ticks, 2]``: the reference
  each arm was commanded on each tick (left, right). The moving arm's comes from the
  motion's log; the other arm's is its ``held_q`` / ``held_gripper`` (its measured state at
  the motion's start, which it is told to hold).
- ``active`` ``[n_ticks]``: which arm the motion moved (0 left, 1 right).
- ``frame_ticks`` ``[T]``: each recorded frame's capture tick. A frame captured at tick k
  observes the state after k ticks; the command driving its next interval is index k.

Written beside ``episode.json`` as ``episode.commands.npz`` (``COMMANDS_VERSION``).
Commanded wrist poses come from these by forward kinematics (``pose_window/real.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

COMMANDS_VERSION = "commands-v1"
COMMANDS_FILE = "episode.commands.npz"
SIDES = ("left", "right")


@dataclass(frozen=True)
class CommandLog:
    q: np.ndarray
    qdot: np.ndarray
    gripper: np.ndarray
    active: np.ndarray
    frame_ticks: np.ndarray
    dt: float
    version: str = COMMANDS_VERSION

    def __post_init__(self) -> None:
        n = len(self.q)
        if self.q.shape[1] != 2 or self.qdot.shape != self.q.shape:
            raise ValueError(f"q {self.q.shape} / qdot {self.qdot.shape} are not [n, 2, dof]")
        if self.gripper.shape != (n, 2) or self.active.shape != (n,):
            raise ValueError("gripper / active do not match the tick count")
        if len(self.frame_ticks) and (self.frame_ticks.min() < 0 or self.frame_ticks.max() > n):
            raise ValueError(f"frame ticks fall outside the {n} commanded ticks")

    @property
    def ticks(self) -> int:
        return len(self.q)

    def at_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """The command driving each frame's next interval: ``q [T, 2, dof]``,
        ``gripper [T, 2]``. The last frame (captured after the final tick) repeats the
        final command, and is flagged by ``frame_ticks == ticks``."""
        idx = np.minimum(self.frame_ticks, self.ticks - 1)
        return self.q[idx], self.gripper[idx]


def from_motions(motions: Iterable, frame_ticks: Iterable[int], dt: float) -> CommandLog:
    """Concatenate ``ExecutedMotion``s in execution order into a per-tick log."""
    q, qdot, grip, active = [], [], [], []
    for motion in motions:
        if motion.held_q is None or motion.held_gripper is None:
            raise ValueError("a motion without its held arm's reference cannot be logged")
        side = SIDES.index(motion.side)
        n = motion.ticks
        mq, mqd = np.asarray(motion.q, float), np.asarray(motion.qdot, float)
        held = np.broadcast_to(np.asarray(motion.held_q, float), mq.shape)
        block_q, block_qd = np.empty((n, 2, mq.shape[1])), np.zeros((n, 2, mq.shape[1]))
        block_q[:, side], block_q[:, 1 - side] = mq, held
        block_qd[:, side] = mqd
        block_g = np.empty((n, 2))
        block_g[:, side], block_g[:, 1 - side] = motion.gripper, motion.held_gripper
        q.append(block_q)
        qdot.append(block_qd)
        grip.append(block_g)
        active.append(np.full(n, side, np.int8))
    if not q:
        raise ValueError("no motions to log")
    return CommandLog(np.concatenate(q), np.concatenate(qdot), np.concatenate(grip),
                      np.concatenate(active), np.asarray(list(frame_ticks), np.int64), float(dt))


def write_commands(log: CommandLog, directory: str | Path) -> Path:
    path = Path(directory) / COMMANDS_FILE
    np.savez_compressed(path, q=log.q.astype(np.float32), qdot=log.qdot.astype(np.float32),
                        gripper=log.gripper.astype(np.float32), active=log.active,
                        frame_ticks=log.frame_ticks, dt=np.float64(log.dt),
                        version=np.array(log.version))
    return path


def read_commands(directory: str | Path) -> CommandLog | None:
    """The episode's command log, or None for a recording made before command logging."""
    path = Path(directory) / COMMANDS_FILE
    if not path.exists():
        return None
    with np.load(path) as z:
        version = str(z["version"])
        if version != COMMANDS_VERSION:
            raise ValueError(f"{path}: command log {version}, expected {COMMANDS_VERSION}")
        return CommandLog(z["q"].astype(float), z["qdot"].astype(float),
                          z["gripper"].astype(float), z["active"], z["frame_ticks"],
                          float(z["dt"]), version)
