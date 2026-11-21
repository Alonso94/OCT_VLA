"""Pose tokens for the executor (stage 3): the PFN's window, or the raw support table.

Two conditionings share one token format and one embedding, so the comparison between
them is about *what* the executor reads, not how:

- ``pose_window``: the frozen BimanualPFN's predicted window, 2H tokens (one per wrist and
  horizon), ordered ``h * 2 + arm``;
- ``support_table``: the direct-demonstration baseline, the support's own measured wrist
  trajectories (subsampled rows) plus every object's start and end pose, no PFN.

Token columns (``POSE_TOKEN_DIM`` = 19)::

    [ 0: 3]  position (workcell, m)          [13]     time in [0, 1] (h / (H-1), or row)
    [ 3: 7]  unit quaternion xyzw            [14:19]  colour one-hot (unknown, red, blue,
    [ 7:13]  role one-hot: left wrist, right          green, yellow)
             wrist, movable, fixture,
             window left, window right

They enter through the future branch (``object_future_kind`` = ``pose_window`` or
``support_table``): a zero-initialised K/V term per hooked layer, so the executor starts
as its unconditioned self and an all-masked set contributes exactly zero.

**Banks.** A view stores K independently drawn supports per frame
(``object_future_bank``); training takes one entry per sample (so pairings change across
epochs), evaluation takes entry 0. **Residual actions** (``action_residual``): the policy
learns the expert chunk minus the PFN window, in the action's own normalised space, for
the 14 pose columns; grippers stay absolute. Beyond the window's H steps the last
predicted pose is held. Rotation residuals are sign-aligned quaternion differences in that
space (invertible exactly; a log-map variant is a later option).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from oct_vla.bpfn.downstream.tokens import (COLOUR_COLS, POS_CENTRE, POS_SCALE, POSE_KINDS,  # noqa: F401
                                            POSE_TOKEN_DIM, ROLES, bank, support_tokens,
                                            window_tokens)

from .kv import KVAttention

#: eeabs action layout: [L xyz, L xyzw, L grip, R xyz, R xyzw, R grip].
POSE_COLS = tuple(range(0, 7)) + tuple(range(8, 15))
GRIP_COLS = (7, 15)


def tokens_to_window(tokens: Tensor) -> Tensor:
    """The inverse for the executor: ``[B, 2H, D]`` -> ``[B, H, 2, 7]``."""
    B, n, _ = tokens.shape
    if n % 2:
        raise ValueError("a pose window has an even token count (two wrists per horizon)")
    return tokens[..., :7].reshape(B, n // 2, 2, 7)


# --- the executor side (torch) -------------------------------------------------------

def select_bank(tokens: Tensor, mask: Tensor, bank_size: int, training: bool,
                generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """``[B, K*R, D]`` -> one entry per sample ``[B, R, D]``: random in training, entry 0
    otherwise. An all-masked entry is never chosen when another entry has tokens."""
    B, n, D = tokens.shape
    if n % bank_size:
        raise ValueError(f"{n} tokens do not split into a bank of {bank_size}")
    R = n // bank_size
    t = tokens.reshape(B, bank_size, R, D)
    m = mask.reshape(B, bank_size, R)
    if training and bank_size > 1:
        filled = m.sum(-1) > 0                                         # [B, K]
        weights = filled.float() + (~filled.any(-1, keepdim=True)).float()
        k = torch.multinomial(weights, 1, generator=generator).squeeze(-1)
    else:
        k = torch.zeros(B, dtype=torch.long, device=tokens.device)
    pick = torch.arange(B, device=tokens.device)
    return t[pick, k], m[pick, k]


class ActionCodec:
    """Expert chunk <-> residual about the PFN window, in the action's normalised space.

    ``normalise(a) = (a - offset) / scale`` per action column (MEAN_STD: offset = mean,
    scale = std; MIN_MAX to [-1, 1]: offset = (lo + hi) / 2, scale = (hi - lo) / 2).
    """

    def __init__(self, offset, scale) -> None:
        self.offset = torch.as_tensor(offset, dtype=torch.float32)
        self.scale = torch.as_tensor(scale, dtype=torch.float32).clamp(min=1e-8)
        if self.offset.shape != (16,) or self.scale.shape != (16,):
            raise ValueError("the residual codec expects the 16-column eeabs action")

    @classmethod
    def from_stats(cls, stats: dict, mode: str) -> "ActionCodec":
        if mode == "MEAN_STD":
            return cls(stats["mean"], stats["std"])
        if mode == "MIN_MAX":
            lo, hi = np.asarray(stats["min"], float), np.asarray(stats["max"], float)
            return cls((lo + hi) / 2, (hi - lo) / 2)
        raise ValueError(f"no residual codec for normalisation {mode!r}")

    def _window_actions(self, window: Tensor, steps: int, like: Tensor) -> Tensor:
        """``[B, H, 2, 7]`` -> raw eeabs-shaped ``[B, steps, 16]`` (grippers zero), the last
        pose held beyond H."""
        B, H = window.shape[:2]
        idx = torch.clamp(torch.arange(steps, device=window.device), max=H - 1)
        w = window[:, idx]                                             # [B, steps, 2, 7]
        out = like.new_zeros(B, steps, 16)
        out[..., 0:7], out[..., 8:15] = w[:, :, 0], w[:, :, 1]
        return out

    def encode(self, action_norm: Tensor, window: Tensor) -> Tensor:
        """Normalised expert chunk ``[B, C, >=16]`` -> normalised residual, same shape. Only
        the 16 eeabs columns are touched (GR00T pads its action past them)."""
        if action_norm.shape[-1] > 16:
            out = action_norm.clone()
            out[..., :16] = self.encode(action_norm[..., :16], window)
            return out
        off, sc = self.offset.to(action_norm), self.scale.to(action_norm)
        raw = action_norm * sc + off
        w = self._window_actions(window.to(raw), raw.shape[1], raw)
        # q and -q are one rotation: align the *target* to the window's sign, so the
        # residual is small and decode (window + residual) returns the same rotation.
        raw = raw.clone()
        for q in (slice(3, 7), slice(11, 15)):
            flip = (w[..., q] * raw[..., q]).sum(-1, keepdim=True) < 0
            raw[..., q] = torch.where(flip, -raw[..., q], raw[..., q])
        out = action_norm.clone()
        cols = list(POSE_COLS)
        out[..., cols] = (raw[..., cols] - w[..., cols]) / sc[cols]
        return out

    def decode(self, residual_norm: Tensor, window: Tensor) -> Tensor:
        """Normalised residual ``[B, C, >=16]`` -> normalised absolute chunk."""
        if residual_norm.shape[-1] > 16:
            out = residual_norm.clone()
            out[..., :16] = self.decode(residual_norm[..., :16], window)
            return out
        off, sc = self.offset.to(residual_norm), self.scale.to(residual_norm)
        w = self._window_actions(window.to(residual_norm), residual_norm.shape[1], residual_norm)
        out = residual_norm.clone()
        cols = list(POSE_COLS)
        out[..., cols] = residual_norm[..., cols] + (w[..., cols] - off[cols]) / sc[cols]
        return out


class PoseTokenAttention(KVAttention):
    """The future K/V branch over pose tokens: one shared pose encoder for every token,
    plus a projection of the categorical columns (role, time, colour)."""

    def __init__(self, config: Any, width: int, *, heads: int) -> None:
        from .future import _GeometryOnly

        super().__init__(_GeometryOnly(config), width, heads=heads)
        self.embedding = _PoseTokenEmbedding(self.entity_width)


class _PoseTokenEmbedding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.pose = nn.Sequential(nn.Linear(7, width), nn.GELU(), nn.Linear(width, width))
        self.category = nn.Linear(POSE_TOKEN_DIM - 7, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens: Tensor) -> Tensor:
        centre = tokens.new_tensor(POS_CENTRE)
        pose = torch.cat([(tokens[..., :3] - centre) / POS_SCALE, tokens[..., 3:7]], -1)
        return self.norm(self.pose(pose) + self.category(tokens[..., 7:]))
