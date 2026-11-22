"""The window loss of the paper (Eq. loss): Huber on translation plus a sign-invariant
rotation term, averaged over valid (arm, horizon) pairs."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

#: Rotation weight. 1 - <q_hat, q>^2 is ~0.005 at 8 degrees against a translation term
#: near 1, so with weight 1 rotation got almost no gradient (stuck at ~8 deg, recipe 1).
LAMBDA_ROT = 10.0


class WindowLoss(nn.Module):
    """Huber(translation / length) + lambda_rot * (1 - <q_hat, q>^2).

    Args:
        length: the workspace length that scales translation errors (m).
        lambda_rot: weight of the rotation term.
    """

    def __init__(self, length: float = 0.05, lambda_rot: float = LAMBDA_ROT) -> None:
        super().__init__()
        self.length = length
        self.lambda_rot = lambda_rot

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                valid: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """pred, target [B, H, 2, 7]; valid [B, H] (both arms share a horizon's validity).
        Returns the loss and 0-d tensor statistics for logging."""
        v = valid[..., None].expand(pred.shape[:-1]).float()             # [B, H, 2]
        n = v.sum().clamp(min=1)

        trans = F.huber_loss(pred[..., :3] / self.length, target[..., :3] / self.length,
                             reduction="none", delta=1.0).sum(-1)
        dot = (pred[..., 3:] * target[..., 3:]).sum(-1)
        rot = 1 - dot ** 2
        loss_trans = (trans * v).sum() / n
        loss_rot = (rot * v).sum() / n
        loss = loss_trans + self.lambda_rot * loss_rot

        with torch.no_grad():
            err_m = (pred[..., :3] - target[..., :3]).norm(dim=-1)
            angle = 2 * torch.acos(dot.abs().clamp(0, 1))
            # Kept as tensors: converting to floats here would make the CPU wait for the GPU
            # on every step. Callers convert when they log.
            stats = {"loss": loss.detach(),
                     "loss_trans": loss_trans.detach(),
                     "loss_rot": loss_rot.detach(),
                     "trans_mm": (err_m * v).sum() / n * 1000,
                     "rot_deg": (angle * v).sum() / n * 180 / math.pi}
        return loss, stats
