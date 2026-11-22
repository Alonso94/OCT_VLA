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

    def per_item(self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """The loss of each item [B], averaged over its own valid (arm, horizon) pairs."""
        v = valid[..., None].expand(pred.shape[:-1]).float()             # [B, H, 2]
        n = v.sum((1, 2)).clamp(min=1)
        trans = F.huber_loss(pred[..., :3] / self.length, target[..., :3] / self.length,
                             reduction="none", delta=1.0).sum(-1)
        rot = 1 - (pred[..., 3:] * target[..., 3:]).sum(-1) ** 2
        return ((trans + self.lambda_rot * rot) * v).sum((1, 2)) / n

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


class HypothesisLoss(nn.Module):
    """Several predicted futures: the best one per item takes the window loss (winner takes
    all), all of them a small share (``spread``, so none is left untrained), and the
    confidence logits learn which one wins (cross-entropy). The statistics are those of the
    most confident future, as at evaluation, plus the best one's error (``oracle_mm``).

    Args:
        window_loss: the per-future loss.
        spread: weight of the mean over all futures.
        confidence_weight: weight of the cross-entropy on the winner.
    """

    def __init__(self, window_loss: WindowLoss, spread: float = 0.05,
                 confidence_weight: float = 0.1) -> None:
        super().__init__()
        self.window_loss = window_loss
        self.spread = spread
        self.confidence_weight = confidence_weight

    def forward(self, windows: torch.Tensor, confidence: torch.Tensor, target: torch.Tensor,
                valid: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """windows [B, K, H, 2, 7], confidence [B, K]."""
        K = windows.shape[1]
        per = torch.stack([self.window_loss.per_item(windows[:, k], target, valid)
                           for k in range(K)], 1)                                  # [B, K]
        best = per.argmin(1)
        winner = per.gather(1, best[:, None]).squeeze(1)
        choose = F.cross_entropy(confidence, best)
        loss = (1 - self.spread) * winner.mean() + self.spread * per.mean() + self.confidence_weight * choose
        rows = torch.arange(windows.shape[0], device=windows.device)
        _, stats = self.window_loss(windows[rows, confidence.argmax(1)], target, valid)
        with torch.no_grad():
            _, oracle = self.window_loss(windows[rows, best], target, valid)
            stats["oracle_mm"] = oracle["trans_mm"]
            stats["choice_ce"] = choose.detach()
            stats["loss"] = loss.detach()
        return loss, stats


def factor_loss(logits: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over the rule's factors and the supports; a label of -1 (unknown)
    is masked out (no data-dependent branch, so no wait for the GPU)."""
    total = logits[0].new_zeros(())
    count = logits[0].new_zeros(())
    for k, factor_logits in enumerate(logits):
        known = (labels[:, k] >= 0).float()
        ce = F.cross_entropy(factor_logits, labels[:, k].clamp(min=0), reduction="none")
        total = total + (ce * known).sum()
        count = count + known.sum()
    return total / count.clamp(min=1)
