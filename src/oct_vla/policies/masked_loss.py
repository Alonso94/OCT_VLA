"""Exclude padded action targets from the pi0.5 loss.

An action chunk reaching past the end of an episode has nowhere to read from,
so LeRobot clamps the query inside the episode and repeats the final action,
flagging those positions in `action_is_pad`. ACT masks them out of its loss;
pi0.5's `forward` ends at `losses.mean()` and does not, so it fits the repeats
as though they were demonstrated.

Measured on this project's corpus at chunk 50: **17.4 % of all action targets
are padding**, and 40.2 % on the shortest clips, because the atomic transfers
are only 61 frames. What the padding teaches depends on the encoding, and none
of it is what the demonstration shows:

* absolute joints -- hold the terminal configuration,
* joint deltas -- a zero increment, i.e. stop,
* end-effector deltas -- repeat the final *nonzero* increment, i.e. keep going.

The last is actively wrong, and the first two bias a policy toward stillness in
exactly the frames where placement happens.

Kept as a mixin rather than a patch to LeRobot so the correction travels with
this project's policies and survives a LeRobot upgrade.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor


def masked_loss(losses: Tensor, batch: dict[str, Any], reduction: str = "mean"):
    """Reduce a (B, T, D) per-element loss, ignoring padded timesteps.

    Falls back to an unmasked reduction when the batch carries no
    `action_is_pad`, which is what a dataloader built without
    `delta_timestamps` produces. Silently returning a *masked* zero there would
    be worse than the padding itself.

    The normaliser counts only valid elements, so the loss keeps its scale
    whatever fraction of a batch is padded. Dividing by the full element count
    instead would make batches near episode ends look artificially good and
    quietly reweight them.
    """
    pad = batch.get("action_is_pad")
    if pad is None:
        if reduction == "none":
            return losses.mean(dim=(1, 2))
        return losses.mean()

    valid = (~pad).unsqueeze(-1).to(losses.dtype)
    if reduction == "none":
        # Per sample, so each episode is normalised by its own valid count; a
        # sample that is mostly padding must not be scored as mostly correct.
        per_sample = (losses * valid).sum(dim=(1, 2))
        counts = valid.expand_as(losses).sum(dim=(1, 2)).clamp_min(1.0)
        return per_sample / counts
    return (losses * valid).sum() / valid.expand_as(losses).sum().clamp_min(1.0)


def padded_fraction(batch: dict[str, Any]) -> float:
    """Share of this batch's action targets that are padding, for logging."""
    pad = batch.get("action_is_pad")
    return 0.0 if pad is None else float(pad.to(dtype=float).mean())
