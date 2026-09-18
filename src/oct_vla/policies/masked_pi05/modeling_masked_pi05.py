"""See configuration_masked_pi05: stock pi0.5 with the padding mask applied."""

from __future__ import annotations

from typing import Any

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION
from torch import Tensor

from oct_vla.policies.masked_loss import masked_loss, padded_fraction

from .configuration_masked_pi05 import MaskedPI05Config


class MaskedPI05Policy(PI05Policy):
    config_class = MaskedPI05Config
    name = "masked_pi05"

    def forward(
        self, batch: dict[str, Tensor], reduction: str = "mean"
    ) -> tuple[Tensor, dict[str, Any]]:
        """PI05Policy.forward, reducing over unpadded timesteps only.

        The per-element losses are recomputed here rather than intercepted,
        because the base implementation reduces before returning and there is
        no hook between the two.
        """
        images, img_masks = self.prepare_images(batch)
        tokens, masks = self.prepare_language(batch)
        states, state_masks = self.prepare_state(batch)
        actions = self.prepare_action(batch)

        losses = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            self.model.sample_noise(actions.shape, actions.device),
            self.model.sample_time(actions.shape[0], actions.device),
            states=states,
            state_masks=state_masks,
        )[:, :, : self.config.output_features[ACTION].shape[0]]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
            "padded_fraction": padded_fraction(batch),
        }
        result = masked_loss(losses, batch, reduction)
        loss_dict["loss"] = result.mean().item() if reduction == "none" else result.item()
        return result, loss_dict
