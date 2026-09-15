"""Object-conditioned π0.5 with a zero-initialized residual injection.

This plugin accepts already-extracted canonical scene tokens.  Perception and
RoboTwin remain outside the policy boundary.

Everything backbone-independent lives in ``oct_vla.policies.object_conditioning``;
what is left here is the π0.5-specific hook.
"""

from __future__ import annotations

from typing import Any

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import require_package
from torch import Tensor

from oct_vla.policies.object_conditioning import (
    ObjectConditionedPolicyMixin,
    ObjectInjectionMixin,
)

from .configuration_control_pi05 import ControlPI05Config


class ControlPI05Pytorch(ObjectInjectionMixin, PI05Pytorch):
    def __init__(self, config: ControlPI05Config, rtc_processor=None) -> None:
        super().__init__(config, rtc_processor=rtc_processor)
        self.init_object_conditioning(config, self.action_in_proj.out_features)

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor):
        # π0.5 returns four values; SmolVLA's analogue returns three. That
        # difference is the only reason these two plugins are separate files.
        action_emb, pad_masks, att_masks, adarms_cond = super().embed_suffix(
            noisy_actions, timestep
        )
        return self.object_residual(action_emb), pad_masks, att_masks, adarms_cond


class ControlPI05Policy(ObjectConditionedPolicyMixin, PI05Policy):
    """LeRobot policy wrapper; action handling remains PI0.5's standard path."""

    config_class = ControlPI05Config
    name = "control_pi05"

    def __init__(self, config: ControlPI05Config, **kwargs: Any) -> None:
        del kwargs
        require_package("transformers", extra="pi")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = ControlPI05Pytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    def _prepare_pretrained_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return self._add_object_state_defaults(super()._prepare_pretrained_state_dict(state_dict))

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self._set_object_inputs(batch)
        try:
            return super().predict_action_chunk(batch, **kwargs)
        finally:
            self.model.clear_object_inputs()

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        self._set_object_inputs(batch)
        try:
            images, img_masks = self._preprocess_images(batch)
            states, state_masks = self._prepare_memory_states(batch)
            actions = self.prepare_action(batch)
            losses = self.model.forward(
                images,
                img_masks,
                batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK],
                actions,
                self.model.sample_noise(actions.shape, actions.device),
                self.model.sample_time(actions.shape[0], actions.device),
                states=states,
                state_masks=state_masks,
            )[:, :, : self.config.output_features[ACTION].shape[0]]
            loss_dict = {"loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist()}
            if reduction == "none":
                result = losses.mean(dim=(1, 2))
                loss_dict["loss"] = result.mean().item()
                return result, loss_dict
            result = losses.mean()
            loss_dict["loss"] = result.item()
            return result, loss_dict
        finally:
            self.model.clear_object_inputs()

    def _get_default_peft_targets(self) -> dict[str, Any]:
        return self._add_object_peft_targets(super()._get_default_peft_targets())
