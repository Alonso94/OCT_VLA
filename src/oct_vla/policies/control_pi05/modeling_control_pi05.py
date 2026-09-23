"""Where each arm attaches in pi0.5.

* ``kv``: the KV term in every action-expert attention layer, using Gemma's own
  query (``conditioning.hosts.install_pi_kv``).
* ``kv_adaln``: the pooled scene added to ``adarms_cond`` -- the expert-width
  time vector every expert layer's adaptive RMSNorm is already modulated by.
* ``kv_tokens``: the entities prepended to the action-expert suffix.
"""

from __future__ import annotations

from typing import Any

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import require_package
from torch import Tensor

from oct_vla.policies.conditioning.adaln import SceneVector
from oct_vla.policies.conditioning.hosts import install_pi_kv
from oct_vla.policies.conditioning.tokens import EntityTokens, prepend_entities, scene_inputs
from oct_vla.policies.masked_loss import masked_loss, padded_fraction
from oct_vla.policies.object_conditioning import (
    ObjectConditionedPolicyMixin,
    ObjectConditioning,
    unwrap_object_conditioning,
)

from .configuration_control_pi05 import ControlPI05Config


class ControlPI05Pytorch(PI05Pytorch):
    """pi0.5 with the conditioning module mounted on its action expert."""

    def __init__(self, config: ControlPI05Config, rtc_processor=None) -> None:
        super().__init__(config, rtc_processor=rtc_processor)
        # The action expert's width; the config states it only as a variant name.
        width = self.action_in_proj.out_features
        self.object_conditioning = ObjectConditioning(config, width)
        install_pi_kv(self)
        if config.object_conditioning == "kv_adaln":
            self.object_conditioning.adaln = SceneVector(config, width, width)
        if config.object_conditioning == "kv_tokens":
            self.object_conditioning.incontext = EntityTokens(config, width)

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor):
        action_emb, pad_masks, att_masks, adarms_cond = super().embed_suffix(
            noisy_actions, timestep
        )
        control = unwrap_object_conditioning(self.object_conditioning)
        inputs = scene_inputs(control)
        adaln = getattr(control, "adaln", None)
        if adaln is not None and inputs is not None:
            delta = adaln(*inputs, adarms_cond.shape[0])
            if delta is not None:
                adarms_cond = adarms_cond + delta.to(adarms_cond.dtype)
        action_emb, pad_masks, att_masks = prepend_entities(
            control, action_emb, pad_masks, att_masks
        )
        return action_emb, pad_masks, att_masks, adarms_cond


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


    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self._set_object_inputs(batch)
        try:
            return super().predict_action_chunk(batch, **kwargs)
        finally:
            self._clear_object_inputs()

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
            loss_dict = {
                "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
                # Logged so a change in clip length shows up as a number rather
                # than as an unexplained shift in the loss.
                "padded_fraction": padded_fraction(batch),
            }
            result = masked_loss(losses, batch, reduction)
            loss_dict["loss"] = (
                result.mean().item() if reduction == "none" else result.item()
            )
            return result, loss_dict
        finally:
            self._clear_object_inputs()
