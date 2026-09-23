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

from oct_vla.policies.masked_loss import masked_loss, padded_fraction
from oct_vla.policies.object_conditioning import (
    ObjectConditionedPolicyMixin,
    ObjectConditioning,
    unwrap_object_conditioning,
)
from oct_vla.policies.vla_branches import prepend_entities, scene_inputs

from .configuration_control_pi05 import ControlPI05Config


class ControlPI05Pytorch(PI05Pytorch):
    """π0.5 with the conditioning module mounted on its action embedding.

    The width comes from the projection rather than the config: it is the
    action expert's width, which the config expresses only as a variant name.
    """

    def __init__(self, config: ControlPI05Config, rtc_processor=None) -> None:
        super().__init__(config, rtc_processor=rtc_processor)
        self.object_conditioning = ObjectConditioning(
            config, self.action_in_proj.out_features
        )
        if config.object_injection_mode == "layerwise":
            from oct_vla.policies.layerwise_backbones import install_pi_layerwise

            install_pi_layerwise(self)
        width = self.action_in_proj.out_features
        if getattr(config, "object_adaln", False):
            from oct_vla.policies.layerwise_attention import SceneVector

            # adarms_cond is the expert-width time vector every expert layer's
            # adaptive RMSNorm is modulated by; the scene adds to it.
            self.object_conditioning.adaln = SceneVector(config, width, width)
        if getattr(config, "object_incontext", False):
            from oct_vla.policies.layerwise_attention import InContextEntities

            self.object_conditioning.incontext = InContextEntities(config, width)

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor):
        # π0.5 returns four values here; SmolVLA's analogue returns three. That
        # arity is now the only difference between the two plugins.
        action_emb, pad_masks, att_masks, adarms_cond = super().embed_suffix(
            noisy_actions, timestep
        )
        control = unwrap_object_conditioning(self.object_conditioning)
        action_emb = control.residual(action_emb)
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
