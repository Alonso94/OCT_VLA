"""Object-conditioned GR00T N1.7.

GR00T's action embedding comes from `GR00TN17ActionHead.action_encoder`, a
`MultiEmbodimentActionEncoder` returning `[B, horizon, input_embedding_dim]` --
the same shape X-VLA and VLA-JEPA produce, so the residual attaches with the
same forward hook and no new mechanism. See `ObjectConditioning.attach_to`.

`object_module_path` is `_groot_model.action_head`: the policy holds its model
under a private attribute, and the width lives on the head rather than on the
config, which states it only through the checkpoint's architecture.
"""

from __future__ import annotations

from typing import Any

import torch
from lerobot.policies.groot.modeling_groot import GrootPolicy
from torch import Tensor

from oct_vla.policies.object_conditioning import (
    ObjectConditionedPolicyMixin,
    ObjectConditioning,
)

from .configuration_control_groot import ControlGrootConfig


class ControlGrootPolicy(ObjectConditionedPolicyMixin, GrootPolicy):
    """LeRobot policy wrapper; action handling remains GR00T's standard path."""

    config_class = ControlGrootConfig
    name = "control_groot"

    #: GR00T keeps its model on a private attribute, and embeds actions in the
    #: diffusion head rather than at the top level.
    object_module_path = "_groot_model.action_head"

    def __init__(self, config: ControlGrootConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        head = self._groot_model.action_head
        head.object_conditioning = ObjectConditioning(config, head.input_embedding_dim)
        if config.object_injection_mode == "layerwise":
            from oct_vla.policies.layerwise_backbones import install_diffusers_layerwise

            self._object_hooks = install_diffusers_layerwise(
                head.model, head.object_conditioning, conditioning_owner=head
            )
        else:
            self._object_hook = head.object_conditioning.attach_to(head.action_encoder)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self._set_object_inputs(batch)
        try:
            return super().predict_action_chunk(batch, **kwargs)
        finally:
            self._clear_object_inputs()

    def forward(self, batch: dict[str, Tensor], **kwargs: Any):
        self._set_object_inputs(batch)
        try:
            return super().forward(batch, **kwargs)
        finally:
            self._clear_object_inputs()

    def _get_default_peft_targets(self) -> dict[str, Any]:
        targets = super()._get_default_peft_targets()
        # These are fresh embodiment-specific projections, not pretrained
        # weights to approximate with LoRA. Keep them fully trainable and in
        # the adapter checkpoint alongside the layerwise object modules.
        saves = list(targets.get("modules_to_save", []))
        saves.extend(
            f"_groot_model.action_head.{name}"
            for name in ("action_encoder", "state_encoder", "action_decoder")
        )
        targets["modules_to_save"] = list(dict.fromkeys(saves))
        return targets
