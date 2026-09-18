"""Object-conditioned VLA-JEPA.

The action embedding is built inside `VLAJEPAActionHead._build_inputs`
(`action_head.py:266`):

    action_features = self.action_encoder(actions, timesteps)

which is reached from both `forward` and `predict_action`, so conditioning the
encoder's output covers training and inference with one attachment. As with
X-VLA this is a forward hook rather than an override, to avoid duplicating
upstream code and to leave the pretrained checkpoint's keys untouched.

`object_module_path` is `model.action_model`: the head, not the policy, is where
the width lives.
"""

from __future__ import annotations

from typing import Any

import torch
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy
from torch import Tensor

from oct_vla.policies.object_conditioning import (
    ObjectConditionedPolicyMixin,
    ObjectConditioning,
)

from .configuration_control_vla_jepa import ControlVLAJEPAConfig


class ControlVLAJEPAPolicy(ObjectConditionedPolicyMixin, VLAJEPAPolicy):
    """LeRobot policy wrapper; action handling remains VLA-JEPA's standard path."""

    config_class = ControlVLAJEPAConfig
    name = "control_vla_jepa"

    #: VLA-JEPA embeds actions in its DiT head, not at the top level.
    object_module_path = "model.action_model"

    def __init__(self, config: ControlVLAJEPAConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        head = self.model.action_model
        # `input_embedding_dim` is the DiT's inner width (num_heads *
        # head_dim), which the config states only as a variant name.
        head.object_conditioning = ObjectConditioning(config, head.input_embedding_dim)
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
