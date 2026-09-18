"""Object-conditioned X-VLA.

X-VLA builds its action embedding inside `SoftPromptedTransformer.forward`
(`soft_transformer.py:382`), with no overridable seam around it:

    x = self.action_encoder(action_tokens, domain_id)   # [B, num_actions, hidden]

so the residual is attached to `action_encoder` with a forward hook rather than
by overriding a forty-line `forward` that would then drift on every LeRobot
upgrade. See `ObjectConditioning.attach_to`.

The conditioning module is mounted on the transformer, which is where the width
lives, so `object_module_path` is `model.transformer` rather than π0.5's `model`.
"""

from __future__ import annotations

from typing import Any

import torch
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from torch import Tensor

from oct_vla.policies.object_conditioning import (
    ObjectConditionedPolicyMixin,
    ObjectConditioning,
)

from .configuration_control_xvla import ControlXVLAConfig


class ControlXVLAPolicy(ObjectConditionedPolicyMixin, XVLAPolicy):
    """LeRobot policy wrapper; action handling remains X-VLA's standard path."""

    config_class = ControlXVLAConfig
    name = "control_xvla"

    #: X-VLA embeds actions in `model.transformer`, not `model`.
    object_module_path = "model.transformer"

    def __init__(self, config: ControlXVLAConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        transformer = self.model.transformer
        transformer.object_conditioning = ObjectConditioning(
            config, transformer.hidden_size
        )
        # Installed once, at construction, so every forward is conditioned
        # without the host knowing. Held so the handle is not garbage collected.
        self._object_hook = transformer.object_conditioning.attach_to(
            transformer.action_encoder
        )
        self.model.to(config.device)

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
