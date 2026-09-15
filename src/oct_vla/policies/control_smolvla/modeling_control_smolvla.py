"""Object-conditioned SmolVLA with a zero-initialized residual injection.

The analogue of ``control_pi05``, and deliberately as close to it as the two
backbones allow -- the sweep compares backbones, so any gratuitous difference
here would be a confound.

Two things differ from π0.5, both in ``VLAFlowMatching``:

* ``embed_suffix`` returns three values, not four (π0.5 also carries
  ``adarms_cond``).
* ``SmolVLAPolicy.forward`` already builds the loss, so this wrapper only has
  to make the object tokens available around it, rather than reimplementing
  the forward pass as the π0.5 arm must.
"""

from __future__ import annotations

from typing import Any

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from torch import Tensor

from oct_vla.policies.object_conditioning import (
    ObjectConditionedPolicyMixin,
    ObjectInjectionMixin,
)

from .configuration_control_smolvla import ControlSmolVLAConfig


class ControlVLAFlowMatching(ObjectInjectionMixin, VLAFlowMatching):
    def __init__(self, config: ControlSmolVLAConfig, rtc_processor=None) -> None:
        super().__init__(config, rtc_processor=rtc_processor)
        # Every suffix projection in SmolVLA is expert_hidden_size wide, and
        # action_in_proj is the one PEFT already targets, so it is the stable
        # place to read that width from.
        self.init_object_conditioning(config, self.action_in_proj.out_features)

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor):
        # Three-tuple, unlike π0.5's four. `embs` here is the concatenated
        # action-time embedding, which is exactly what the residual conditions.
        embs, pad_masks, att_masks = super().embed_suffix(noisy_actions, timestep)
        return self.object_residual(embs), pad_masks, att_masks


class ControlSmolVLAPolicy(ObjectConditionedPolicyMixin, SmolVLAPolicy):
    """LeRobot policy wrapper; action handling remains SmolVLA's standard path."""

    config_class = ControlSmolVLAConfig
    name = "control_smolvla"

    def __init__(self, config: ControlSmolVLAConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        # Swap in the conditioned model. Rebuilding rather than patching the
        # base class keeps SmolVLAPolicy.__init__ as the single owner of every
        # other construction detail (RTC processor, dtype, device, queues),
        # which is what makes this arm comparable to the unconditioned one.
        self.model = ControlVLAFlowMatching(config, rtc_processor=self.rtc_processor)
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

    def forward(self, batch: dict[str, Tensor], **kwargs: Any):
        self._set_object_inputs(batch)
        try:
            return super().forward(batch, **kwargs)
        finally:
            self.model.clear_object_inputs()

    def _get_default_peft_targets(self) -> dict[str, Any]:
        return self._add_object_peft_targets(super()._get_default_peft_targets())
