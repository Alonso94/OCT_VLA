"""Where each arm attaches in SmolVLA.

* ``kv``: the KV term in every expert attention layer
  (``conditioning.hosts.install_smol_kv``).
* ``kv_adaln``: SmolVLA's expert has no adaptive norm to condition, unlike
  pi0.5's and GR00T's, so the pooled scene drives a zero-initialised FiLM,
  ``out * (1 + g) + b``, on every expert RMSNorm.
* ``kv_tokens``: the entities prepended to the expert suffix.

``SmolVLAPolicy.forward`` already builds the loss, so the policy wrapper only
makes the entities available around it.
"""

from __future__ import annotations

from typing import Any

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from torch import Tensor

from oct_vla.policies.conditioning.adaln import SceneVector
from oct_vla.policies.conditioning.hosts import install_smol_kv
from oct_vla.policies.conditioning.tokens import EntityTokens, prepend_entities, scene_inputs
from oct_vla.policies.object_conditioning import (
    ObjectConditionedPolicyMixin,
    ObjectConditioning,
    unwrap_object_conditioning,
)

from .configuration_control_smolvla import ControlSmolVLAConfig


class ControlVLAFlowMatching(VLAFlowMatching):
    def __init__(self, config: ControlSmolVLAConfig, rtc_processor=None) -> None:
        super().__init__(config, rtc_processor=rtc_processor)
        # Every suffix projection in SmolVLA is expert_hidden_size wide, and
        # action_in_proj is the one PEFT already targets, so it is the stable
        # place to read that width from.
        self.object_conditioning = ObjectConditioning(
            config, self.action_in_proj.out_features
        )
        install_smol_kv(self)
        width = self.action_in_proj.out_features
        self._adaln_cache = None
        if config.object_conditioning == "kv_adaln":
            self._install_adaln(config, width)
        if config.object_conditioning == "kv_tokens":
            self.object_conditioning.incontext = EntityTokens(config, width)

    def _install_adaln(self, config, width: int) -> None:
        """FiLM every expert RMSNorm from the pooled scene: out * (1 + g) + b.

        SmolVLA's expert has no adaptive norm to condition, unlike pi0.5's and
        GR00T's, so the modulation sites are made here -- the pre-norm
        (DiT) form of control_act's hooks. The expert's layers are its own
        modules and only ever see the suffix, so no prefix token is touched.
        """
        norms = []
        for layer in self.vlm_with_expert.lm_expert.layers:
            norms += [layer.input_layernorm, layer.post_attention_layernorm]
        widths = {norm.weight.shape[0] for norm in norms}
        if widths != {width}:
            raise ValueError(f"expert norms are {widths} wide, expected {width}")
        self._adaln_sites = len(norms)
        self.object_conditioning.adaln = SceneVector(config, width, len(norms) * 2 * width)
        self._adaln_hooks = [
            norm.register_forward_hook(self._make_norm_hook(index))
            for index, norm in enumerate(norms)
        ]

    def _modulation(self, batch: int):
        """``[B, sites, 2, D]`` for the current inputs, computed once per inputs."""
        control = unwrap_object_conditioning(self.object_conditioning)
        inputs = scene_inputs(control)
        if inputs is None:
            return None
        if self._adaln_cache is None or self._adaln_cache[0] is not control._inputs:
            delta = control.adaln(*inputs, batch)
            if delta is not None:
                delta = delta.view(batch, self._adaln_sites, 2, -1)
            self._adaln_cache = (control._inputs, delta)
        return self._adaln_cache[1]

    def _make_norm_hook(self, index: int):
        def hook(module, args, output):
            modulation = self._modulation(output.shape[0])
            if modulation is None:
                return output
            gamma = modulation[:, index, 0].to(output.dtype)[:, None]
            beta = modulation[:, index, 1].to(output.dtype)[:, None]
            return output * (1 + gamma) + beta

        return hook

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor):
        # Three-tuple, unlike π0.5's four. `embs` here is the concatenated
        # action-time embedding, which is exactly what the residual conditions.
        embs, pad_masks, att_masks = super().embed_suffix(noisy_actions, timestep)
        control = unwrap_object_conditioning(self.object_conditioning)
        return prepend_entities(control, embs, pad_masks, att_masks)


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
