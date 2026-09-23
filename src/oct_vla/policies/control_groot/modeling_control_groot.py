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
    unwrap_object_conditioning,
)
from oct_vla.policies.vla_branches import scene_inputs
from oct_vla.policies.slim_groot.modeling_slim_groot import GROOT_BACKBONE
from oct_vla.policies.stage_loading import SlimCheckpointMixin

from .configuration_control_groot import ControlGrootConfig


class ControlGrootPolicy(SlimCheckpointMixin, ObjectConditionedPolicyMixin, GrootPolicy):
    """LeRobot policy wrapper; action handling remains GR00T's standard path.

    Slim checkpoints, as `slim_groot`: the frozen backbone is rebuilt from the
    base rather than saved, and a stage-1 slim_groot checkpoint loads with only
    the object subtree fresh.
    """

    config_class = ControlGrootConfig
    name = "control_groot"
    rebuilt_prefixes = GROOT_BACKBONE

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
        self._branch_hooks = []
        self._incontext_count = 0
        if getattr(config, "object_adaln", False):
            self._install_adaln(config, head)
        if getattr(config, "object_incontext", False):
            self._install_incontext(config, head)

    def _control(self):
        return unwrap_object_conditioning(self._groot_model.action_head.object_conditioning)

    def _install_adaln(self, config, head) -> None:
        """Add the pooled scene to the DiT's timestep embedding.

        `temb` is what every block's AdaLayerNorm and the output norm are
        modulated by (cross_attention_dit.py), so this conditions the whole
        action model through its pretrained modulation layers, as DiT does a
        class label. Zero-initialised, so step 0 is stage 1 exactly.
        """
        from oct_vla.policies.layerwise_attention import SceneVector

        encoder = head.model.timestep_encoder
        dim = encoder.timestep_embedder.linear_2.out_features
        head.object_conditioning.adaln = SceneVector(config, head.input_embedding_dim, dim)

        def hook(module, args, output):
            control = self._control()
            inputs = scene_inputs(control)
            if inputs is None:
                return output
            delta = control.adaln(*inputs, output.shape[0])
            return output if delta is None else output + delta.to(output.dtype)

        self._branch_hooks.append(encoder.register_forward_hook(hook))

    def _install_incontext(self, config, head) -> None:
        """Prepend entity tokens to the DiT sequence; strip them from its output.

        The head reads actions back as the *last* horizon positions
        (`pred[:, -T:]`), so tokens in front are dropped by construction, and
        they are stripped here too so the decoder never sees them. GR00T's DiT
        self-attention takes no mask, so padded slots cannot be hidden: only
        the real entities are inserted, which requires every row of a batch to
        hold the same number -- true of each object-count profile, and checked.
        """
        from oct_vla.policies.layerwise_attention import InContextEntities

        head.object_conditioning.incontext = InContextEntities(config, head.input_embedding_dim)
        model = head.model

        def before(module, args, kwargs):
            self._incontext_count = 0
            control = self._control()
            inputs = scene_inputs(control)
            hidden = kwargs.get("hidden_states")
            if inputs is None or hidden is None:
                return None
            entities, padding = control.incontext.embed(*inputs, hidden.shape[0], hidden.dtype)
            real = (~padding).sum(dim=1)
            if not bool((real == real[0]).all()):
                raise ValueError(
                    "GR00T in-context entities need the same number of real entities in "
                    f"every row of a batch; got {real.tolist()}"
                )
            count = int(real[0])
            if count == 0:
                return None
            entities = entities[~padding].view(hidden.shape[0], count, -1)
            self._incontext_count = count
            return args, {**kwargs, "hidden_states": torch.cat([entities, hidden], dim=1)}

        def after(module, args, kwargs, output):
            count, self._incontext_count = self._incontext_count, 0
            if not count:
                return output
            if isinstance(output, tuple):
                return (output[0][:, count:], [h[:, count:] for h in output[1]], *output[2:])
            return output[:, count:]

        self._branch_hooks.append(model.register_forward_pre_hook(before, with_kwargs=True))
        self._branch_hooks.append(model.register_forward_hook(after, with_kwargs=True))

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
