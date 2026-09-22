"""Preserve ACT attention and inject object KV at every decoder layer.

Optionally also reach the encoder, which the layerwise branch never does:
LeRobot ACT has one decoder layer, so that branch alone is a single seam, and
the four encoder layers that fuse images with proprioception see no objects.
``object_adaln`` modulates every encoder and decoder block from a pooled scene
vector; ``object_incontext`` appends the entity tokens to the encoder sequence.
Both mount by hooks, so every stage-1 ACT state-dict key survives.
"""

import torch
from lerobot.policies.act.modeling_act import ACTPolicy

from oct_vla.policies.layerwise_attention import (
    InContextEntities,
    SceneAdaLN,
    packed_mha_query,
)
from oct_vla.policies.object_conditioning import ObjectConditionedPolicyMixin, ObjectConditioning

from .configuration_control_act import ControlACTConfig


def adaln_sites(model):
    """(sublayer, following norm) pairs of the main encoder and decoder.

    Never the VAE encoder, which is also an ``ACTEncoder``. With post-norm the
    norm follows its sublayer; with pre-norm it precedes it, which is DiT's form.
    """
    sites = []
    for layer in model.encoder.layers:
        sites += [(layer.self_attn, layer.norm1), (layer.linear2, layer.norm2)]
    for layer in model.decoder.layers:
        sites += [
            (layer.self_attn, layer.norm1),
            (layer.multihead_attn, layer.norm2),
            (layer.linear2, layer.norm3),
        ]
    return sites


class ControlACTPolicy(ObjectConditionedPolicyMixin, ACTPolicy):
    config_class = ControlACTConfig
    name = "control_act"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        conditioning = ObjectConditioning(config, config.dim_model)
        self.model.object_conditioning = conditioning
        self._object_hooks = []
        for index, layer in enumerate(self.model.decoder.layers):
            host = layer.multihead_attn
            conditioning.add_layer(str(index), host.embed_dim, host.num_heads)
            self._object_hooks.append(
                host.register_forward_hook(self._make_hook(str(index)), with_kwargs=True)
            )
        self._adaln_cache = None
        if getattr(config, "object_adaln", False):
            sites = adaln_sites(self.model)
            conditioning.adaln = SceneAdaLN(config, config.dim_model, len(sites))
            for index, (sublayer, norm) in enumerate(sites):
                self._object_hooks.append(
                    sublayer.register_forward_hook(self._make_gate_hook(index))
                )
                self._object_hooks.append(norm.register_forward_hook(self._make_norm_hook(index)))
        self._incontext_count = None
        if getattr(config, "object_incontext", False):
            conditioning.incontext = InContextEntities(config, config.dim_model)
            encoder = self.model.encoder
            self._object_hooks.append(
                encoder.register_forward_pre_hook(self._incontext_pre_hook, with_kwargs=True)
            )
            self._object_hooks.append(encoder.register_forward_hook(self._incontext_post_hook))

    def _make_hook(self, layer_name):
        def hook(host, args, kwargs, output):
            conditioning = self.object_conditioning
            inputs = conditioning._inputs
            if inputs is None:
                return output
            query = kwargs.get("query", args[0] if args else None)
            residual = conditioning.layers[layer_name](
                packed_mha_query(host, query),
                *inputs,
                output_weight=host.out_proj.weight,
                # The host's own attention dropout, so the two terms of the sum
                # are regularised alike. ACT defaults to 0.1, and leaving the
                # object branch undropped would make it the less regularised of
                # the two -- a thumb on the scale in the object-versus-RGB
                # comparison this policy exists to run.
                dropout=host.dropout,
            )
            if not host.batch_first:
                residual = residual.transpose(0, 1)
            return (output[0] + residual, output[1])

        return hook

    def _modulation(self, batch):
        """``[B, sites, 3, D]`` for this forward pass, computed once and cached.

        Keyed on the inputs tuple, so a stale cache cannot survive a new batch
        even if a caller bypasses `_set_object_inputs`.
        """
        conditioning = self.object_conditioning
        inputs = conditioning._inputs
        if inputs is None:
            return None
        if self._adaln_cache is None or self._adaln_cache[0] is not inputs:
            self._adaln_cache = (inputs, conditioning.adaln(*inputs, batch))
        return self._adaln_cache[1]

    def _make_gate_hook(self, index):
        def hook(module, args, output):
            attention = isinstance(output, tuple)
            value = output[0] if attention else output
            modulation = self._modulation(value.shape[1])
            if modulation is None:
                return output
            gated = value * (1 + modulation[:, index, 0].to(value.dtype))[None]
            return (gated, *output[1:]) if attention else gated

        return hook

    def _make_norm_hook(self, index):
        def hook(module, args, output):
            modulation = self._modulation(output.shape[1])
            if modulation is None:
                return output
            gamma = modulation[:, index, 1].to(output.dtype)[None]
            beta = modulation[:, index, 2].to(output.dtype)[None]
            return output * (1 + gamma) + beta

        return hook

    def _incontext_pre_hook(self, encoder, args, kwargs):
        inputs = self.object_conditioning._inputs
        self._incontext_count = None
        if inputs is None or inputs[0] is None or inputs[0].shape[-2] == 0:
            return None
        kwargs = dict(kwargs)
        x = kwargs.pop("x", args[0] if args else None)
        pos_embed = kwargs.pop("pos_embed", args[1] if len(args) > 1 else None)
        padding = kwargs.pop("key_padding_mask", args[2] if len(args) > 2 else None)
        extended, pos_embed, padding = self.object_conditioning.incontext.extend(
            x, pos_embed, padding, *inputs
        )
        self._incontext_count = extended.shape[0] - x.shape[0]
        return (extended,), {"pos_embed": pos_embed, "key_padding_mask": padding}

    def _incontext_post_hook(self, encoder, args, output):
        count, self._incontext_count = self._incontext_count, None
        if not count:
            return output
        # Discarded after the encoder, as LPWM drops its condition tokens: the
        # decoder's memory and positional embedding stay exactly the host's.
        return output[:-count]

    def _set_object_inputs(self, batch):
        self._adaln_cache = None
        super()._set_object_inputs(batch)

    def _clear_object_inputs(self):
        self._adaln_cache = None
        super()._clear_object_inputs()

    def forward(self, batch):
        self._set_object_inputs(batch)
        try:
            return super().forward(batch)
        finally:
            self._clear_object_inputs()

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        self._set_object_inputs(batch)
        try:
            return super().predict_action_chunk(batch)
        finally:
            self._clear_object_inputs()
