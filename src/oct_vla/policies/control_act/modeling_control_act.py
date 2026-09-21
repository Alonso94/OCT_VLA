"""Preserve ACT attention and inject object KV at every decoder layer."""

import torch
from lerobot.policies.act.modeling_act import ACTPolicy

from oct_vla.policies.layerwise_attention import packed_mha_query
from oct_vla.policies.object_conditioning import ObjectConditionedPolicyMixin, ObjectConditioning

from .configuration_control_act import ControlACTConfig


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
