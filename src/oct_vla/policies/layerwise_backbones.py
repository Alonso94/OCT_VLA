"""Native-query attention adapters; object branches stay in one checkpoint subtree.

π0.5 calls Gemma attention directly during joint training and through Gemma
layers during cached inference. A context-local dispatcher covers both routes;
other policies and prefix-only calls retain the unmodified function.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from types import MethodType

import torch

_ACTIVE_PI = ContextVar("octvla_pi_attention", default=None)


def _active_control(host, fallback):
    """Resolve PEFT's active modules-to-save copy for every forward."""
    from oct_vla.policies.object_conditioning import unwrap_object_conditioning

    return unwrap_object_conditioning(getattr(host, "object_conditioning", fallback)) or fallback


def _context(branch, query, inputs, scaling=None):
    width = query.shape[1] * query.shape[-1]
    identity = torch.eye(width, device=query.device, dtype=query.dtype)
    return branch(query, *inputs, output_weight=identity, scaling=scaling)


def install_pi_layerwise(model):
    from transformers.models.gemma import modeling_gemma

    host = model.paligemma_with_expert
    control = model.object_conditioning
    mapping = {}
    for index, (prefix, expert) in enumerate(
        zip(host.paligemma.model.language_model.layers, host.gemma_expert.model.layers, strict=True)
    ):
        attn = expert.self_attn
        width = attn.q_proj.out_features
        branch = control.add_layer(str(index), width, width // attn.head_dim)
        mapping[id(prefix.self_attn)] = str(index)
        mapping[id(attn)] = str(index)
    # Inference must use the same eager entrypoint as joint training.
    host.gemma_expert.config._attn_implementation = "eager"
    host.gemma_expert.model.config._attn_implementation = "eager"
    if not getattr(modeling_gemma.eager_attention_forward, "_octvla_dispatch", False):
        native_attention = modeling_gemma.eager_attention_forward

        @wraps(native_attention)
        def attention(module, query, key, value, attention_mask, scaling, *args, **kwargs):
            output, weights = native_attention(
                module, query, key, value, attention_mask, scaling, *args, **kwargs
            )
            active = _ACTIVE_PI.get()
            if active is not None:
                layer_names, steps = active
                current = _active_control(module._octvla_control_host, None)
                layer_name = layer_names.get(id(module))
                branch = current.layers[layer_name] if layer_name in current.layers else None
                if branch is not None and current._inputs is not None and steps:
                    residual = _context(branch, query[:, :, -steps:], current._inputs, scaling)
                    extra = torch.zeros_like(output)
                    extra[:, -steps:] = residual.reshape_as(output[:, -steps:])
                    output = output + extra
            return output, weights

        attention._octvla_dispatch = True
        modeling_gemma.eager_attention_forward = attention
    native_forward = host.forward
    # The global eager dispatcher receives only a Gemma attention module.
    # Attach the owning conditioned model so it can resolve PEFT's live copy.
    for layer in (*host.paligemma.model.language_model.layers, *host.gemma_expert.model.layers):
        layer.self_attn._octvla_control_host = model

    def forward(_self, *args, **kwargs):
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            raise ValueError("layerwise pi0.5 expects named inputs_embeds")
        steps = embeddings[1].shape[1] if embeddings[1] is not None else 0
        token = _ACTIVE_PI.set((mapping, steps))
        try:
            return native_forward(*args, **kwargs)
        finally:
            _ACTIVE_PI.reset(token)

    host.forward = MethodType(forward, host)


def install_smol_layerwise(model):
    host = model.vlm_with_expert
    control = model.object_conditioning
    branches = {}
    for index, layer in enumerate(host.lm_expert.layers):
        attention = layer.self_attn
        width = attention.q_proj.out_features
        control.add_layer(str(index), width, width // attention.head_dim)
        branches[id(layer)] = str(index)
    context = ContextVar(f"octvla_smol_{id(host)}", default=None)
    native_interface = host.get_attention_interface()

    def interface(mask, batch, head_dim, query, key, value):
        output = native_interface(mask, batch, head_dim, query, key, value)
        current = context.get()
        if current is not None:
            layer_name, steps, calls_to_skip = current
            if calls_to_skip[0]:
                calls_to_skip[0] -= 1
            else:
                current_control = _active_control(model, control)
                branch = (
                    current_control.layers[layer_name]
                    if layer_name in current_control.layers
                    else None
                )
                if branch is None or current_control._inputs is None:
                    return output
                q = query[:, -steps:].transpose(1, 2)
                residual = _context(branch, q, current_control._inputs)
                # The branch has already applied the host output weight and
                # returns the native flattened [B, S, width] layout. Keep the
                # prefix exactly unchanged.
                extra = torch.zeros_like(output)
                extra[:, -steps:] = residual.to(output.dtype)
                output = output + extra
        return output

    host.get_attention_interface = MethodType(lambda _self: interface, host)
    for method_name in ("forward_attn_layer", "forward_cross_attn_layer"):
        native = getattr(host, method_name)

        def wrap(native, cross):
            def forward(_self, model_layers, inputs_embeds, layer_idx, *args, **kwargs):
                expert = model_layers[1][layer_idx]
                suffix = inputs_embeds[1]
                branch = branches.get(id(expert)) if suffix is not None else None
                steps = suffix.shape[1] if suffix is not None else 0
                skip = int(cross and inputs_embeds[0] is not None)
                token = context.set((branch, steps, [skip]))
                try:
                    return native(model_layers, inputs_embeds, layer_idx, *args, **kwargs)
                finally:
                    context.reset(token)

            return forward

        setattr(host, method_name, MethodType(wrap(native, "cross" in method_name), host))


def install_diffusers_layerwise(head, control, *, conditioning_owner=None):
    """Inject before native output projection using the actual projected query.

    ``head`` is the DiT module holding attention layers; its parent owns the
    conditioning attribute, so the owner is explicit for PEFT live resolution.
    """
    owner = conditioning_owner if conditioning_owner is not None else head
    handles = []
    candidates = [
        (name, module)
        for name, module in head.named_modules()
        if hasattr(module, "to_q") and hasattr(module, "to_out") and hasattr(module, "heads")
    ]
    if not candidates:
        raise ValueError("no supported action-attention layers found")
    for index, (_name, host) in enumerate(candidates):
        width = host.to_q.out_features
        layer_name = str(index)
        control.add_layer(layer_name, width, host.heads)
        captured = {}

        def query_hook(_module, _args, output, captured=captured):
            captured["query"] = output

        def before_output(_module, args, captured=captured, layer_name=layer_name, host=host):
            query = captured.pop("query", None)
            current = _active_control(owner, control)
            if current._inputs is None:
                return args
            branch = current.layers[layer_name]
            if query is None:
                raise RuntimeError("native action attention did not project its query")
            q = query.reshape(query.shape[0], query.shape[1], host.heads, -1).transpose(1, 2)
            if getattr(host, "norm_q", None) is not None:
                q = host.norm_q(q)
            residual = _context(branch, q, current._inputs, getattr(host, "scale", None))
            # The branch has already applied an identity output projection
            # and returns [B, S, H*D], exactly the input expected by to_out.
            return (args[0] + residual.to(args[0].dtype), *args[1:])

        handles.append(host.to_q.register_forward_hook(query_hook))
        handles.append(host.to_out[0].register_forward_pre_hook(before_output))
    return handles
