"""Where KV attaches in each host's attention, using the host's real query.

KV (``kv.py``) needs the host's own projected query and output projection, and
the hosts expose them differently:

* ``install_pi_kv``        pi0.5: Gemma attention, called directly in joint
                           training and through Gemma layers in cached
                           inference; one context-local dispatcher covers both.
* ``install_smol_kv``      SmolVLA: its expert's attention interface.
* ``install_diffusers_kv`` any diffusers ``Attention`` (``to_q``/``to_out``):
                           GR00T's and VLA-JEPA's DiT action heads.

Each adds the branch's residual to the action tokens only and leaves the prefix
exactly unchanged. These hooks encode bugs already found once (PEFT's active
copy, cached-inference skips, a module cycle): read the comments before
changing them.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from types import MethodType

import torch

from .tokens import entity_key_bias, realign_suffix_positions

_ACTIVE_PI = ContextVar("octvla_pi_attention", default=None)


def _active_control(host, fallback):
    """Resolve PEFT's active modules-to-save copy for every forward."""
    from oct_vla.policies.object_conditioning import unwrap_object_conditioning

    return unwrap_object_conditioning(getattr(host, "object_conditioning", fallback)) or fallback


def _context(branch, query, inputs, scaling=None):
    width = query.shape[1] * query.shape[-1]
    identity = torch.eye(width, device=query.device, dtype=query.dtype)
    return branch(query, *inputs, output_weight=identity, scaling=scaling)


def install_pi_kv(model):
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
    #
    # object.__setattr__, not plain assignment: nn.Module.__setattr__ files any
    # Module value into `_modules`, so assigning the root model onto one of its
    # own descendants makes the tree cyclic. `Module._apply` and `state_dict`
    # both recurse over children with no memo, so `.to(device)` and
    # `save_pretrained` then raise RecursionError -- while `named_parameters`
    # survives, because it deduplicates, which is how this hides in a partial
    # smoke test. Bypassing __setattr__ keeps the reference a plain attribute.
    for layer in (*host.paligemma.model.language_model.layers, *host.gemma_expert.model.layers):
        object.__setattr__(layer.self_attn, "_octvla_control_host", model)

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


def install_smol_kv(model):
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

    def entities():
        from oct_vla.policies.object_conditioning import unwrap_object_conditioning

        return getattr(unwrap_object_conditioning(model.object_conditioning), "_suffix_entities", None)

    def interface(mask, batch, head_dim, query, key, value):
        current = context.get()
        layout = entities() if current is not None else None
        # kv_tokens: a self-attention call over a suffix holding entities gets
        # ACT's gate on the entity keys. Cross-attention layers attend the
        # prefix only, where there are no entity keys.
        if layout is not None and not current[3] and current[1]:
            bias = entity_key_bias(layout, mask.shape[-2], mask.shape[-1], current[1], torch.float32)
            output = _biased_eager(host, mask, bias, batch, head_dim, query, key, value)
        else:
            output = native_interface(mask, batch, head_dim, query, key, value)
        if current is not None:
            layer_name, steps, calls_to_skip, _ = current
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
                # Mirror upstream's own predicate for whether a prefix
                # attention call happens first, rather than a predicate that
                # merely agrees with it on the usual path. SmolVLA gates that
                # call on `len(inputs_embeds) == 2 and not past_key_values`
                # (smolvlm_with_expert.py:309); testing only
                # `inputs_embeds[0] is not None` diverges whenever a prefix
                # arrives alongside a populated cache -- upstream then makes
                # one interface call, this wrapper consumes it as the skip, and
                # the object residual is silently dropped for every
                # cross-attention layer of that step.
                # Upstream passes `past_key_values` by keyword at both call
                # sites (smolvlm_with_expert.py:453, :467); index 5 is its
                # positional slot after `layer_idx`, kept as a fallback rather
                # than guessed at, so a positional caller cannot silently read
                # `position_ids` as a cache.
                cache = kwargs.get("past_key_values", args[5] if len(args) > 5 else None)
                prefix_call = cross and len(inputs_embeds) == 2 and not cache
                skip = int(bool(prefix_call))
                token = context.set((branch, steps, [skip], cross))
                try:
                    return native(model_layers, inputs_embeds, layer_idx, *args, **kwargs)
                finally:
                    context.reset(token)

            return forward

        setattr(host, method_name, MethodType(wrap(native, "cross" in method_name), host))

    # kv_tokens: every action keeps its stage-1 RoPE position (tokens.py).
    native_forward = host.forward

    def forward(*args, position_ids=None, inputs_embeds=None, **kwargs):
        layout = entities()
        suffix = inputs_embeds[1] if inputs_embeds is not None else None
        if layout is not None and suffix is not None:
            if position_ids is None:
                raise RuntimeError("entity positions need position_ids by keyword")
            position_ids = realign_suffix_positions(position_ids, suffix.shape[1], layout)
        return native_forward(*args, position_ids=position_ids, inputs_embeds=inputs_embeds,
                              **kwargs)

    host.forward = forward


def _biased_eager(host, mask, bias, batch_size, head_dim, query, key, value):
    """SmolVLM's eager attention (smolvlm_with_expert.eager_attention_forward)
    with an additive ``bias`` on the scores before its boolean mask."""
    heads, kv_heads = host.num_attention_heads, host.num_key_value_heads
    groups = heads // kv_heads
    length = key.shape[1]
    key = key[:, :, :, None, :].expand(batch_size, length, kv_heads, groups, head_dim)
    key = key.reshape(batch_size, length, kv_heads * groups, head_dim)
    value = value[:, :, :, None, :].expand(batch_size, length, kv_heads, groups, head_dim)
    value = value.reshape(batch_size, length, kv_heads * groups, head_dim)
    query = query.to(torch.float32).transpose(1, 2)
    key = key.to(torch.float32).transpose(1, 2)
    scores = torch.matmul(query, key.transpose(2, 3)) * head_dim**-0.5 + bias
    scores = torch.where(mask[:, None, :, :], scores, torch.finfo(torch.float32).min)
    probs = torch.nn.functional.softmax(scores, dim=-1).to(value.dtype)
    output = torch.matmul(probs, value.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    return output.reshape(batch_size, -1, kv_heads * groups * head_dim)


def install_diffusers_kv(head, control, *, conditioning_owner=None):
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
