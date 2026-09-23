"""KV: ControlVLA's added attention term, for one host attention layer.

ControlVLA (arXiv:2506.16211; official ``kvcontrol_transformer.py``) adds a
second, independently normalised attention term over a control set to a host
attention layer::

    out = W_o [ softmax(Q K^T / sqrt d) V  +  softmax(Q K_z^T / sqrt d) V_z ] + b_o

``Q`` is the host's own projected query and ``W_o, b_o`` its output
projection; only ``K_z, V_z`` are new, and both are zero-initialised, so the
conditioned policy is numerically its stage-1 self at step 0.

`KVAttention` computes just the added term ``W_o softmax(Q K_z^T) V_z``, with no
output bias (the host applies ``b_o`` once). How the host's query is obtained
and where the result is added is the backbone's business (``hosts.py`` and
``control_act``). The module never stores a reference to a host module: that
would register the host twice and change checkpoint keys.

Differences from ControlVLA, all deliberate: the control set is an explicit
entity set rather than visual object features; padded entities are masked; and
there is no positional code over entities, because a set has no order.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .entity import EntityEmbedding, padding_of, repeat_to


def packed_mha_query(attention: nn.MultiheadAttention, query: Tensor) -> Tensor:
    """The ``[B, H, T, D]`` query a packed ``nn.MultiheadAttention`` computes.

    ACT's attention is packed ``MultiheadAttention`` with no RoPE, so slicing
    the Q third of its in-projection reproduces its native query exactly.
    """
    if not attention.batch_first:
        query = query.transpose(0, 1)
    if attention.in_proj_weight is None:
        weight = attention.q_proj_weight
    else:
        weight = attention.in_proj_weight[: attention.embed_dim]
    bias = None if attention.in_proj_bias is None else attention.in_proj_bias[: attention.embed_dim]
    projected = F.linear(query, weight, bias)
    batch, steps, width = projected.shape
    return projected.view(batch, steps, attention.num_heads, width // attention.num_heads).transpose(1, 2)


class KVAttention(nn.Module):
    """The zero-initialised entity K/V branch of one host attention layer."""

    def __init__(self, config: Any, width: int, *, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"heads={heads} does not divide host width {width}")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.embedding = EntityEmbedding(width, config.object_entity_normalizer)
        self.to_k = nn.Linear(width, width)
        self.to_v = nn.Linear(width, width)
        for layer in (self.to_k, self.to_v):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @property
    def is_live(self) -> bool:
        """Whether training has moved V_z off zero; False means the branch is inert."""
        return bool(self.to_v.weight.any().item() or self.to_v.bias.any().item())

    def forward(
        self,
        projected_query: Tensor,
        tokens: Tensor | None,
        mask: Tensor | None,
        *,
        output_weight: Tensor,
        scaling: float | None = None,
        dropout: float = 0.0,
    ) -> Tensor:
        """``W_o softmax(Q K_z^T * scale) V_z`` as ``[B, T, width_out]``, no bias.

        ``projected_query`` is the host's ``[B, H, T, D]`` query after its own
        positional transform. ``dropout`` is the host's attention dropout,
        applied here too so both terms of the sum are regularised alike -- an
        undropped object path would be the less regularised one, biasing the
        very comparison this branch exists for. An empty scene returns an exact
        zero residual.
        """
        batch, heads, steps, head_dim = projected_query.shape
        if heads != self.heads or head_dim != self.head_dim:
            raise ValueError("projected_query heads do not match this KVAttention")
        if tokens is None or tokens.shape[-2] == 0:
            return projected_query.new_zeros((batch, steps, output_weight.shape[0]))
        tokens, mask = repeat_to(batch, tokens, mask)
        if tokens.ndim == 4:
            tokens = tokens[:, -1]
        padding = padding_of(tokens, mask)
        # Mask before embedding: nothing in a padded slot, even a NaN, reaches a gradient.
        memory = self.embedding(tokens.masked_fill(padding[..., None], 0).to(self.to_k.weight.dtype))
        empty = padding.all(dim=1)

        def heads_of(x: Tensor) -> Tensor:
            return x.to(projected_query.dtype).view(batch, -1, heads, head_dim).transpose(1, 2)

        k, v = heads_of(self.to_k(memory)), heads_of(self.to_v(memory))
        # Scores in at least float32: bf16 accumulation over the head dimension
        # shifts the softmax. (`.float()` would also *down*cast float64, which
        # silently caps any numerical gradient check.)
        accum = torch.promote_types(projected_query.dtype, torch.float32)
        scale = scaling if scaling is not None else 1 / math.sqrt(head_dim)
        scores = (projected_query.to(accum) @ k.to(accum).transpose(-2, -1)) * scale
        # An all-padding row would softmax to NaN; unmask one slot for the
        # arithmetic, then zero the whole row's residual below.
        safe = padding.clone()
        if empty.any():
            safe[empty, 0] = False
        weights = torch.softmax(scores.masked_fill(safe[:, None, None, :], float("-inf")), dim=-1)
        if dropout:
            weights = F.dropout(weights, p=dropout, training=self.training)
        context = (weights.to(v.dtype) @ v).transpose(1, 2).reshape(batch, steps, self.width)
        residual = F.linear(context.to(output_weight.dtype), output_weight, None).to(projected_query.dtype)
        if empty.any():
            residual = residual.masked_fill(empty[:, None, None], 0)
        return residual
