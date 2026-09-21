"""Layerwise ControlVLA object-attention primitives.

The module deliberately owns only the new object path.  It never stores a
reference to a host attention module: doing so registers the host twice and
changes checkpoint key paths.  Callers pass the host's already projected query
(after any RoPE) and its output projection weight.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def packed_mha_query(attention: nn.MultiheadAttention, query: Tensor) -> Tensor:
    """Return a batch-first, head-split query projected by ``attention``.

    This is the ACT path.  ACT's decoder cross-attention is standard packed
    ``MultiheadAttention``; it has no RoPE, so its native Q projection is the
    exact query used here.
    """
    if not attention.batch_first:
        query = query.transpose(0, 1)
    if attention.in_proj_weight is None:
        projected = F.linear(
            query,
            attention.q_proj_weight,
            None
            if attention.in_proj_bias is None
            else attention.in_proj_bias[: attention.embed_dim],
        )
    else:
        projected = F.linear(
            query,
            attention.in_proj_weight[: attention.embed_dim],
            attention.in_proj_bias[: attention.embed_dim]
            if attention.in_proj_bias is not None
            else None,
        )
    batch, steps, width = projected.shape
    return projected.view(
        batch, steps, attention.num_heads, width // attention.num_heads
    ).transpose(1, 2)


class EntityV2Embedding(nn.Module):
    """Encode the 17-column simulator entity contract without mixing units.

    The first 13 columns are continuous geometry (position, 6D rotation,
    size, aperture); the final four columns are a fixed entity type one-hot.
    Separate normalisation/projections prevent a LayerNorm over metres and a
    categorical label from silently changing the meaning of either.
    """

    numeric_dim = 13
    type_dim = 4

    def __init__(self, width: int, normalizer: dict | None = None) -> None:
        super().__init__()
        mean, scale = torch.zeros(13), torch.ones(13)
        if normalizer is not None:
            from oct_vla.data.entity_tokens import EntityTokenNormalizer

            stats = EntityTokenNormalizer.from_dict(normalizer)
            mean[:3], scale[:3] = (
                torch.tensor(stats.position_mean),
                torch.tensor(stats.position_std),
            )
            mean[9:12], scale[9:12] = torch.tensor(stats.size_mean), torch.tensor(stats.size_std)
        self.register_buffer("numeric_mean", mean)
        self.register_buffer("numeric_scale", scale)
        self.output_norm = nn.LayerNorm(width)
        self.numeric_projection = nn.Linear(self.numeric_dim, width)
        self.type_projection = nn.Linear(self.type_dim, width, bias=False)
        self.activation = nn.GELU()

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim == 4:
            tokens = tokens[:, -1]
        if tokens.ndim != 3 or tokens.shape[-1] != 17:
            raise ValueError("entity_v2 tokens must have shape [B,N,17] (or [B,T,N,17])")
        tokens = tokens.to(self.numeric_projection.weight.dtype)
        numeric = (tokens[..., :13] - self.numeric_mean) / self.numeric_scale
        entity_type = tokens[..., 13:]
        return self.output_norm(
            self.activation(self.numeric_projection(numeric) + self.type_projection(entity_type))
        )


class LayerwiseObjectAttention(nn.Module):
    """ControlVLA's added KV branch for one host action-attention layer.

    ``forward`` returns only the object residual after the *host* output
    projection with its bias removed.  The caller adds it to native attention
    output.  Therefore the native and object contexts are conceptually summed
    before the shared output projection, while the host bias is applied once.
    """

    def __init__(self, config: Any, width: int, *, heads: int | None = None) -> None:
        super().__init__()
        self.width = width
        self.heads = heads or int(getattr(config, "object_attention_heads", 8))
        if width % self.heads:
            raise ValueError(f"heads={self.heads} does not divide host width {width}")
        self.head_dim = width // self.heads
        rep = getattr(config, "object_representation", "legacy")
        if rep not in ("legacy", "entity_v2"):
            raise ValueError("object_representation must be 'legacy' or 'entity_v2'")
        self.object_representation = rep
        if rep == "entity_v2":
            self.embedding = EntityV2Embedding(
                width, getattr(config, "object_entity_normalizer", None)
            )
        else:
            token_dim = int(
                getattr(
                    config, "effective_object_token_dim", getattr(config, "object_token_dim", 15)
                )
            )
            self.embedding = nn.Sequential(
                nn.LayerNorm(token_dim), nn.Linear(token_dim, width), nn.GELU()
            )
        self.to_k = nn.Linear(width, width)
        self.to_v = nn.Linear(width, width)
        # Both K and V, including biases, are zeroed as in ControlVLA.
        for layer in (self.to_k, self.to_v):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @property
    def is_live(self) -> bool:
        return bool(self.to_v.weight.any().item() or self.to_v.bias.any().item())

    def encode(self, tokens: Tensor) -> Tensor:
        if tokens.ndim == 4:
            tokens = tokens[:, -1]
        return self.embedding(tokens)

    def _padding(self, tokens: Tensor, mask: Tensor | None) -> Tensor:
        if tokens.ndim == 4:
            tokens = tokens[:, -1]
        if mask is None:
            return torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if mask.ndim == 3:
            mask = mask[:, -1]
        if mask.shape != tokens.shape[:2]:
            raise ValueError("entity mask must have shape [B,N]")
        return ~mask.bool()

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
        """Return ``W_out(softmax(QKz^T/sqrt(d))Vz)`` with no output bias.

        ``projected_query`` is `[B,H,T,D]`, specifically the host's Q after
        its native positional transform.  Empty scenes return an exact zero
        residual, rather than a finite but semantically invented dummy token.

        ``dropout`` is the host's own attention dropout, applied to this
        branch's probabilities so both terms of the sum are regularised alike.
        A host that drops 10 % of its native attention while the object branch
        drops none leaves the object path systematically less regularised --
        which would bias an object-versus-RGB comparison toward conditioning,
        in the experiment built to measure exactly that. ACT's default is 0.1.
        """
        batch, heads, steps, head_dim = projected_query.shape
        if heads != self.heads or head_dim != self.head_dim:
            raise ValueError("projected_query heads do not match LayerwiseObjectAttention")
        if tokens is None or tokens.shape[-2] == 0:
            return projected_query.new_zeros((batch, steps, output_weight.shape[0]))
        if tokens.shape[0] != batch:
            if not tokens.shape[0] or batch % tokens.shape[0]:
                raise ValueError("action batch must be a whole repeat of the entity batch")
            repeats = batch // tokens.shape[0]
            tokens = tokens.repeat(repeats, *([1] * (tokens.ndim - 1)))
            if mask is not None:
                mask = mask.repeat(repeats, *([1] * (mask.ndim - 1)))
        if tokens.ndim == 4:
            tokens = tokens[:, -1]
        padding = self._padding(tokens, mask)
        # Mask before projection: even NaNs in padded storage cannot pollute gradients.
        tokens = tokens.masked_fill(padding[..., None], 0)
        memory = self.encode(tokens.to(self.to_k.weight.dtype))
        empty = padding.all(dim=1)
        k = (
            self.to_k(memory)
            .to(projected_query.dtype)
            .view(batch, -1, heads, head_dim)
            .transpose(1, 2)
        )
        v = (
            self.to_v(memory)
            .to(projected_query.dtype)
            .view(batch, -1, heads, head_dim)
            .transpose(1, 2)
        )
        scores = (projected_query.float() @ k.float().transpose(-2, -1)) * (
            scaling if scaling is not None else 1 / math.sqrt(head_dim)
        )
        # Prevent NaNs; reset empty rows to zero after attention to guarantee
        # they cannot gain a bias or a residual from the output projection.
        safe_padding = padding.clone()
        if empty.any():
            safe_padding[empty, 0] = False
        scores = scores.masked_fill(safe_padding[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        if dropout:
            weights = F.dropout(weights, p=dropout, training=self.training)
        context = weights.to(v.dtype) @ v
        context = context.transpose(1, 2).reshape(batch, steps, self.width)
        residual = F.linear(context.to(output_weight.dtype), output_weight, None).to(
            projected_query.dtype
        )
        if empty.any():
            residual = residual.masked_fill(empty[:, None, None], 0)
        return residual

    def mha_residual(
        self,
        attention: nn.MultiheadAttention,
        query: Tensor,
        tokens: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        """Convenience route for ACT's packed ``nn.MultiheadAttention``."""
        return self(
            packed_mha_query(attention, query),
            tokens,
            mask,
            output_weight=attention.out_proj.weight,
        )
