"""A non-invasive ACT decoder cross-attention adapter."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from oct_vla.policies.layerwise_attention import LayerwiseObjectAttention


class LayerwiseACTCrossAttention(nn.Module):
    """Apply a ControlVLA residual to ACT's native cross-attention result.

    The native module is passed into ``forward`` rather than stored, avoiding
    duplicate registration and preserving every pretrained ACT state-dict key.
    Mount this adapter on each ACT decoder layer as ``object_attention``.
    """

    def __init__(self, config: Any, width: int, heads: int | None = None) -> None:
        super().__init__()
        self.object_attention = LayerwiseObjectAttention(config, width, heads=heads)

    def forward(
        self,
        host_attention: nn.MultiheadAttention,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        entity_tokens: Tensor | None,
        entity_mask: Tensor | None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor | None]:
        """Run ACT cross-attention and add the no-bias object residual."""
        native, weights = host_attention(query, key, value, **kwargs)
        residual = self.object_attention.mha_residual(
            host_attention, query, entity_tokens, entity_mask
        )
        if not host_attention.batch_first:
            residual = residual.transpose(0, 1)
        return native + residual, weights
