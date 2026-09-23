"""AdaLN: a pooled scene vector that modulates the host's blocks.

Inspired by LPWM's context conditioning (arXiv:2603.04553, official
``ctx_mode="adaln"``), which follows DiT: a condition vector ``c`` produces a
scale, shift and gate per block through a zero-initialised projection. Here
``c`` is an attention pool over the entity set (one learned query, masked, so it
is permutation-invariant and ignores padding). Unlike LPWM, the condition is a
single scene summary rather than per-particle context, because the host is a
pretrained policy, not a particle dynamics model.

Two forms, by host:

* `SceneAdaLN` -- the host has no adaptive norm (post-norm ACT). It returns a
  (gate, scale, shift) triple per site, applied around each sublayer as
  ``x = LN(x + (1+a) f(x)) (1+g) + b``. DiT's ``a f(x)`` with ``a = 0`` would
  delete the pretrained sublayer, so every quantity is a delta around identity.
* `SceneVector` -- the host already modulates its blocks from a condition
  vector (pi0.5's adaRMS time vector, GR00T's DiT timestep embedding). The
  scene is *added* to that vector, so it reaches every block through the
  host's own pretrained modulation layers.

Both are exact identity at init, and an empty scene contributes exactly zero.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .entity import EntityEmbedding, embed_entities


class _PooledScene(nn.Module):
    """Entity embedding plus a one-query attention pool, shared by both forms."""

    def __init__(self, config: Any, width: int, heads: int | None = None) -> None:
        super().__init__()
        self.width = width
        self.embedding = EntityEmbedding(width, config.object_entity_normalizer)
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        # The host's dropout, for the same reason the KV branch takes it.
        self.pool = nn.MultiheadAttention(
            width,
            heads or int(config.object_attention_heads),
            dropout=float(getattr(config, "dropout", 0.0) or 0.0),
            batch_first=True,
        )

    def pooled(self, tokens: Tensor, mask: Tensor | None, batch: int) -> tuple[Tensor, Tensor]:
        """``[B, width]`` scene summaries and a ``[B]`` flag for empty scenes."""
        memory, padding = embed_entities(self.embedding, tokens, mask, batch)
        empty = padding.all(dim=1)
        safe = padding.clone()
        if empty.any():
            safe[empty, 0] = False
        query = self.query.to(memory.dtype).expand(memory.shape[0], -1, -1)
        pooled = self.pool(query, memory, memory, key_padding_mask=safe, need_weights=False)[0]
        return F.silu(pooled[:, 0]), empty

    @staticmethod
    def _no_scene(tokens: Tensor | None) -> bool:
        return tokens is None or tokens.shape[-2] == 0


class SceneAdaLN(_PooledScene):
    """(gate, scale, shift) per modulation site, for a host with no adaptive norm."""

    def __init__(self, config: Any, width: int, sites: int, *, heads: int | None = None) -> None:
        super().__init__(config, width, heads)
        if sites < 1:
            raise ValueError("SceneAdaLN needs at least one modulation site")
        self.sites = int(sites)
        self.modulation = nn.Linear(width, self.sites * 3 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    @property
    def is_live(self) -> bool:
        return bool(self.modulation.weight.any().item() or self.modulation.bias.any().item())

    def forward(self, tokens: Tensor | None, mask: Tensor | None, batch: int) -> Tensor | None:
        """``[B, sites, 3, width]`` as (gate, scale, shift), or None with no scene."""
        if self._no_scene(tokens):
            return None
        pooled, empty = self.pooled(tokens, mask, batch)
        out = self.modulation(pooled)
        # Zeroed after the projection: once its bias has trained, a zero input
        # would still modulate, and an empty scene must not.
        if empty.any():
            out = out.masked_fill(empty[:, None], 0)
        return out.view(pooled.shape[0], self.sites, 3, self.width)


class SceneVector(_PooledScene):
    """An additive delta on a host's own condition vector, ``[B, out_dim]``.

    ``out_dim`` may also be ``sites * 2 * width`` for a host whose norms take no
    condition at all (SmolVLA), which the caller reshapes into per-site
    (scale, shift) pairs.
    """

    def __init__(self, config: Any, width: int, out_dim: int, *, heads: int | None = None) -> None:
        super().__init__(config, width, heads)
        self.out_dim = int(out_dim)
        self.projection = nn.Linear(width, self.out_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    @property
    def is_live(self) -> bool:
        return bool(self.projection.weight.any().item() or self.projection.bias.any().item())

    def forward(self, tokens: Tensor | None, mask: Tensor | None, batch: int) -> Tensor | None:
        if self._no_scene(tokens):
            return None
        pooled, empty = self.pooled(tokens, mask, batch)
        out = self.projection(pooled)
        if empty.any():
            out = out.masked_fill(empty[:, None], 0)
        return out
