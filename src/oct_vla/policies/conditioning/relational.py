"""The relational composer: one learned message-passing step over the entity set (``kv_rel``).

The skeleton's RQ2 (``paper/2_paper_skeleton.tex``, eq. 1, goal-free variant): does
*learning how entities relate* add anything over the flat entity tokens ``kv`` attends?
The composer runs after the entity embedding and before the KV branch's ``to_k``/``to_v``::

    h_i = e_i + gamma * sum_{j != i} alpha_ij * phi(e_j, r_ij)
    alpha_ij = softmax_j ( q(e_i) . (k(e_j) + r_ij) / sqrt(d) )

- ``r_ij`` embeds the relative geometry (position difference, in decimetres) and the
  pair's entity types. That is where gripper–object and gripper–gripper relations live.
- ``gamma`` starts at zero, so a ``kv_rel`` model is numerically ``kv`` at step 0. Its
  gradient only appears once ``to_v`` has moved; it is not a zero gate on a zero branch.

Same entities, same KV operator, same host hooks: only the per-entity tokens differ,
so ``kv`` against ``kv_rel`` tests composition, not access to poses. The PFN's own
relational ablation (difference edges, ``prior/model.py``) is the offline counterpart.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .kv import KVAttention

#: v2 token layout (data/entity_tokens.py): position first, type one-hot last.
_POS = slice(0, 3)
_TYPES = slice(13, 17)
#: Relative positions in decimetres: shelf-scale offsets become O(1).
_POS_SCALE = 10.0


class RelationalComposer(nn.Module):
    """One masked relation-attention step over embedded entities, residual and zero-gated."""

    def __init__(self, width: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or width
        self.query = nn.Linear(width, hidden, bias=False)
        self.key = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.relation = nn.Sequential(nn.Linear(3 + 2 * 4, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden))
        self.out = nn.Linear(hidden, width)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = 1.0 / math.sqrt(hidden)

    def forward(self, embedded: Tensor, tokens: Tensor) -> Tensor:
        """``embedded`` [B, N, W] (entity embeddings), ``tokens`` [B, N, 17] raw rows."""
        types = tokens[..., _TYPES]
        valid = types.sum(-1) > 0.5
        pos = tokens[..., _POS] * _POS_SCALE
        n = tokens.shape[1]
        rel_in = torch.cat([
            pos[:, :, None] - pos[:, None, :],
            types[:, :, None].expand(-1, -1, n, -1),
            types[:, None].expand(-1, n, -1, -1),
        ], -1).to(embedded.dtype)
        r = self.relation(rel_in)  # [B, N, N, H]
        q, k, v = self.query(embedded), self.key(embedded), self.value(embedded)
        scores = torch.einsum("bih,bijh->bij", q, k[:, None] + r) * self.scale
        allowed = valid[:, None, :] & valid[:, :, None]
        allowed = allowed & ~torch.eye(n, dtype=torch.bool, device=tokens.device)[None]
        scores = scores.masked_fill(~allowed, float("-inf"))
        # A row with no valid partner (padding, or a lone entity) gets no message.
        none = ~allowed.any(-1)
        weights = torch.softmax(scores.masked_fill(none[..., None], 0.0), -1)
        weights = weights.masked_fill(none[..., None], 0.0)
        message = torch.einsum("bij,bijh->bih", weights, v[:, None] + r)
        return embedded + self.gamma * self.out(message)


class RelationalKVAttention(KVAttention):
    """``KVAttention`` whose entity tokens are composed relationally first (``kv_rel``)."""

    def __init__(self, config: Any, width: int, *, heads: int) -> None:
        super().__init__(config, width, heads=heads)
        if getattr(config, "object_visual_dim", None):
            raise ValueError("kv_rel reads the geometry-only v2 tokens; "
                             "it has no visual-feature variant")
        # A sibling of the embedding, so a kv_rel checkpoint is kv's keys plus
        # `composer.*` and nothing renamed.
        self.composer = RelationalComposer(self.entity_width)

    def _compose(self, memory: Tensor, tokens: Tensor) -> Tensor:
        return self.composer(memory, tokens)
