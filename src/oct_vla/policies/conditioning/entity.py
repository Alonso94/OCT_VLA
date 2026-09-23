"""The entity contract every conditioning path reads, and its embedding.

A scene is an unordered set of up to ``N`` entities -- objects, grippers and
support surfaces -- each a ``TOKEN_DIM``-wide row, with a boolean mask marking
real rows. The columns are fixed by ``oct_vla.data.entity_tokens``:

    [ 0: 3]  position (workcell frame, metres)
    [ 3: 9]  rotation, 6D
    [ 9:12]  size (metres)
    [12]     gripper aperture
    [13:17]  entity type, one-hot

It is a set on purpose: nothing here depends on row order, so permuting the
entities cannot change any output.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

ENTITY_TOKENS = "observation.entity_tokens"
ENTITY_MASK = "observation.entity_mask"
TOKEN_DIM = 17
NUMERIC_DIM = 13
TYPE_DIM = TOKEN_DIM - NUMERIC_DIM


class EntityEmbedding(nn.Module):
    """Embed each entity row to the host's width, keeping units apart.

    The 13 continuous columns (metres, a 6D rotation, an aperture) and the
    4-column type one-hot get separate projections: one LayerNorm over metres
    and a categorical label would let either silently change the other's
    meaning. Position and size are standardised with statistics fitted on the
    *training split only* and carried in the policy config, so they travel with
    the checkpoint.
    """

    def __init__(self, width: int, normalizer: dict | None = None) -> None:
        super().__init__()
        mean, scale = torch.zeros(NUMERIC_DIM), torch.ones(NUMERIC_DIM)
        if normalizer is not None:
            from oct_vla.data.entity_tokens import EntityTokenNormalizer

            stats = EntityTokenNormalizer.from_dict(normalizer)
            mean[:3], scale[:3] = torch.tensor(stats.position_mean), torch.tensor(stats.position_std)
            mean[9:12], scale[9:12] = torch.tensor(stats.size_mean), torch.tensor(stats.size_std)
        self.register_buffer("numeric_mean", mean)
        self.register_buffer("numeric_scale", scale)
        self.numeric_projection = nn.Linear(NUMERIC_DIM, width)
        self.type_projection = nn.Linear(TYPE_DIM, width, bias=False)
        self.activation = nn.GELU()
        self.output_norm = nn.LayerNorm(width)

    def forward(self, tokens: Tensor) -> Tensor:
        """``[B, N, TOKEN_DIM]`` (or ``[B, T, N, TOKEN_DIM]``, last step) -> ``[B, N, width]``."""
        if tokens.ndim == 4:
            tokens = tokens[:, -1]
        if tokens.ndim != 3 or tokens.shape[-1] != TOKEN_DIM:
            raise ValueError(f"entity tokens must be [B,N,{TOKEN_DIM}] (or [B,T,N,{TOKEN_DIM}])")
        tokens = tokens.to(self.numeric_projection.weight.dtype)
        numeric = (tokens[..., :NUMERIC_DIM] - self.numeric_mean) / self.numeric_scale
        embedded = self.numeric_projection(numeric) + self.type_projection(tokens[..., NUMERIC_DIM:])
        return self.output_norm(self.activation(embedded))


def padding_of(tokens: Tensor, mask: Tensor | None) -> Tensor:
    """``[B, N]`` True where a row is padding. Reads the last step of a sequence."""
    if tokens.ndim == 4:
        tokens = tokens[:, -1]
    if mask is None:
        return torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    if mask.ndim == 3:
        mask = mask[:, -1]
    if mask.shape != tokens.shape[:2]:
        raise ValueError("entity mask must have shape [B,N]")
    return ~mask.bool()


def repeat_to(batch: int, tokens: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor | None]:
    """Tile the entity batch up to a host batch that repeats it whole.

    Flow-matching heads (GR00T, VLA-JEPA) repeat their batch with
    ``Tensor.repeat(r, ...)`` -- whole-batch tiling, not ``repeat_interleave``
    -- so the entities must be tiled the same way or each scene would condition
    another scene's samples.
    """
    if tokens.shape[0] == batch:
        return tokens, mask
    if not tokens.shape[0] or batch % tokens.shape[0]:
        raise ValueError("host batch must be a whole repeat of the entity batch")
    repeats = batch // tokens.shape[0]
    tokens = tokens.repeat(repeats, *([1] * (tokens.ndim - 1)))
    if mask is not None:
        mask = mask.repeat(repeats, *([1] * (mask.ndim - 1)))
    return tokens, mask


def embed_entities(
    embedding: EntityEmbedding, tokens: Tensor, mask: Tensor | None, batch: int
) -> tuple[Tensor, Tensor]:
    """Embedded ``[B, N, D]`` entities and their ``[B, N]`` padding, for a host batch.

    Padded rows are zeroed *before* embedding, so nothing stored in a padded
    slot (even a NaN) can reach a gradient.
    """
    tokens, mask = repeat_to(batch, tokens, mask)
    if tokens.ndim == 4:
        tokens = tokens[:, -1]
    padding = padding_of(tokens, mask)
    tokens = tokens.masked_fill(padding[..., None], 0)
    return embedding(tokens.to(embedding.numeric_projection.weight.dtype)), padding
