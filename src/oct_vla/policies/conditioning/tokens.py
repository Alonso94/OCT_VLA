"""Tokens: the entities placed in the host's own sequence, then dropped.

Inspired by LPWM's token conditioning (official ``ctx_mode="token"``), which
concatenates its context to the particle sequence. Here every *entity* becomes
a token -- a stronger, object-level version of the idea -- and the tokens are
removed again after the host has attended over them, so whatever reads the
host's output sees exactly its native layout.

Where the tokens go, by host:

* ACT: appended to the encoder sequence, behind a learned logit gate (`extend`).
* pi0.5, SmolVLA: prepended to the action-expert suffix as their own attention
  block (`prepend_entities`); the hosts read actions back as the last positions.
  As in ACT, the actions reach the entities through the learned gate
  (`gate_suffix_mask`), and the entities take no RoPE positions of their own
  (`realign_suffix_positions`), so every action sits where stage 1 put it.
  Without both, prepending moved the pi0.5 action chunk by 137 % at step 0.
* GR00T: prepended to the DiT sequence (in ``control_groot``).

Unlike KV and AdaLN this is **not identity at init**: the new keys enter the
host's softmax. ACT's gate starts them at ~e^gate of a native token's mass --
small, but with a live gradient, which a gate near -20 would not have.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .entity import EntityEmbedding, embed_entities


class EntityTokens(nn.Module):
    """Entity embeddings as host-width tokens, with a type code marking them."""

    def __init__(self, config: Any, width: int) -> None:
        super().__init__()
        self.width = width
        self.embedding = EntityEmbedding(width, config.object_entity_normalizer)
        self.projection = nn.Linear(width, width)
        self.position = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.gate_init = float(getattr(config, "object_tokens_gate_init", -4.0))
        self.gate = nn.Parameter(torch.tensor(self.gate_init))

    @property
    def is_live(self) -> bool:
        return bool(self.gate.item() != self.gate_init)

    def embed(self, tokens: Tensor, mask: Tensor | None, batch: int, dtype: torch.dtype):
        """Batch-first ``[B, N, D]`` tokens and their ``[B, N]`` padding."""
        memory, padding = embed_entities(self.embedding, tokens, mask, batch)
        entities = self.projection(memory) + self.position.to(memory.dtype)
        return entities.to(dtype), padding

    def extend(
        self,
        x: Tensor,
        pos_embed: Tensor | None,
        key_padding_mask: Tensor | None,
        tokens: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """ACT: append to a seq-first ``x [S, B, D]``; return x, positions, float mask.

        The float key-padding mask is what PyTorch attention *adds* to the
        scores: 0 for native tokens, the learned gate for real entities, -inf
        for padding.
        """
        steps, batch, _ = x.shape
        memory, padding = embed_entities(self.embedding, tokens, mask, batch)
        entities = self.projection(memory).to(x.dtype).transpose(0, 1)
        count = entities.shape[0]
        if pos_embed is not None:
            extra = self.position.to(pos_embed.dtype).expand(count, pos_embed.shape[1], -1)
            pos_embed = torch.cat([pos_embed, extra], dim=0)
        native = torch.zeros(batch, steps, dtype=x.dtype, device=x.device)
        if key_padding_mask is not None:
            native = native.masked_fill(key_padding_mask.bool(), float("-inf"))
        gate = self.gate.to(x.dtype).expand(batch, count).masked_fill(padding, float("-inf"))
        return torch.cat([x, entities], dim=0), pos_embed, torch.cat([native, gate], dim=1)


def scene_inputs(control):
    """(tokens, mask) for the current forward, or None when there is no scene."""
    inputs = control._inputs
    if inputs is None or inputs[0] is None or inputs[0].shape[-2] == 0:
        return None
    return inputs


def prepend_entities(control, embs: Tensor, pad_masks: Tensor, att_masks: Tensor):
    """Put the entity tokens in front of an ``embed_suffix`` sequence (pi0.5, SmolVLA).

    In front, because both hosts read their actions back as the *last*
    ``chunk_size`` suffix positions. The entities open an attention block of
    their own, so the block-causal mask (``make_att_2d_masks``) lets actions
    attend to entities and entities to the prefix, but never entities to the
    noisy actions: the conditioning is the same at every denoising step.
    Padded slots are pad-masked out as both queries and keys.
    """
    tokens = getattr(control, "incontext", None)
    inputs = scene_inputs(control)
    control._suffix_entities = None
    if tokens is None or inputs is None:
        return embs, pad_masks, att_masks
    entities, padding = tokens.embed(*inputs, embs.shape[0], embs.dtype)
    # Read by the host's expert-forward wrapper, which sees the masks and
    # positions the host builds from these pad masks.
    control._suffix_entities = SuffixEntities(entities.shape[1], ~padding, tokens.gate)
    opens = torch.zeros(padding.shape, dtype=att_masks.dtype, device=att_masks.device)
    opens[:, 0] = 1
    return (
        torch.cat([entities, embs], dim=1),
        torch.cat([~padding, pad_masks], dim=1),
        torch.cat([opens, att_masks], dim=1),
    )


class SuffixEntities:
    """Where the prepended entities sit in the suffix, for the host wrapper."""

    def __init__(self, count: int, real: Tensor, gate: Tensor) -> None:
        self.count = count
        self.real = real  # [B, N] True for a real (non-padded) entity
        self.gate = gate


def realign_suffix_positions(position_ids: Tensor, suffix_len: int, layout: SuffixEntities):
    """Give every action the RoPE position it had with no entities in front.

    The hosts number positions by a cumulative sum over the pad mask, so N real
    entities push every action N places along. Actions are moved back; the
    entities share the first action's position. Only the last ``suffix_len``
    columns (the suffix) change.
    """
    n = layout.count
    positions = position_ids.clone()
    suffix = positions[:, -suffix_len:]
    real = layout.real.sum(dim=1, keepdim=True).to(positions.dtype)
    suffix[:, n:] -= real
    suffix[:, :n] = suffix[:, n : n + 1]
    return positions


def entity_key_bias(layout: SuffixEntities, rows: int, keys: int, suffix_len: int, dtype):
    """``[B, 1, rows, keys]`` additive scores: the gate where an action query
    meets a real entity key, zero elsewhere. The queries are the last ``rows``
    of the sequence and the keys end with the suffix."""
    n = layout.count
    batch = layout.real.shape[0]
    bias = torch.zeros(batch, 1, rows, keys, dtype=dtype, device=layout.real.device)
    start = keys - suffix_len
    actions = rows - (suffix_len - n)
    gate = layout.gate.to(dtype) * layout.real.to(dtype)  # [B, N]
    bias[:, 0, actions:, start : start + n] = gate[:, None, :]
    return bias


def gate_suffix_mask(mask4d: Tensor, suffix_len: int, layout: SuffixEntities) -> Tensor:
    """Add the gate to an additive ``[B, 1, Lq, Lk]`` mask (pi0.5), leaving
    masked entries masked."""
    bias = entity_key_bias(layout, mask4d.shape[-2], mask4d.shape[-1], suffix_len, mask4d.dtype)
    return torch.where(mask4d == 0, mask4d + bias, mask4d)
