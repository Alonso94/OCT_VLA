"""Ablation transforms over already-built object tokens.

The tokens stored in the dataset always carry the full 15-d layout of
:mod:`oct_vla.data.object_tokens`: position (3), orientation (4), size (3),
visibility/confidence (2), role one-hot (3), ordered target-first.

Two of the sweep's arms need a *different* view of that same data, and both are
answers to "is the object arm winning because of object-centric perception, or
because the tokens hand it the task state?":

``role_stripped``
    Drop the role one-hot *and* the target-first ordering. Dropping the three
    dimensions alone is not enough -- the ordering leaks the same information,
    since the target is simply whichever token comes first.

    The replacement ordering must be *stable in time* as well as free of task
    state. Sorting on each frame's current positions satisfies the second but
    not the first: when two objects' coordinates cross, the token order flips
    mid-episode and the policy sees its inputs permute for no reason it can
    observe. Measured: the arm ordered that way fit worse than RGB alone
    (eval loss 0.088-0.094 against 0.073-0.076), which is the opposite of what
    removing a leak should do. So the ordering is instead fixed per episode,
    from the geometry at the first frame, and carried alongside the tokens.

``shuffle`` (evaluation only)
    Permute the valid tokens across object slots. A policy that genuinely reads
    the tokens degrades; one that has learned to ignore them does not.

Transforming here rather than at export time keeps one dataset on disk for
every arm, and -- more importantly -- means training and evaluation cannot
disagree about the layout, because both go through this module.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Width of the role one-hot at the tail of a stored token.
ROLE_DIMS = 3

#: Slice of a stored token holding the workcell position, used to re-order
#: tokens once the role ordering is discarded.
POSITION_SLICE = slice(0, 3)

TOKEN_MODES = ("full", "role_stripped")


def token_dim_for_mode(stored_dim: int, mode: str) -> int:
    """Width the policy sees for `mode`, given the stored token width."""
    if mode not in TOKEN_MODES:
        raise ValueError(f"unknown object token mode {mode!r}, expected one of {TOKEN_MODES}")
    return stored_dim - ROLE_DIMS if mode == "role_stripped" else stored_dim


def _sort_key(tokens: Tensor, mask: Tensor, rank: Tensor | None) -> Tensor:
    """Ordering indices free of task state, padding last.

    `rank` is the episode-stable ordering built at export time (see
    `oct_vla.data.object_tokens.stable_ranks`): each slot's position in an
    ordering fixed once, from the scene's geometry at the first frame. It is
    preferred because it cannot permute mid-episode.

    Without it -- an older dataset that has no rank column -- this falls back
    to sorting on the current positions. That is still task-independent, but it
    reorders whenever two objects cross, so it is a compatibility path rather
    than the intended behaviour.
    """
    if rank is not None:
        order = rank.to(torch.float32).argsort(dim=1, stable=True)
    else:
        positions = tokens[..., POSITION_SLICE]
        order = (
            torch.arange(tokens.shape[1], device=tokens.device).expand(tokens.shape[:2]).clone()
        )
        # A single sortable scalar would need known coordinate bounds, so sort
        # the axes in reverse priority and let the stable sort compose them
        # into a lexicographic (x, y, z) order.
        for axis in reversed(range(positions.shape[-1])):
            values = positions.gather(
                1, order.unsqueeze(-1).expand(-1, -1, positions.shape[-1])
            )[..., axis]
            order = order.gather(1, values.argsort(dim=1, stable=True))
    # Padding slots carry zeros and would otherwise sort into the middle.
    invalid = (~mask.bool()).gather(1, order).to(torch.int8)
    return order.gather(1, invalid.argsort(dim=1, stable=True))


def _permute(tokens: Tensor, mask: Tensor, order: Tensor) -> tuple[Tensor, Tensor]:
    gathered = tokens.gather(1, order.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
    return gathered, mask.gather(1, order)


def strip_roles(
    tokens: Tensor, mask: Tensor, rank: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Remove the role one-hot and the ordering that encodes the same thing."""
    tokens, mask = _permute(tokens, mask, _sort_key(tokens, mask, rank))
    return tokens[..., :-ROLE_DIMS].contiguous(), mask


def shuffle_tokens(
    tokens: Tensor, mask: Tensor, *, generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Permute the *valid* tokens within each scene.

    Padding stays at the tail, so the permutation changes which object sits in
    which slot without changing how many objects the policy is told about --
    otherwise the control would confound "tokens are scrambled" with "tokens
    are missing".
    """
    batch, slots = mask.shape
    valid = mask.bool()
    # Random keys for valid slots, +inf for padding, so argsort shuffles the
    # real objects among themselves and leaves padding at the end.
    keys = torch.rand((batch, slots), device=tokens.device, generator=generator)
    keys = keys.masked_fill(~valid, float("inf"))
    return _permute(tokens, mask, keys.argsort(dim=1))


def apply_token_mode(
    tokens: Tensor | None,
    mask: Tensor | None,
    *,
    mode: str = "full",
    shuffle: bool = False,
    generator: torch.Generator | None = None,
    rank: Tensor | None = None,
) -> tuple[Tensor | None, Tensor | None]:
    """Apply the configured ablations, in the order the arms define them.

    Shuffling runs *after* role-stripping so that the shuffled control of a
    role-stripped model scrambles exactly the tokens that model was trained on.
    """
    if tokens is None:
        return tokens, mask
    # Collapse an observation-history axis exactly as ObjectExpert does, so the
    # transforms below only ever see [B, N, D] and the two cannot disagree
    # about which frame the tokens describe.
    if tokens.ndim == 4:
        tokens = tokens[:, -1]
    if mask is not None and mask.ndim == 3:
        mask = mask[:, -1]
    if mask is None:
        mask = torch.ones(tokens.shape[:-1], dtype=tokens.dtype, device=tokens.device)
    if mode not in TOKEN_MODES:
        raise ValueError(f"unknown object token mode {mode!r}, expected one of {TOKEN_MODES}")
    if mode == "role_stripped":
        if rank is not None and rank.ndim == 3:
            rank = rank[:, -1]
        tokens, mask = strip_roles(tokens, mask, rank)
    if shuffle:
        tokens, mask = shuffle_tokens(tokens, mask, generator=generator)
    return tokens, mask
