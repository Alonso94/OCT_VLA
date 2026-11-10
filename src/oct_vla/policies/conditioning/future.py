"""The future-keyframe branch: a second, separately normalised KV term per layer.

PFN-VLA conditions the action head on the *next interaction keyframe* -- which
object moves, where to, with which gripper -- predicted by the frozen prior
(``oct_vla.prior``) or, as an upper bound, computed by the oracle. It enters as
its own set ``observation.future_keyframe`` ``[B, N_f, FUTURE_DIM]`` with mask
``observation.future_keyframe_mask`` ``[B, N_f]``, attended by a second
`KVAttention` in every hooked layer::

    out = W_o [ softmax(Q K^T) V + softmax(Q K_z^T) V_z + softmax(Q K_f^T) V_f ] + b_o

Why a second branch rather than more rows in the entity set: appending rows to
the entity keys would change the entity softmax of a trained ``kv`` model, so
``kv`` and ``kv + future`` would no longer share a starting point. The two
softmaxes here are independent, and ``V_f`` is zero-initialised, so a ``kv +
future`` model is numerically its ``kv`` self until training moves it.

There is no extra scalar gate: a zero gate times a zero ``V_f`` gives both a
zero gradient, and the branch would never leave its initialisation.

Row layout (``FUTURE_DIM`` = 17 + 3 + 4 = 24 columns)::

    [ 0:17]  entity-schema geometry of the row (entity.py): position, 6D
             rotation, size, aperture, type one-hot
    [17:20]  role one-hot: object at the keyframe | acting gripper at the
             keyframe | object at its goal
    [20:24]  event one-hot: grasp | place | regrasp | release (prior.schema)
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from oct_vla.data.future_keyframes import (  # noqa: F401  (the row contract)
    FUTURE_DIM,
    FUTURE_EVENTS,
    FUTURE_EXTRA,
    FUTURE_ROLES,
)

from .entity import TOKEN_DIM
from .kv import KVAttention

if FUTURE_DIM - FUTURE_EXTRA != TOKEN_DIM:
    raise ImportError("future keyframe rows must start with an entity-schema token")

FUTURE_KEYFRAME = "observation.future_keyframe"
FUTURE_MASK = "observation.future_keyframe_mask"


class FutureKVAttention(KVAttention):
    """`KVAttention` over keyframe rows: the same zero-initialised ``to_k``/``to_v``,
    with the role and event columns embedded by a projection of their own."""

    def __init__(self, config: Any, width: int, *, heads: int) -> None:
        super().__init__(_GeometryOnly(config), width, heads=heads)
        # Rebuild the embedding with the 7 categorical columns. Unlike the
        # visual columns of the entity branch, they are not zero-initialised:
        # to_k/to_v already make the branch inert, and a second zero factor
        # would leave the role/event projection with no gradient at step 0.
        from .entity import EntityEmbedding

        self.embedding = EntityEmbedding(
            self.entity_width, config.object_entity_normalizer, visual_dim=FUTURE_EXTRA
        )
        self.embedding.visual_projection.reset_parameters()


class _GeometryOnly:
    """A config view without ``object_visual_dim``: the future rows carry no
    per-object visual feature, whatever the entity branch uses."""

    def __init__(self, config: Any) -> None:
        self._config = config

    def __getattr__(self, name: str) -> Any:
        if name == "object_visual_dim":
            return None
        return getattr(self._config, name)


def check_future_inputs(keyframe: Tensor, mask: Tensor, *, rows: int) -> None:
    """The shape contract at the policy boundary: ``[B, rows, FUTURE_DIM]`` and ``[B, rows]``.

    Strict on purpose. The entity path quietly reads the last step of a 4-D
    history; a keyframe tensor of the wrong rank here is a wiring bug, not a
    history, and must not be reinterpreted.
    """
    if keyframe.ndim != 3 or keyframe.shape[1:] != (rows, FUTURE_DIM):
        raise ValueError(f"{FUTURE_KEYFRAME} must be [B, {rows}, {FUTURE_DIM}]; "
                         f"got {tuple(keyframe.shape)}")
    if mask.shape != keyframe.shape[:2]:
        raise ValueError(f"{FUTURE_MASK} must be [B, {rows}]; got {tuple(mask.shape)}")

