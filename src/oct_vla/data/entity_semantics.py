"""A per-variant semantic vector, sized to balance the geometric block.

The entity token carried 13 continuous geometry columns and a 4-way type
one-hot. Identity was going to ride along as an integer id looked up in an
embedding table, which has two problems. The table lives in the *policy*, so the
dataset stops being self-describing and two arms trained on the same corpus can
disagree about what id 3 means; and an id that was never trained has no row, so
an unseen mesh degrades to a reserved zero rather than to something the network
can process at all.

So the semantic content is a *vector in the dataset*, the same shape for every
entity, seen or unseen. Sixteen columns of geometry against sixteen of
semantics, which is what "balanced" means here: neither block can dominate the
encoder's input by sheer width before a single weight is trained.

    [ 0: 3]  position                        [16:20]  entity type one-hot
    [ 3: 9]  rotation, first two columns     [20:32]  variant code
    [ 9:12]  size
    [12]     gripper aperture
    [13]     volume
    [14:16]  two aspect ratios

The three derived scalars are real geometry, not padding -- volume and aspect
ratio are what distinguish a flat box from a tall one at equal bounding size,
and they give the geometric block its sixteen columns honestly.

**What the variant code can and cannot carry.** It is a deterministic function
of the variant's geometry, because in this corpus geometry *is* identity: one
asset, seven meshes, each with a distinct size. So the code adds no information
the geometry lacks. What it adds is *accessibility* -- identity as a direction
in a 12-d space rather than something the network must first recover by
discretising three continuous millimetre-scale columns. A corpus with two
visually distinct meshes at identical size would need a code derived from
appearance instead, and this module would be the place to change.

Deterministic from the geometry rather than assigned in encounter order, so the
same mesh gets the same code in every export, in training and at evaluation, and
an unseen mesh gets a code of its own instead of a reserved null.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

#: Columns of semantics: the 4-way entity type plus the variant code.
SEMANTIC_DIM = 16
#: Columns of geometry, matched to it.
GEOMETRIC_DIM = 16
#: The balanced entity token.
ENTITY_V3_DIM = GEOMETRIC_DIM + SEMANTIC_DIM

#: How much of the semantic block the variant code occupies; the rest is the
#: entity type one-hot, which is semantic too -- "this is a gripper" is not a
#: measurement.
VARIANT_CODE_DIM = 12


def variant_code(size_xyz: Sequence[float], *, dim: int = VARIANT_CODE_DIM) -> tuple[float, ...]:
    """A unit vector naming this mesh, reproducible from its geometry alone.

    Hashed rather than enumerated: an index assigned in encounter order would
    change when episodes are read in a different order, and would silently mean
    something different in a dataset exported from a different subset. A hash of
    the rounded size is stable across exports, machines and runs.

    Unit-normalised so no variant enters the encoder louder than another, and so
    the code's scale does not drift with `dim`.
    """
    from oct_vla.data.entity_identity import size_key

    seed = ",".join(f"{value:.4f}" for value in size_key(size_xyz)).encode("ascii")
    digest = hashlib.sha256(seed).digest()
    # Two bytes per component, mapped to [-1, 1]; sha256 gives 32 bytes, enough
    # for 16 components before it would need extending.
    if dim * 2 > len(digest):
        raise ValueError(f"variant code of {dim} components needs a longer digest")
    raw = [
        (int.from_bytes(digest[i * 2 : i * 2 + 2], "big") / 32767.5) - 1.0 for i in range(dim)
    ]
    norm = math.sqrt(sum(value * value for value in raw)) or 1.0
    return tuple(value / norm for value in raw)


def geometric_extras(size_xyz: Sequence[float]) -> tuple[float, float, float]:
    """Volume and two aspect ratios: shape at a size the bounding box hides.

    A flat bar and an upright box can share a largest extent and be entirely
    different to grasp. Log-scaled, because these span orders of magnitude while
    every other geometric column is a length in metres.
    """
    x, y, z = (max(float(value), 1e-6) for value in size_xyz)
    return (math.log(x * y * z), math.log(x / y), math.log(x / z))
