"""Object identity recovered from geometry, for corpora recorded without it.

The leftmost corpus holds six visually distinct meshes -- six of the seven
`113_coffee-box` variants, 18 to 81 episodes each -- but it predates
`ObjectState.category`, so nothing in it names them. Re-collecting to add a
label would be wasteful and, as the multi-category attempt showed, risky: five
of six candidate assets turned out to be unplannable, and a recollection buys
nothing the recordings do not already contain.

An asset variant's size is deterministic: `upright_size(model, model_id)` is a
property of the mesh, not of the episode. So distinct sizes *are* distinct
identities, and the map inverts exactly with no simulator access and no asset
table. Six sizes appear in the corpus and no scene mixes two, which is why this
is a lookup rather than a clustering problem.

Fitted on the training partition alone, like the geometric statistics. A variant
appearing only in validation resolves to `UNSEEN_IDENTITY` rather than silently
extending the table, because a category embedding trained on ids it never saw is
the kind of leak that looks like generalisation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

#: Decimals kept when matching a recorded size to a variant. The corpus's six
#: geometries differ in the third decimal at the closest pair (0.0708 against
#: 0.0729), and a recorded size is a stored float rather than a re-measured one,
#: so four decimals separates them without splitting one variant in two.
IDENTITY_PRECISION = 4

#: Reserved for an object whose geometry was never seen in training. Zero so an
#: embedding table's first row carries "unknown", and every fitted identity is
#: numbered from one.
UNSEEN_IDENTITY = 0


def size_key(size_xyz: Sequence[float]) -> tuple[float, ...]:
    """The comparable form of a recorded size."""
    return tuple(round(float(value), IDENTITY_PRECISION) for value in size_xyz)


class EntityIdentityTable:
    """Maps an object's geometry to a stable integer identity."""

    def __init__(self, keys: Sequence[tuple[float, ...]]) -> None:
        # Sorted, so the same corpus always yields the same ids regardless of
        # the order episodes happen to be read in -- the ids travel in the
        # dataset and in the checkpoint, and a reshuffle would silently
        # relabel every object.
        self._ids = {key: index for index, key in enumerate(sorted(keys), start=1)}

    @classmethod
    def fit(cls, sizes: Iterable[Sequence[float]]) -> EntityIdentityTable:
        return cls(sorted({size_key(size) for size in sizes}))

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def cardinality(self) -> int:
        """Embedding rows required: every fitted identity, plus unseen."""
        return len(self._ids) + 1

    def identity(self, size_xyz: Sequence[float]) -> int:
        return self._ids.get(size_key(size_xyz), UNSEEN_IDENTITY)

    def to_dict(self) -> dict:
        return {
            "precision": IDENTITY_PRECISION,
            "unseen": UNSEEN_IDENTITY,
            "identities": [
                {"size_xyz": list(key), "id": value} for key, value in sorted(self._ids.items())
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> EntityIdentityTable:
        if data.get("precision") != IDENTITY_PRECISION:
            raise ValueError(
                f"identity table was fitted at precision {data.get('precision')}, "
                f"but this build matches at {IDENTITY_PRECISION}"
            )
        table = cls(())
        table._ids = {tuple(row["size_xyz"]): int(row["id"]) for row in data["identities"]}
        if UNSEEN_IDENTITY in table._ids.values():
            raise ValueError("a fitted identity collides with the reserved unseen id")
        return table
