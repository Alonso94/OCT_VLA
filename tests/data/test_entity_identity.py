"""Identity recovered from geometry, for a corpus recorded before it existed."""

import pytest

from oct_vla.data.entity_identity import (
    UNSEEN_IDENTITY,
    EntityIdentityTable,
    size_key,
)

# The six geometries the leftmost corpus actually contains.
CORPUS = [
    (0.0540, 0.0479, 0.0774),
    (0.0565, 0.0592, 0.0771),
    (0.0581, 0.0777, 0.0692),
    (0.0662, 0.0744, 0.0965),
    (0.0708, 0.0781, 0.0666),
    (0.0729, 0.0778, 0.0697),
]


def test_distinct_meshes_get_distinct_identities():
    table = EntityIdentityTable.fit(CORPUS * 3)
    assert len(table) == 6
    assert table.cardinality == 7  # six variants plus the unseen row
    assert len({table.identity(s) for s in CORPUS}) == 6
    assert UNSEEN_IDENTITY not in {table.identity(s) for s in CORPUS}


def test_identities_are_stable_under_reordering():
    """The ids travel in the dataset and in the checkpoint, so a corpus read in
    a different episode order must not silently relabel every object."""
    forward = EntityIdentityTable.fit(CORPUS)
    backward = EntityIdentityTable.fit(list(reversed(CORPUS)))
    assert forward.to_dict() == backward.to_dict()


def test_geometry_never_seen_in_training_resolves_to_unseen():
    """Extending the table at evaluation would be a leak wearing the costume of
    generalisation: the embedding would carry a row it never trained."""
    table = EntityIdentityTable.fit(CORPUS[:4])
    assert table.identity(CORPUS[5]) == UNSEEN_IDENTITY
    assert table.identity((0.5, 0.5, 0.5)) == UNSEEN_IDENTITY


def test_the_closest_pair_in_the_corpus_stays_separate():
    """0.0708 against 0.0729 is the tightest pair; rounding must not merge
    two meshes into one identity, nor split one across two."""
    table = EntityIdentityTable.fit(CORPUS)
    assert table.identity((0.0708, 0.0781, 0.0666)) != table.identity((0.0729, 0.0778, 0.0697))
    # A stored float differing below the match precision is the same mesh.
    assert table.identity((0.07080001, 0.07810004, 0.06660002)) == table.identity(
        (0.0708, 0.0781, 0.0666)
    )


def test_round_trip_and_a_precision_change_is_refused():
    table = EntityIdentityTable.fit(CORPUS)
    assert EntityIdentityTable.from_dict(table.to_dict()).to_dict() == table.to_dict()
    stale = table.to_dict() | {"precision": 2}
    with pytest.raises(ValueError, match="precision"):
        EntityIdentityTable.from_dict(stale)


def test_size_key_rounds_consistently():
    assert size_key([1 / 3, 0.0, -0.0]) == size_key((0.3333, 0.0, 0.0))
