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


# ------------------------------------------------- the balanced v3 token


def test_v3_is_sixteen_geometric_against_sixteen_semantic():
    """Balance is the point: neither block may dominate the encoder's input by
    sheer width before a single weight is trained."""
    from oct_vla.data.entity_semantics import (
        ENTITY_V3_DIM,
        GEOMETRIC_DIM,
        SEMANTIC_DIM,
        VARIANT_CODE_DIM,
    )
    from oct_vla.data.entity_tokens import ENTITY_TYPES

    assert GEOMETRIC_DIM == SEMANTIC_DIM == 16
    assert ENTITY_V3_DIM == 32
    assert VARIANT_CODE_DIM + len(ENTITY_TYPES) == SEMANTIC_DIM


def test_the_variant_code_is_a_deterministic_unit_vector_per_mesh():
    from oct_vla.data.entity_semantics import variant_code

    codes = {s: variant_code(s) for s in CORPUS}
    assert len({tuple(c) for c in codes.values()}) == len(CORPUS), "two meshes share a code"
    for s, c in codes.items():
        assert abs(sum(v * v for v in c) - 1.0) < 1e-9, "codes must be unit norm"
        assert variant_code(s) == c, "a code must not depend on call order"
    # Derived from geometry, so a mesh never seen in training still has one --
    # unlike an embedding row, which would not exist.
    unseen = variant_code((0.0216, 0.0774, 0.0571))
    assert abs(sum(v * v for v in unseen) - 1.0) < 1e-9
    assert tuple(unseen) not in {tuple(c) for c in codes.values()}


def test_a_v3_token_carries_both_blocks_and_grippers_carry_neither_size_nor_code():
    from oct_vla.core.frames import WORKCELL_FRAME, Pose
    from oct_vla.core.objects import ObjectScene, ObjectState
    from oct_vla.core.state import ArmState, EEFState
    from oct_vla.data.entity_tokens import ENTITY_TOKEN_SCHEMA_V3, build_entity_tokens

    pose = Pose((0.1, 0.0, 0.9), (0.0, 0.0, 0.0, 1.0), WORKCELL_FRAME)
    scene = ObjectScene(0.0, (ObjectState("a", pose, (0.06, 0.07, 0.08), 1.0, 1.0),))
    eef = EEFState(ArmState(pose, 1.0), ArmState(pose, 0.0))
    tokens, mask = build_entity_tokens(scene, eef, (), 6, schema=ENTITY_TOKEN_SCHEMA_V3)

    assert all(len(t) == 32 for t in tokens)
    obj = tokens[0]
    assert obj[16:20] == (1.0, 0.0, 0.0, 0.0), "movable type one-hot at the semantic head"
    assert any(v != 0.0 for v in obj[20:32]), "a sized object must carry a variant code"
    assert any(v != 0.0 for v in obj[13:16]), "and its derived shape scalars"

    gripper = tokens[1]
    assert gripper[16:20] == (0.0, 1.0, 0.0, 0.0)
    assert gripper[20:32] == (0.0,) * 12, "a gripper has no mesh identity to name"
    assert gripper[13:16] == (0.0, 0.0, 0.0), "nor a shape, since its size is structural"
    # One movable plus two grippers; the rest of the capacity is padding.
    assert mask[:3] == (True, True, True) and not any(mask[3:])
    assert all(v == 0.0 for v in tokens[3])
