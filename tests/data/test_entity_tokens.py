"""Schema-v2 entity-token contract."""

import math

import pytest

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene, ObjectState
from oct_vla.core.state import ArmState, EEFState
from oct_vla.data.entity_tokens import (
    ENTITY_TOKEN_DIM,
    ENTITY_TOKEN_SCHEMA,
    EntitySupport,
    EntityTokenNormalizer,
    build_entity_tokens,
)


def _pose(x, y=0.0, z=0.9):
    return Pose((x, y, z), (0.0, 0.0, 0.0, 1.0), WORKCELL_FRAME)


def _eef():
    return EEFState(ArmState(_pose(-0.2), 0.25), ArmState(_pose(0.2), 0.75))


def _scene(*objects):
    return ObjectScene(0.0, tuple(objects))


def test_schema_v2_has_only_geometry_aperture_and_type_and_ignores_object_ids():
    left = ObjectState("arbitrary-id", _pose(0.1), (0.02, 0.03, 0.04), 1.0, 1.0)
    right = ObjectState("different-id", _pose(0.3), (0.05, 0.06, 0.07), 1.0, 1.0)
    tokens, mask = build_entity_tokens(_scene(right, left), _eef(), max_entities=6)

    assert len(tokens) == len(mask) == 6
    assert all(len(token) == ENTITY_TOKEN_DIM for token in tokens)
    assert mask == (True, True, True, True, False, False)
    # objects sort by geometry, not caller order or object identifiers
    assert tokens[0][:3] == (0.1, 0.0, 0.9)
    assert tokens[0][13:] == (1.0, 0.0, 0.0, 0.0)
    assert tokens[2][13:] == (0.0, 1.0, 0.0, 0.0)
    assert tokens[3][13:] == (0.0, 0.0, 1.0, 0.0)
    assert tokens[4] == (0.0,) * ENTITY_TOKEN_DIM


def test_supports_are_typed_and_count_toward_capacity():
    support = EntitySupport(_pose(0.0, 0.0, 0.5), (0.8, 0.5, 0.1))
    tokens, mask = build_entity_tokens(_scene(), _eef(), (support,), max_entities=3)
    assert mask == (True, True, True)
    assert tokens[-1][9:13] == (0.8, 0.5, 0.1, 0.0)
    assert tokens[-1][13:] == (0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="capacity"):
        build_entity_tokens(_scene(), _eef(), (support,), max_entities=2)


def test_train_stats_ignore_padding_and_leave_rotation_aperture_and_type_raw():
    normalizer = EntityTokenNormalizer.fit(
        [
            (
                (
                    (1.0, 2.0, 3.0, 0, 0, 0, 0, 0, 0, 2.0, 4.0, 6.0, 0, 1, 0, 0, 0),
                    (999.0,) * ENTITY_TOKEN_DIM,
                ),
                (True, False),
            ),
        ]
    )
    assert normalizer.position_mean == (1.0, 2.0, 3.0)
    assert normalizer.size_mean == (2.0, 4.0, 6.0)
    result = normalizer.normalize(
        ((1.0, 2.0, 3.0, 0, 0, 0, 0, 0, 0, 2.0, 4.0, 6.0, 0.7, 1, 0, 0, 0),), (True,)
    )
    assert result[0][:3] == (0.0, 0.0, 0.0)
    assert result[0][9:13] == (0.0, 0.0, 0.0, 0.7)
    assert result[0][13:] == (1.0, 0.0, 0.0, 0.0)
    payload = normalizer.to_dict()
    assert payload["schema"] == ENTITY_TOKEN_SCHEMA
    assert EntityTokenNormalizer.from_dict(payload) == normalizer


def test_nonfinite_valid_tokens_and_stats_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        EntityTokenNormalizer.fit([(((math.nan,) * ENTITY_TOKEN_DIM,), (True,))])
    with pytest.raises(ValueError, match="positive"):
        EntityTokenNormalizer((0, 0, 0), (1, 0, 1), (0, 0, 0), (1, 1, 1))


def test_rotation6_uses_xyzw_first_two_columns():
    half = math.sqrt(0.5)
    pose = Pose((0.0, 0.0, 0.0), (0.0, 0.0, half, half), WORKCELL_FRAME)
    obj = ObjectState("rotated", pose, (0.1, 0.2, 0.3), 1.0, 1.0)
    tokens, _ = build_entity_tokens(_scene(obj), _eef(), max_entities=3)
    assert tokens[0][3:9] == pytest.approx((0, 1, 0, -1, 0, 0), abs=1e-7)
