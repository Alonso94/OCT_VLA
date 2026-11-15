"""The online effect oracle (keyframes.ShelfEffectOracle) and its server source."""

from __future__ import annotations

import numpy as np

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene, ObjectState
from oct_vla.core.state import ArmState, EEFState
from oct_vla.data.entity_tokens import ENTITY_TOKEN_DIM
from oct_vla.data.future_keyframes import FUTURE_DIM, FUTURE_ROWS


def _scene():
    down = (0.0, 0.0, 0.0, 1.0)
    objs = tuple(ObjectState(f"obj_{i}", Pose((x, -0.25, 0.79), down, WORKCELL_FRAME),
                             (0.05, 0.05, 0.10), 1.0, 1.0)
                 for i, x in enumerate((-0.25, 0.0, 0.25)))
    return ObjectScene(0.0, objs)


def _eef():
    left = Pose((-0.3, -0.1, 1.2), (1.0, 0.0, 0.0, 0.0), WORKCELL_FRAME)
    right = Pose((0.4, -0.1, 1.2), (1.0, 0.0, 0.0, 0.0), WORKCELL_FRAME)
    return EEFState(ArmState(left, 1.0), ArmState(right, 1.0))


def test_effect_rows_are_goal_only_and_name_the_rules_object():
    from oct_vla.data.effects import yaw_token
    from oct_vla.tasks.shelf_restock.keyframes import ShelfEffectOracle
    from oct_vla.tasks.shelf_restock.spec import spec_for_rule

    scene = _scene()
    for rule, expected in (("leftmost", "obj_0"), ("rightmost", "obj_2")):
        oracle = ShelfEffectOracle(spec_for_rule(rule))
        rows, mask = oracle(scene, _eef())
        assert rows.shape == (FUTURE_ROWS, FUTURE_DIM)
        assert mask.tolist() == [True, False, True]
        assert oracle.inner.target == expected
        obj = scene.get(expected)
        np.testing.assert_allclose(rows[0, :ENTITY_TOKEN_DIM],
                                   yaw_token(obj.pose.position, obj.pose.orientation,
                                             obj.size_xyz), atol=1e-6)
        assert rows[0, ENTITY_TOKEN_DIM:].tolist() == [1, 0, 0, 0, 0, 0, 0]
        assert rows[2, ENTITY_TOKEN_DIM:].tolist() == [0, 0, 1, 0, 0, 0, 0]
        assert not rows[1].any()
        # The goal is on the upper shelf, not where the object is now.
        assert rows[2, 2] > 0.9 and abs(rows[2, 1] + 0.06) < 0.07


def test_the_server_offers_the_effect_sources():
    from oct_vla.serve.server import FUTURE_KEYFRAME_SOURCES

    assert {"oracle_effect", "oracle_effect:leftmost", "oracle_effect:rightmost",
            "oracle_effect:middle"} <= set(FUTURE_KEYFRAME_SOURCES)
