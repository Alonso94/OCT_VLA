"""Packing of prior items, conflict pairs, evaluation interventions, token conversion."""

from __future__ import annotations

import numpy as np
import pytest

from oct_vla.prior import batches as B
from oct_vla.prior import interventions as I
from oct_vla.prior.schema import GRIPPER_TYPES, INITIAL, MOVABLE, from_entity_tokens


@pytest.fixture(scope="module")
def batch():
    return B.sample_batch(np.random.default_rng(7), B.ItemConfig(), 48)


def test_target_column_is_a_movable_and_its_features_match(batch):
    rows = np.arange(len(batch["t_obj"]))
    assert np.all(batch["q_types"][rows, batch["t_obj"]] == MOVABLE)
    # A grasp target leaves the object where it is in the current state.
    grasp = batch["t_event"] == 1
    cur = batch["q_feats"][rows, 1, batch["t_obj"], :3]
    np.testing.assert_allclose(
        cur[grasp & (batch["alpha"] == 0)],
        batch["t_obj_pose"][grasp & (batch["alpha"] == 0), :3],
        atol=1e-5,
    )


def test_context_rows_flag_the_acted_object_and_gripper(batch):
    for i in range(len(batch["t_obj"])):
        for d in np.flatnonzero(batch["demo_valid"][i]):
            types = batch["ctx_types"][i, d]
            for r in np.flatnonzero(batch["ctx_rows"][i, d]):
                flagged = np.flatnonzero(batch["ctx_flag"][i, d, r])
                if batch["ctx_event"][i, d, r] == INITIAL:
                    assert len(flagged) == 0
                    continue
                kinds = sorted(types[flagged].tolist())
                g = GRIPPER_TYPES[batch["ctx_gripper"][i, d, r]]
                assert kinds == sorted([MOVABLE, g])


def test_conflict_pairs_share_the_query_and_differ_in_the_target(batch):
    pairs = [(i, j) for i, j in enumerate(batch["pair"]) if j > i]
    assert pairs, "the default config makes conflict pairs"
    for i, j in pairs:
        assert batch["program"][i] != batch["program"][j]
        # Same state up to the column permutation: compare as multisets of rows.
        qa = np.sort(batch["q_feats"][i].reshape(-1, 9), axis=0)
        qb = np.sort(batch["q_feats"][j].reshape(-1, 9), axis=0)
        np.testing.assert_allclose(qa, qb, atol=1e-6)
        ta = (
            batch["q_feats"][i, 1, batch["t_obj"][i]],
            batch["t_grip"][i],
            batch["t_event"][i],
            batch["t_obj_pose"][i, :3],
        )
        tb = (
            batch["q_feats"][j, 1, batch["t_obj"][j]],
            batch["t_grip"][j],
            batch["t_event"][j],
            batch["t_obj_pose"][j, :3],
        )
        same = (
            np.allclose(ta[0], tb[0])
            and ta[1] == tb[1]
            and ta[2] == tb[2]
            and np.linalg.norm(ta[3] - tb[3]) <= B.DIFFERENT_POSITION
        )
        assert not same


def test_language_index_matches_kind(batch):
    assert np.array_equal(batch["lang_index"] < 0, batch["lang_kind"] == B.LANG_NONE)


def test_interventions(batch):
    rng = np.random.default_rng(0)
    shuffled, m = I.support(batch, "shuffled", rng)
    changed = np.flatnonzero(m)
    assert len(changed) > len(batch["t_obj"]) // 2
    none, _ = I.support(batch, "none", rng)
    assert not none["demo_valid"].any() and (none["ctx_types"] == -1).all()
    conf, m = I.support(batch, "conflicting", rng)
    for i in np.flatnonzero(m):
        np.testing.assert_array_equal(conf["ctx_feats"][i], batch["ctx_feats"][batch["pair"][i]])
    partner = I.partner_targets(batch)
    for i in np.flatnonzero(batch["pair"] >= 0):
        # Columns are permuted per item: the mapped column must hold the *same entity*
        # as the partner's target, not the same index.
        j = batch["pair"][i]
        np.testing.assert_allclose(
            batch["q_feats"][i, :, partner["t_obj"][i]], batch["q_feats"][j, :, batch["t_obj"][j]]
        )
        assert partner["t_event"][i] == batch["t_event"][j]
    exp = I.language(batch, "explicit", rng)
    assert (exp["lang_kind"] == B.LANG_EXPLICIT).all() and (exp["lang_index"] >= 0).all()


def test_from_entity_tokens_reads_schema_v2():
    from oct_vla.core.frames import WORKCELL_FRAME, Pose
    from oct_vla.core.geometry import exp
    from oct_vla.data.entity_tokens import EntitySupport, build_entity_tokens

    class Obj:
        def __init__(self, pose, size):
            self.pose, self.size_xyz = pose, size

    class Scene:
        objects = [
            Obj(Pose((0.1, -0.2, 0.78), exp((0.0, 0.0, 0.3)), WORKCELL_FRAME), (0.04, 0.05, 0.06))
        ]

    class Arm:
        def __init__(self, x):
            self.pose = Pose((x, -0.1, 1.0), exp((3.14159, 0.0, 0.0)), WORKCELL_FRAME)
            self.gripper = 0.7

    class EEF:
        left, right = Arm(-0.3), Arm(0.3)

    sup = EntitySupport(
        Pose((0.0, -0.25, 0.74), (0.0, 0.0, 0.0, 1.0), WORKCELL_FRAME), (1.0, 0.2, 0.02)
    )
    tokens, mask = build_entity_tokens(Scene(), EEF(), (sup,), max_entities=8)
    feats, types = from_entity_tokens(np.array(tokens), np.array(mask))
    assert types.tolist()[:4] == [0, 1, 2, 3] and (types[4:] == -1).all()
    assert np.arctan2(feats[0, 3], feats[0, 4]) == pytest.approx(0.3)
    np.testing.assert_allclose(feats[0, 5:8], (0.04, 0.05, 0.06))
    assert feats[1, 8] == pytest.approx(0.7) and feats[1, 3] == 0.0 and feats[1, 4] == 1.0
