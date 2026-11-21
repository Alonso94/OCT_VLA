"""First-slice acceptance (plan stage 0): the colour-swap and delayed-handover fixtures,
the schema operations they rely on, and the quaternion helpers against core.geometry."""

import numpy as np
import pytest

from oct_vla.core import geometry as G
from oct_vla.bpfn.data.syn_data import branches
from oct_vla.prior.pose_window import fixtures as F
from oct_vla.bpfn import quat
from oct_vla.bpfn import schema
def test_quaternions_match_core_geometry():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = quat.normalize(rng.normal(size=4)), quat.normalize(rng.normal(size=4))
        v = rng.normal(size=3)
        np.testing.assert_allclose(quat.multiply(a, b), np.array(G.multiply(a, b)) * np.sign(
            quat.multiply(a, b) @ np.array(G.multiply(a, b))), atol=1e-9)
        np.testing.assert_allclose(quat.rotate(a, v), G.rotate(a, v), atol=1e-9)
        np.testing.assert_allclose(quat.rotate(quat.from_matrix(quat.to_matrix(a)), v), quat.rotate(a, v), atol=1e-9)
    r6 = quat.to_matrix(a)[:, :2].T.reshape(-1)
    np.testing.assert_allclose(quat.angle(quat.from_6d(r6), a), 0.0, atol=1e-6)
    track = quat.continuous(np.stack([a, -a, a]))
    assert np.all((track[1:] * track[:-1]).sum(-1) > 0)
    assert quat.angle(a, -a) == pytest.approx(0.0, abs=1e-6)


def test_target_window_slices_once_and_masks_the_end_and_the_cue():
    ep = branches.handover("nominal")
    w, v = schema.target_window(ep, 10, horizon=8)
    np.testing.assert_allclose(w[0], ep.wrist[11], atol=1e-6)  # t+1, never t or t+2
    np.testing.assert_allclose(w[-1], ep.wrist[18], atol=1e-6)
    assert v.all()
    w, v = schema.target_window(ep, ep.T - 3, horizon=8)
    assert v.tolist() == [True, True] + [False] * 6
    _, v = schema.target_window(ep, 10, horizon=8, cue=14)
    assert v.tolist() == [True, True, True] + [False] * 5   # rows 11..13 only


def test_query_never_contains_rows_after_t_and_masks_goals():
    ep = branches.handover("nominal")
    q = schema.query_table(ep, 20, history=16)
    assert q.time[-1] == pytest.approx(ep.time[20]) and len(q.time) == 16
    assert not q.goal_valid.any() and not q.goal.any()
    s = schema.support_table(ep, rows=np.array([0, 5, 9]))
    np.testing.assert_allclose(s.goal[0], ep.wrist[5], atol=1e-6)   # the next *selected* row
    assert s.goal_valid.tolist() == [[True, True], [True, True], [False, False]]


def test_column_shuffle_carries_headers_and_relabel_keeps_reserved_ids():
    ep = F.restock_episode([-0.2, 0.0, 0.2], ["red", "blue", "red"], "red")
    t = schema.support_table(ep)
    ts, perm = schema.shuffle_columns(t, np.random.default_rng(3))
    for new, old in enumerate(perm):
        assert ts.tracks[new] == t.tracks[old] and ts.colour[new] == t.colour[old]
        np.testing.assert_array_equal(ts.pose[:, new], t.pose[:, old])
    m = schema.semantic_relabeling(np.random.default_rng(4))
    r = schema.relabel(t, m)
    assert m["colour"][schema.UNKNOWN] == schema.UNKNOWN
    np.testing.assert_array_equal(r.kind, t.kind)
    assert r.colour[t.column("left")] == schema.UNKNOWN
    assert "tracks" not in t.model_inputs()


def test_episode_refuses_non_unit_quaternions():
    ep = branches.handover("nominal")
    bad = ep.wrist.copy()
    bad[0, 0, 3:] = 0.3
    with pytest.raises(ValueError, match="non-unit"):
        schema.Episode(ep.time, bad, ep.aperture, ep.objects, ep.kind, ep.colour, ep.category,
                  ep.size, ep.track_ids)


@pytest.mark.parametrize("seed", [0, 1])
def test_colour_fixture_acceptance(seed):
    r = F.colour_fixture(seed=seed)
    assert all(r["checks"].values()), r["checks"]
    assert r["sets"]["red"] == {"obj_1", "obj_2"} and r["sets"]["blue"] == {"obj_0", "obj_3"}
    for ep in r["supports"].values():
        assert ep.annotations["physical"] == 4 and ep.annotations["eligible_count"] == 3


def test_colour_reader_refuses_an_uninformative_support():
    """All objects moved: no colour is singled out, so the reader must not guess."""
    ep = F.restock_episode([-0.2, 0.1], ["red", "red"], "red")
    assert F.demonstrated_colour(schema.support_table(ep)) == schema.colour_id("red")
    mixed = F.restock_episode([-0.2, 0.1], ["red", "blue"], "red")
    assert F.demonstrated_colour(schema.support_table(mixed)) == schema.colour_id("red")
    # A support whose moved set has two colours identifies nothing.
    both = schema.support_table(F.restock_episode([-0.2, 0.1], ["red", "red"], "red"))
    both.colour[both.column("obj_1")] = schema.colour_id("blue")
    assert F.demonstrated_colour(both) is None


@pytest.mark.parametrize("giver", [0, 1])
def test_handover_fixture_acceptance(giver):
    eps = branches.family(giver=giver)
    r = branches.handover_checks(eps)
    assert all(r["checks"].values()), r["checks"]
    assert r["extra_hold_s"]["delayed"] > 0.5
    # The giver's hold is a decision on the receiver's state, not a schedule: the
    # receiver is at its grasp pose and closed at the giver's release.
    for ep in eps.values():
        assert ep.aperture[ep.annotations["release_row"], 1 - giver] < branches.CLOSED_BELOW


def test_handover_branches_differ_only_after_the_cue():
    eps = branches.family()
    cue = eps["delayed"].annotations["cue"]
    np.testing.assert_array_equal(eps["nominal"].wrist[:cue], eps["delayed"].wrist[:cue])
    assert not np.array_equal(eps["nominal"].wrist[cue], eps["delayed"].wrist[cue])
