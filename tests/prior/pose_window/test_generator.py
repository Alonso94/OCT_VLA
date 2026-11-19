"""The analytic prior: programs, the geometric checker (with negative controls), the
observation process, splits, count matrices and the batch packer."""

import collections
import copy

import numpy as np
import pytest

from oct_vla.prior.pose_window import batches as Bt
from oct_vla.prior.pose_window import checker as C
from oct_vla.prior.pose_window import generator as Gn
from oct_vla.prior.pose_window import observation as O
from oct_vla.prior.pose_window import schema as S
from oct_vla.prior.pose_window import splits as Sp


def _episode(manip="carry_left", elig="all", order="left_to_right", repeat="until_empty",
             compaction="none", eligible=3, seed=0):
    p = Gn.Program(manip, elig, order, repeat, compaction)
    d = 0 if p.eligibility_kind == "all" else 1
    return Gn.sample_episode(p, eligible, d, np.random.default_rng(seed))


@pytest.mark.parametrize("manip,elig,compaction", [
    ("carry_left", "all", "none"), ("carry_nearest", "colour:red", "toward_neighbour"),
    ("carry_right", "category:can", "toward_neighbour"), ("handover", "all", "none"),
    ("handover", "category:box", "none"), ("push", "colour:blue", "none")])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_manipulation_passes_the_independent_checker(manip, elig, compaction, seed):
    ep = _episode(manip, elig, compaction=compaction, seed=seed)
    assert C.violations(ep) == 0, C.check(ep)


def test_checker_catches_each_violation():
    ep = _episode("carry_left", "colour:red")
    distractor = next(t for t in ep.track_ids[:ep.annotations["physical"]]
                      if t not in ep.annotations["selected"])
    j = ep.track_ids.index(distractor)
    bad = copy.deepcopy(ep)
    bad.objects[40:, j, 0] += 0.05                      # a distractor slides away untouched
    found = C.check(bad)
    assert found["distractor_moved"] and found["moved_without_contact"]
    bad = copy.deepcopy(ep)
    sel = ep.track_ids.index(ep.annotations["selected"][0])
    bad.objects[-1, sel, 2] += 0.02                     # a placed object floats
    assert C.check(bad)["not_resting"]
    bad = copy.deepcopy(ep)
    other = ep.track_ids.index(ep.annotations["selected"][1])
    bad.objects[-1, other, :2] = bad.objects[-1, sel, :2]   # two objects in one place
    assert C.check(bad)["overlap"]
    bad = copy.deepcopy(ep)
    bad.wrist[5:, 0, 3:] *= -1                          # a sign jump
    assert C.check(bad)["bad_quaternion"]


def test_selection_rules_and_distractors():
    ep = _episode("carry_left", "colour:red", eligible=3)
    colours = [S.COLOURS[c] for c in ep.colour[:ep.annotations["physical"]]]
    sel = ep.annotations["selected"]
    assert len(sel) == 3 and all(colours[ep.track_ids.index(t)] == "red" for t in sel)
    xs = [ep.objects[0, ep.track_ids.index(t), 0] for t in sel]
    assert xs == sorted(xs)                             # left to right
    fixed = _episode("carry_left", "all", repeat="fixed_2", eligible=4)
    assert len(fixed.annotations["selected"]) == 2 and fixed.annotations["eligible_count"] == 4


def test_generation_is_deterministic_per_seed():
    a, b = _episode(seed=5), _episode(seed=5)
    np.testing.assert_array_equal(a.wrist, b.wrist)
    assert not np.array_equal(a.wrist, _episode(seed=6).wrist)


def test_handover_branch_shares_a_prefix_and_delays_the_release():
    p = Gn.Program("handover", "all", "left_to_right", "fixed_2")
    rng = np.random.default_rng(3)
    scene = Gn.sample_scene(p, 2, 0, rng)
    nominal = Gn.execute(p, scene, copy.deepcopy(rng))
    delayed = Gn.execute(p, scene, copy.deepcopy(rng), perturb={"transfer": 0, "at": 0.3, "slow": 0.3})
    cue = Bt.first_divergence(nominal, delayed)
    assert 0 < cue < nominal.T
    np.testing.assert_array_equal(nominal.wrist[:cue], delayed.wrist[:cue])
    rel = lambda ep: next(e["row"] for e in ep.annotations["events"] if e["event"] == "transfer")  # noqa: E731
    assert rel(delayed) > rel(nominal)
    assert C.violations(delayed) == 0


def test_observation_faults_hold_the_last_measurement_and_never_touch_wrists():
    ep = _episode("handover", "all")
    pose, observed, age = O.observe(ep, np.random.default_rng(0), O.FaultConfig(rate=3.0))
    assert observed[:, :2].all() and not observed.all()
    t, c = map(int, np.argwhere(~observed)[0])
    last = max(r for r in range(t) if observed[r, c])
    np.testing.assert_array_equal(pose[t, c], pose[last, c])
    assert age[t, c] == pytest.approx(ep.time[t] - ep.time[last])
    clean = O.observe(ep, np.random.default_rng(0))
    assert clean[1].all() and not clean[2].any()


def test_splits_seal_the_held_out_composition_and_counts():
    train = Sp.programs_for("full")
    assert all(Sp.held_out_name(p) is None for p in train)
    held = Sp.heldout_programs()
    assert held and all(p.manipulation == "handover" and p.eligibility_kind == "colour" for p in held)
    assert {p.family for p in train} >= {("carry", "colour"), ("handover", "all"), ("handover", "category")}
    assert not any(p.manipulation == "handover" for p in Sp.programs_for("no_handover"))
    a = Sp.item_rng("train", 0).integers(1 << 30)
    assert a != Sp.item_rng("dev", 0).integers(1 << 30)
    rng = np.random.default_rng(0)
    equal = [Sp.CountMatrix("equal").draw(rng) for _ in range(200)]
    full = [Sp.CountMatrix("full").draw(rng) for _ in range(400)]
    assert all(s == q for s, q in equal)
    assert collections.Counter(s for s, _ in equal).keys() == collections.Counter(s for s, _ in full).keys()
    assert any(s != q for s, q in full)
    assert all(s == 3 for s, _ in (Sp.CountMatrix("eval3").draw(rng) for _ in range(50)))


def test_support_rows_use_only_observables_and_keep_contact_changes():
    ep = _episode("carry_left", "all", eligible=4)
    rows = Bt.support_rows(ep, budget=64)
    assert len(rows) <= 64 and rows[0] == 0 and rows[-1] == ep.T - 1
    grasps = [e["row"] for e in ep.annotations["events"] if e["event"] == "grasp"]
    # A grasp's aperture ramp ends at its event row; a kept row lies within the ramp.
    for g in grasps:
        assert np.any(np.abs(rows - g) <= 5)
    # Scrambling every annotation changes nothing the sampler sees.
    scrambled = copy.deepcopy(ep)
    scrambled.annotations = {"events": [], "selected": [], "program": "x"}
    np.testing.assert_array_equal(Bt.support_rows(scrambled, 64), rows)


def test_packed_inputs_ignore_prohibited_fields():
    """Rule, program, selection, events and phase live in meta; scrambling them leaves the
    packed model inputs bit-identical."""
    cfg = Bt.ItemConfig(programs=Sp.programs_for("full"), conflict_fraction=0.0)
    item = Bt.sample_item(np.random.default_rng(0), cfg)
    packed = Bt.pack([item])
    tampered = dict(item, meta={k: "scrambled" for k in item["meta"]})
    again = Bt.pack([tampered])
    assert all(np.array_equal(packed[k], again[k]) for k in packed)
    assert not any(k.startswith(("meta", "program", "rule", "phase", "selected")) for k in packed)


def test_conflict_pairs_share_query_inputs_and_differ_in_target():
    cfg = Bt.ItemConfig(programs=Sp.programs_for("full"))
    rng = np.random.default_rng(1)
    n = 0
    while n < 4:
        try:
            a, b = Bt.sample_pair(rng, cfg)
        except Gn.Infeasible:
            continue
        n += 1
        qa, qb = a["query"], b["query"]
        order = [qb.tracks.index(t) for t in qa.tracks]
        np.testing.assert_allclose(qa.pose, qb.pose[:, order])
        assert a["meta"]["first_pick"] != b["meta"]["first_pick"]
        assert a["meta"]["t"] < a["meta"]["diverge"]


def test_pack_shapes_and_query_never_carries_a_goal():
    cfg = Bt.ItemConfig(programs=Sp.programs_for("full"))
    items = Bt.sample_items(np.random.default_rng(2), cfg, 8)
    b = Bt.pack(items)
    assert b["s_pose"].shape[1:] == (Bt.S_ROWS, Bt.C_MAX, 7)
    assert b["q_pose"].shape[1:] == (Bt.HISTORY, Bt.C_MAX, 7)
    assert b["target"].shape[1:] == (Bt.HORIZON, 2, 7)
    assert not b["q_goal_valid"].any()
    np.testing.assert_allclose(np.linalg.norm(b["s_pose"][..., 3:], axis=-1), 1.0, atol=1e-5)
