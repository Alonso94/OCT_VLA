"""The time-row data contract on the unchanged v1 generator (prior/time_row_batches.py)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from oct_vla.prior import programs as P
from oct_vla.prior import time_row_schema as S
from oct_vla.prior.batches import ItemConfig, _differ, _target
from oct_vla.prior.generator import K_MAX, N_MAX, sample_episode
from oct_vla.prior.schema import APERTURE, GRASP, OPEN, PLACE, POS, REGRASP, RELEASE
from oct_vla.prior.time_row_batches import (
    TimeRowConfig,
    build_time_row_eval_set,
    episode_rows,
    load_time_row_eval_set,
    pack_time_row_item,
    row_labels,
    sample_time_row_batch,
)

ARM_PROGRAMS = {arm: P.Program("left_to_right", "higher", "row_left", arm) for arm in P.ARMS}


def _episodes(program, count=10, seed=0):
    rng = np.random.default_rng(seed)
    return [sample_episode(rng, program) for _ in range(count)]


def test_schema_fields_match_what_the_converter_writes():
    ep = _episodes(ARM_PROGRAMS["left"], 1)[0]
    labels = row_labels(ep, 1)
    assert set(labels) == set(S.ROW_LABELS)
    for k, shape in S.ROW_LABELS.items():
        assert np.shape(labels[k]) == shape, k
    assert S.ARM_EVENTS[GRASP] == "grasp" and S.ARM_EVENTS[RELEASE] == "release"


@pytest.mark.parametrize("arm", P.ARMS)
def test_every_label_is_the_generators_next_keyframe(arm):
    for ep in _episodes(ARM_PROGRAMS[arm]):
        n = ep.scene.n
        for row in episode_rows(ep, 1, np.random.default_rng(1)):
            labels = row_labels(ep, row["next"])
            if row["next"] is None:
                assert labels["target_valid"] == 0 and labels["g_event"] == S.NO_EVENT
                continue
            kf = ep.keyframes[row["next"]]
            assert labels["target_valid"] == 1
            assert labels["g_event"] == kf.event and labels["g_object"] == kf.obj
            assert labels["y_participation"].tolist() == [float(a == kf.gripper) for a in (0, 1)]
            for a in (0, 1):
                np.testing.assert_allclose(labels["y_pos"][a], kf.feats[n + a, POS], atol=1e-6)
                assert labels["y_aperture"][a] == pytest.approx(kf.feats[n + a, APERTURE])
                assert labels["y_engaged"][a] == float(kf.feats[n + a, APERTURE] < OPEN - 1e-9)
                if a == kf.gripper:
                    assert labels["y_event"][a] == kf.event and labels["y_object"][a] == kf.obj
                else:
                    assert labels["y_event"][a] == S.HOLD and labels["y_object"][a] == -1
            fp = ep.final_pose[kf.obj]
            np.testing.assert_allclose(labels["g_goal"][:3], fp[:3], atol=1e-6)


def test_single_arm_programs_never_mark_the_other_arm():
    for arm, other in (("left", 1), ("right", 0)):
        for ep in _episodes(ARM_PROGRAMS[arm], 6):
            for j in range(1, len(ep.keyframes)):
                assert row_labels(ep, j)["y_participation"][other] == 0


def test_handover_roles_regrasp_by_the_holder_release_by_the_giver():
    for ep in _episodes(ARM_PROGRAMS["handover"], 8):
        kfs = ep.keyframes
        for j in range(1, len(kfs)):
            if kfs[j].event != REGRASP:
                continue
            giver, holder = kfs[j - 1].gripper, kfs[j].gripper
            assert kfs[j - 1].event == GRASP and holder == 1 - giver
            regrasp, release = row_labels(ep, j), row_labels(ep, j + 1)
            assert regrasp["y_participation"][holder] == 1
            # Both arms hold the object at the regrasp: the giver is engaged, not acting.
            assert regrasp["y_engaged"].tolist() == [1.0, 1.0]
            assert kfs[j + 1].event == RELEASE and release["y_participation"][giver] == 1
            assert release["y_engaged"][holder] == 1 and release["y_engaged"][giver] == 0
            assert kfs[j + 2].event == PLACE and row_labels(ep, j + 2)["y_participation"][holder] == 1


def test_rows_are_labelled_with_the_next_keyframe_never_the_current_one():
    ep = _episodes(ARM_PROGRAMS["nearest"], 1, seed=3)[0]
    rows = episode_rows(ep, 2, np.random.default_rng(0))
    assert len(rows) == len(ep.keyframes) + 2 * (len(ep.keyframes) - 1)
    for row in rows:
        if row["next"] is not None:
            assert row["next"] == row["segment"] + 1
    # Interpolated rows of segment j share keyframe j's row label; times increase.
    times = [r["segment"] + r["alpha"] for r in rows]
    assert times == sorted(times)
    by_segment = {}
    for row in rows:
        by_segment.setdefault(row["segment"], set()).add(row["next"])
    assert all(len(v) == 1 for v in by_segment.values())
    assert rows[-1]["next"] is None


def _item(seed=0, perms=None, k=3, program=ARM_PROGRAMS["handover"]):
    rng = np.random.default_rng(seed)
    demos = [sample_episode(rng, program) for _ in range(k)]
    q = sample_episode(rng, program)
    return demos, q, pack_time_row_item(np.random.default_rng(99), program, demos, q, 1, 0.3, 1,
                                        TimeRowConfig(), perms)


def test_query_targets_never_enter_the_inputs():
    demos, q, (inputs, targets, meta) = _item()
    assert not set(inputs) & set(targets)
    assert not any(k.startswith("q_") and k[2:] in S.ROW_LABELS for k in inputs)
    assert "program" not in inputs and "j" not in inputs and "alpha" not in inputs
    for v in targets.values():
        for w in inputs.values():
            assert not np.shares_memory(np.asarray(v), np.asarray(w))
    # The query is a separate episode: its initial scene matches no support's.
    first = inputs["q_feats"][0]
    for d in range(len(demos)):
        support0 = inputs["s_feats"][d, 0]
        assert not np.allclose(np.sort(first[:, :3], 0), np.sort(support0[:, :3], 0))
    # The query's current scene precedes its target event.
    assert targets["target_valid"] == 1 and targets["g_event"] == q.keyframes[2].event


def test_object_pointers_follow_the_column_permutation():
    rng = np.random.default_rng(5)
    program = ARM_PROGRAMS["nearest"]
    demos = [sample_episode(rng, program) for _ in range(2)]
    q = sample_episode(rng, program)
    ident = [np.arange(len(e.types)) for e in demos + [q]]
    shuf = [np.random.default_rng(i).permutation(len(e.types)) for i, e in enumerate(demos + [q])]
    cfg = TimeRowConfig()
    a = pack_time_row_item(np.random.default_rng(7), program, demos, q, 0, 0.0, 0, cfg, ident)
    b = pack_time_row_item(np.random.default_rng(7), program, demos, q, 0, 0.0, 0, cfg, shuf)
    for d, perm in enumerate(shuf[:-1]):
        n = len(perm)
        np.testing.assert_array_equal(b[0]["s_feats"][d][:, :n], a[0]["s_feats"][d][:, perm])
        inv = np.argsort(perm)
        for key in ("s_y_object", "s_g_object"):
            x, y = a[0][key][d], b[0][key][d]
            mapped = np.where(x >= 0, inv[np.clip(x, 0, n - 1)], -1)
            np.testing.assert_array_equal(y, mapped)
        for key in ("s_y_pos", "s_y_participation", "s_g_goal", "s_g_event"):
            np.testing.assert_array_equal(a[0][key][d], b[0][key][d])
    qinv = np.argsort(shuf[-1])
    assert b[1]["g_object"] == qinv[a[1]["g_object"]]
    np.testing.assert_array_equal(a[1]["y_pos"], b[1]["y_pos"])


def test_masks_and_shapes_for_no_support_and_padding():
    rng = np.random.default_rng(2)
    program = ARM_PROGRAMS["left"]
    q = sample_episode(rng, program)
    inputs, targets, _ = pack_time_row_item(rng, program, [], q, 0, 0.0, 2, TimeRowConfig())
    assert not inputs["demo_valid"].any() and not inputs["s_row_valid"].any()
    assert inputs["s_feats"].shape == (K_MAX, TimeRowConfig().t_max, N_MAX, 9)
    assert (inputs["q_types"][len(q.types):] == -1).all()
    assert not set(S.RESERVED) & set(inputs)  # reserved fields are absent (masked) in v1
    _, _, (inputs, _, _) = _item(k=2)
    valid = inputs["s_row_valid"]
    assert valid[:2].any(1).all() and not valid[2:].any()
    assert (inputs["s_target_valid"][~valid] == 0).all()
    # Padding never looks like a pointer to column 0.
    assert (inputs["s_y_object"][~valid] == -1).all() and (inputs["s_g_object"][~valid] == -1).all()


def test_conflict_partners_share_the_query_and_differ_in_the_target():
    cfg = TimeRowConfig(item=dataclasses.replace(ItemConfig(), conflict_fraction=1.0))
    inputs, targets, metas = sample_time_row_batch(np.random.default_rng(4), cfg, 12)
    pairs = [(i, m["pair"]) for i, m in enumerate(metas) if m["pair"] > i]
    assert pairs
    for i, j in pairs:
        np.testing.assert_array_equal(inputs["q_feats"][i], inputs["q_feats"][j])
        np.testing.assert_array_equal(inputs["s_types"][i], inputs["s_types"][j])
        t = lambda x: {k: targets[k][x] for k in targets}  # noqa: E731
        differ = (t(i)["g_object"] != t(j)["g_object"] or t(i)["g_event"] != t(j)["g_event"]
                  or not np.array_equal(t(i)["y_participation"], t(j)["y_participation"])
                  or not np.allclose(t(i)["y_pos"], t(j)["y_pos"], atol=0.02))
        assert differ


def test_eval_packs_are_deterministic_and_never_overwritten(tmp_path):
    cfg = TimeRowConfig(item=ItemConfig(programs=P.program_split("shelf_family")))
    a = build_time_row_eval_set(tmp_path / "a", "t", "shelf_family", 20, chunk=10, workers=1, cfg=cfg)
    b = build_time_row_eval_set(tmp_path / "b", "t", "shelf_family", 20, chunk=10, workers=1, cfg=cfg)
    ia, ta, ma = load_time_row_eval_set(a)
    ib, tb, mb = load_time_row_eval_set(b)
    for k in ia:
        np.testing.assert_array_equal(ia[k], ib[k])
    for k in ta:
        np.testing.assert_array_equal(ta[k], tb[k])
    assert ma == mb and len(ma) == 20
    with pytest.raises(FileExistsError):
        build_time_row_eval_set(tmp_path / "a", "t", "shelf_family", 20, chunk=10, workers=1, cfg=cfg)
    assert all(P.ALL_PROGRAMS[m["program"]] in P.SHELF_FAMILY for m in ma)
    assert _differ is not None and _target is not None  # v1 helpers reused, not copied


# --- approach waypoints (TimeRowConfig.waypoints) ------------------------------------

def test_waypoint_labels_sit_above_the_acting_target_and_clear_the_others():
    from oct_vla.data.future_keyframes import WAYPOINT_CLEARANCE, WAYPOINT_LIFT
    from oct_vla.prior.schema import SIZE

    seen_raised = False
    for arm in P.ARMS:
        for ep in _episodes(ARM_PROGRAMS[arm], 5):
            n = ep.scene.n
            for j in range(len(ep.keyframes) - 1):
                lab = row_labels(ep, j + 1, waypoints=True)
                kf = ep.keyframes[j + 1]
                act, idle = kf.gripper, 1 - kf.gripper
                np.testing.assert_allclose(lab["y_wp"][idle], lab["y_pos"][idle])
                np.testing.assert_allclose(lab["y_wp"][act, :2], lab["y_pos"][act, :2])
                assert lab["y_wp"][act, 2] >= lab["y_pos"][act, 2] + WAYPOINT_LIFT - 1e-6
                for i in range(n):
                    if i == kf.obj:
                        continue
                    d = np.linalg.norm(kf.feats[i, POS][:2] - lab["y_pos"][act, :2])
                    if d <= 0.10 + 0.5 * kf.feats[i, SIZE][:2].max():
                        top = kf.feats[i, POS][2] + 0.5 * kf.feats[i, SIZE][2]
                        assert lab["y_wp"][act, 2] >= top + WAYPOINT_CLEARANCE - 1e-6
                        seen_raised |= lab["y_wp"][act, 2] > lab["y_pos"][act, 2] + WAYPOINT_LIFT + 1e-6
            assert row_labels(ep, None, waypoints=True)["y_wp"].tolist() == [[0.0] * 3] * 2
    assert seen_raised  # some neighbour actually raised a waypoint


def test_waypoint_config_adds_the_label_and_versions_the_schema():
    cfg = TimeRowConfig(item=ItemConfig(language=(1, 0, 0)), waypoints=True)
    inputs, targets, _ = sample_time_row_batch(np.random.default_rng(0), cfg, 3)
    assert cfg.schema.endswith(S.WAYPOINT_SUFFIX)
    assert inputs["s_y_wp"].shape[-2:] == (2, 3) and targets["y_wp"].shape == (3, 2, 3)
    plain = TimeRowConfig(item=ItemConfig(language=(1, 0, 0)))
    assert "y_wp" not in plain.labels and not plain.schema.endswith(S.WAYPOINT_SUFFIX)
