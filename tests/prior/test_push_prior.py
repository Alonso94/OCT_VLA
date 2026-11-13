"""The v2 push prior (programs_v2, generator_v2) and its time-row data."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from oct_vla.prior import programs as P
from oct_vla.prior import programs_v2 as P2
from oct_vla.prior import time_row_schema as S
from oct_vla.prior.batches import ItemConfig
from oct_vla.prior.generator import Infeasible, execute, sample_scene
from oct_vla.prior.generator_v2 import (
    COMPACT_GAP,
    CONTACT,
    FINGER,
    PUSH,
    ROT,
    execute_v2,
    scene_for_v2,
)
from oct_vla.prior.schema import N_FEATURES, PLACE, POS
from oct_vla.prior.time_row_batches import TimeRowConfig, row_labels, sample_time_row_batch


def _episodes(program, count=8, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        try:
            out.append(scene_for_v2(rng, [program])[0])
        except Infeasible:
            continue
    return out


COMPACTING = [P2.ProgramV2("left_to_right", "higher", arr, arm, "toward_neighbour")
              for arr in ("row_left", "row_right") for arm in ("left", "nearest", "handover")]


def test_v1_programs_are_untouched_and_v2_is_separate():
    assert len(P.ALL_PROGRAMS) == 480
    assert len(P2.ALL_PROGRAMS_V2) == 480 + 320  # rows get a compacting twin; stacks do not
    with pytest.raises(ValueError, match="stack"):
        P2.ProgramV2("left_to_right", "higher", "stack", "left", "toward_neighbour")
    assert not set(P2.TRAIN_PROGRAMS_V2) & (P2.SHELF_FAMILY_V2 | P2.COMPOSITION_HOLDOUT_V2)
    shelf = P2.ProgramV2("left_to_right", "higher", "row_left", "left", "toward_neighbour")
    assert shelf in P2.SHELF_FAMILY_V2 and shelf not in P2.TRAIN_PROGRAMS_V2
    compacted = P2.ProgramV2("front_to_back", "lower", "row_left", "right", "toward_neighbour")
    assert compacted in P2.TRAIN_PROGRAMS_V2  # the push mechanism itself is trained
    assert compacted.instructions()[0].endswith("push each one against the previous.")


def test_without_compaction_v2_only_adds_columns():
    for arm in P.ARMS:
        prog = P2.ProgramV2("left_to_right", "higher", "row_left", arm)
        scene = None
        for seed in range(30):
            try:
                scene = sample_scene(np.random.default_rng(seed), n=3)
                v1 = execute(scene, prog.v1, np.random.default_rng(1))
                break
            except Infeasible:
                continue
        v2 = execute_v2(scene, prog, np.random.default_rng(1))
        assert len(v1.keyframes) == len(v2.keyframes)
        for a, b in zip(v1.keyframes, v2.keyframes):
            np.testing.assert_array_equal(a.feats, b.feats[:, :N_FEATURES])
            assert (a.event, a.gripper, a.obj) == (b.event, b.gripper, b.obj)


@pytest.mark.parametrize("program", COMPACTING, ids=lambda p: f"{p.arrangement}-{p.arm}")
def test_a_push_compacts_the_row_and_moves_nothing_else(program):
    pushes = 0
    for ep in _episodes(program, 6):
        n, kfs = ep.scene.n, ep.keyframes
        for j, kf in enumerate(kfs):
            if kf.event != PUSH:
                continue
            pushes += 1
            contact, place = kfs[j - 1], kfs[j - 2]
            assert contact.event == CONTACT and place.event == PLACE
            o, c = kf.obj, kf.gripper
            assert c == 1 - place.gripper and contact.gripper == c  # the other arm compacts
            moved = [i for i in range(n) if not np.allclose(kf.feats[i, POS], contact.feats[i, POS])]
            assert moved == [o]
            # Compacted against the nearest already-placed object along the row.
            others = [i for i in range(n) if i != o and abs(kf.feats[i, 1] - kf.feats[o, 1]) < 1e-6
                      and abs(kf.feats[i, 2] - kf.feats[o, 2]) < 0.05]
            gaps = [abs(kf.feats[o, 0] - kf.feats[i, 0]) - (ep.scene.obj_size[o][0]
                    + ep.scene.obj_size[i][0]) / 2 for i in others]
            assert min(gaps) == pytest.approx(COMPACT_GAP, abs=1e-6)
            # The compactor: closed, horizontal, at the object's far face.
            rot = contact.feats[n + c, ROT]
            assert abs(abs(rot[0]) - 1) < 1e-9 and abs(rot[2]) < 1e-9
            assert contact.feats[n + c, 8] == 0.0
            face = abs(contact.feats[n + c, 0] - contact.feats[o, 0])
            assert face == pytest.approx(ep.scene.obj_size[o][0] / 2 + FINGER)
            # Later keyframes keep the compacted pose; the final pose is updated.
            for later in kfs[j + 1:]:
                np.testing.assert_allclose(later.feats[o, POS], kf.feats[o, POS])
            np.testing.assert_allclose(ep.final_pose[o][:3], kf.feats[o, POS])
    assert pushes > 0


def test_push_orientations_vary_and_grasps_are_top_down():
    closings = set()
    for ep in _episodes(COMPACTING[0], 12, seed=3):
        n = ep.scene.n
        for kf in ep.keyframes:
            rot = kf.feats[n + kf.gripper, ROT] if kf.gripper >= 0 else None
            if kf.event == PUSH:
                closings.add(tuple(np.round(rot[3:], 6)))
            elif kf.event == PLACE:
                np.testing.assert_allclose(rot[:3], (0, 0, -1))
    assert len(closings) == 2  # horizontal and vertical closing axes both occur


def test_time_row_v2_labels_carry_push_events_and_rotations():
    cfg = TimeRowConfig(item=ItemConfig(programs=tuple(COMPACTING)))
    assert cfg.v2 and cfg.n_features == N_FEATURES + 6 and cfg.schema == S.SCHEMA_VERSION_V2
    ep = _episodes(COMPACTING[1], 1, seed=5)[0]
    n = ep.scene.n
    seen = set()
    for j in range(1, len(ep.keyframes)):
        labels = row_labels(ep, j)
        kf = ep.keyframes[j]
        seen.add(kf.event)
        assert labels["y_participation"][kf.gripper] == 1
        for a in (0, 1):
            np.testing.assert_allclose(labels["y_rot"][a], kf.feats[n + a, ROT])
    assert {CONTACT, PUSH} <= seen
    assert S.ARM_EVENTS_V2[PUSH] == "push" and S.ARM_EVENTS_V2[CONTACT] == "contact"


def test_compaction_conflict_pairs_need_the_support():
    """Partners that differ only in `compact`: same query, different next keyframe."""
    progs = (COMPACTING[0], dataclasses.replace(COMPACTING[0], compact="none"))
    cfg = TimeRowConfig(item=dataclasses.replace(ItemConfig(programs=progs), conflict_fraction=1.0))
    inputs, targets, metas = sample_time_row_batch(np.random.default_rng(2), cfg, 8)
    pairs = [(i, m["pair"]) for i, m in enumerate(metas) if m["pair"] > i]
    assert pairs
    for i, j in pairs:
        np.testing.assert_array_equal(inputs["q_feats"][i], inputs["q_feats"][j])
        assert {targets["g_event"][i], targets["g_event"][j]} & {CONTACT}
