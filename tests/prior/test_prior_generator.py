"""The interaction prior's generator: physical invariants, program semantics, splits."""

from __future__ import annotations

import numpy as np
import pytest

from oct_vla.prior import generator as G
from oct_vla.prior import programs as P
from oct_vla.prior.schema import GRASP, MAX_OPENING, PLACE, POS, REGRASP, RELEASE


def _episodes(program, count=12, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        try:
            out.append(G.execute(G.sample_scene(rng, n=int(rng.integers(2, 5))), program, rng))
        except G.Infeasible:
            pass
    return out


PROGRAMS = [
    P.Program("left_to_right", "higher", "row_left", "nearest"),
    P.Program("large_to_small", "lower", "stack", "right"),
    P.Program("back_to_front", "left_side", "row_right", "handover"),
]


@pytest.mark.parametrize("program", PROGRAMS, ids=lambda p: p.arm)
def test_held_object_keeps_its_offset_to_the_holding_gripper(program):
    for ep in _episodes(program):
        n = ep.scene.n
        held: dict[int, tuple[int, np.ndarray]] = {}
        for kf in ep.keyframes[1:]:
            obj, grip = kf.feats[kf.obj, POS], kf.feats[n + kf.gripper, POS]
            if kf.event in (GRASP, REGRASP):
                held[kf.obj] = (kf.gripper, grip - obj)
            elif kf.event in (PLACE,):
                g, offset = held.pop(kf.obj)
                assert g == kf.gripper
                np.testing.assert_allclose(grip - obj, offset, atol=1e-9)
        assert not held, "every grasped object is placed"


@pytest.mark.parametrize("program", PROGRAMS, ids=lambda p: p.arrangement)
def test_placements_rest_on_the_destination_without_overlap(program):
    for ep in _episodes(program):
        s, last = ep.scene, ep.keyframes[-1].feats
        dest = G.destination(s, program.destination)
        top = s.sup_top(dest)
        boxes = []
        for i in range(s.n):
            p, size = last[i, POS], s.obj_size[i]
            assert abs(p[0] - s.sup_pos[dest, 0]) <= s.sup_size[dest, 0] / 2 + 1e-9
            assert abs(p[1] - s.sup_pos[dest, 1]) <= s.sup_size[dest, 1] / 2 + 1e-9
            bottom = p[2] - size[2] / 2
            assert bottom >= top - 1e-9
            boxes.append((p, size))
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                (pa, sa), (pb, sb) = boxes[a], boxes[b]
                overlap_xy = np.all(np.abs(pa[:2] - pb[:2]) < (sa[:2] + sb[:2]) / 2 - 1e-9)
                overlap_z = abs(pa[2] - pb[2]) < (sa[2] + sb[2]) / 2 - 1e-9
                assert not (overlap_xy and overlap_z), "placed objects interpenetrate"


def test_grasp_closes_to_the_object_width_and_place_opens():
    for ep in _episodes(PROGRAMS[0]):
        n = ep.scene.n
        for kf in ep.keyframes[1:]:
            ap = kf.feats[n + kf.gripper, 8]
            if kf.event == GRASP:
                size = ep.scene.obj_size[kf.obj]
                assert ap == pytest.approx(min(size[0], size[1]) / MAX_OPENING)
                assert ap < 1.0
            if kf.event in (PLACE, RELEASE):
                assert ap == 1.0


def test_selector_order_is_what_the_program_says():
    for ep in _episodes(PROGRAMS[0]):
        x0 = ep.keyframes[0].feats[: ep.scene.n, 0]
        grasped = [kf.obj for kf in ep.keyframes if kf.event == GRASP]
        assert list(np.argsort(x0)) == grasped


def test_handover_has_regrasp_then_release_by_the_other_arm():
    for ep in _episodes(PROGRAMS[2]):
        events = [(kf.event, kf.gripper) for kf in ep.keyframes[1:]]
        for k in range(0, len(events), 4):
            (e0, g0), (e1, g1), (e2, g2), (e3, g3) = events[k : k + 4]
            assert (e0, e1, e2, e3) == (GRASP, REGRASP, RELEASE, PLACE)
            assert g1 == 1 - g0 and g2 == g0 and g3 == g1


def test_interpolation_keeps_a_carried_object_with_its_gripper():
    ep = _episodes(PROGRAMS[0], count=1)[0]
    n = ep.scene.n
    a, b = ep.keyframes[1], ep.keyframes[2]  # grasp, then place of the same object
    assert (a.event, b.event, a.obj) == (GRASP, PLACE, b.obj)
    mid = G.interpolate(a, b, 0.4)
    off_a = a.feats[n + a.gripper, POS] - a.feats[a.obj, POS]
    np.testing.assert_allclose(mid[n + a.gripper, POS] - mid[a.obj, POS], off_a, atol=1e-9)
    assert mid[n + a.gripper, 8] == a.feats[n + a.gripper, 8], "aperture changes only at the event"


def test_shelf_family_and_compositions_are_never_trained_on():
    train = set(P.TRAIN_PROGRAMS)
    assert not train & P.SHELF_FAMILY
    assert not train & P.COMPOSITION_HOLDOUT
    shelf_rule = P.Program("left_to_right", "higher", "row_left", "left")
    assert shelf_rule in P.SHELF_FAMILY
    # Every component of a held-out composition is trained with other partners.
    for p in P.COMPOSITION_HOLDOUT:
        assert any(q.selector == p.selector for q in train)
        assert any(q.arrangement == p.arrangement for q in train)
        assert any(q.arm == p.arm for q in train)


def test_instructions_are_distinct_per_program():
    texts = [p.instructions()[0] for p in P.ALL_PROGRAMS]
    assert len(set(texts)) == len(texts)
