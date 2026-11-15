"""Scoring of the time-row pilot (scripts/time_row_prior_eval.py): per event, compaction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from oct_vla.prior import time_row_schema as S
from oct_vla.prior.programs_v2 import ALL_PROGRAMS_V2

_spec = importlib.util.spec_from_file_location(
    "time_row_prior_eval", Path(__file__).resolve().parents[2] / "scripts/time_row_prior_eval.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)


def _twins():
    """Two programs differing only in `compact`, and one differing in more."""
    by_v1 = {}
    for i, p in enumerate(ALL_PROGRAMS_V2):
        by_v1.setdefault(p.v1, []).append(i)
    a, b = next(v for v in by_v1.values() if len(v) == 2)
    other = next(i for i, p in enumerate(ALL_PROGRAMS_V2) if p.v1 != ALL_PROGRAMS_V2[a].v1)
    return a, b, other


def _case():
    a, b, other = _twins()
    push = S.GROUP_EVENTS_V2.index("push")
    grasp = S.GROUP_EVENTS_V2.index("grasp")
    compact_first = ALL_PROGRAMS_V2[a].compact != "none"
    g_event = np.array([push if compact_first else grasp, grasp if compact_first else push,
                        grasp, grasp])
    targets = {"target_valid": np.ones(4), "g_event": g_event, "g_object": np.zeros(4, int),
               "y_participation": np.tile([1.0, 0.0], (4, 1)),
               "y_event": np.stack([g_event, np.zeros(4, int)], 1),
               "y_pos": np.zeros((4, 2, 3)), "g_goal": np.zeros((4, 5)),
               "y_rot": np.tile([0, 0, -1, 1, 0, 0], (4, 2, 1)).astype(float)}
    pred = {"group_event": g_event.copy(), "group_pointer": np.zeros(4, int),
            "participation": targets["y_participation"].copy(),
            "arm_event": targets["y_event"].copy(), "pos": np.zeros((4, 2, 3)),
            "goal": np.zeros((4, 5)), "rot": targets["y_rot"].copy()}
    metas = [{"program": a, "pair": 1}, {"program": b, "pair": 0},
             {"program": a, "pair": 3}, {"program": other, "pair": 2}]
    return pred, targets, metas


def test_only_compact_twins_count_as_compaction_pairs():
    pred, targets, metas = _case()
    out = E.score(pred, targets, metas)
    assert out["contrast_pairs"] == 2 and out["compact_pairs"] == 1
    assert out["compact_contrast"] == 1.0 and out["push_recall"] == 1.0
    assert out["event/push/n"] == 1 and out["event/grasp/recall"] == 1.0


def test_a_prior_that_ignores_compaction_fails_the_twin_pair():
    pred, targets, metas = _case()
    pred["group_event"][:] = S.GROUP_EVENTS_V2.index("grasp")
    out = E.score(pred, targets, metas)
    assert out["compact_contrast"] == 0.0 and out["push_recall"] == 0.0
    assert out["false_push_rate"] == 0.0
    pred["group_event"][:] = S.GROUP_EVENTS_V2.index("push")
    assert E.score(pred, targets, metas)["false_push_rate"] == 1.0


def test_position_error_is_split_by_the_acting_arms_pointer():
    pred, targets, metas = _case()
    targets["y_object"] = np.array([[1, -1]] * 4)
    pred["arm_pointer"] = np.array([[1, 0], [1, 0], [0, 0], [0, 0]])
    pred["pos"][2:, 0, 0] = 0.05  # the two wrong-pointer items are 5 cm off
    pred["anchor_max"] = np.array([[1.0, 1.0], [0.5, 1.0], [1.0, 1.0], [1.0, 1.0]])
    out = E.score(pred, targets, metas)
    assert out["n_ptr_ok"] == 2 and out["n_ptr_wrong"] == 2
    assert out["acting_pos_cm_ptr_ok"] == 0.0
    assert out["acting_pos_cm_ptr_wrong"] == pytest.approx(5.0)
    assert out["anchor_blended"] == 0.25
