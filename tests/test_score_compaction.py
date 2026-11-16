"""Compaction counting (scripts/score_compaction.py) on synthetic transfers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "score_compaction", Path(__file__).resolve().parents[1] / "scripts" / "score_compaction.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

_fx = importlib.util.spec_from_file_location(
    "effects_fixtures", Path(__file__).resolve().parent / "data" / "test_effects.py")
_F = importlib.util.module_from_spec(_fx)
_fx.loader.exec_module(_F)
_transfer = _F._transfer


def _two(second_pushed: bool):
    """Two transfers: the first placed and never pushed, the second placed at frame
    offset 200 and (optionally) pushed 5 cm by the right arm."""
    p1, e1, a1, u1 = _transfer(pushed=False)
    p2, e2, a2, u2 = _transfer(pushed=second_pushed)
    T = len(p1)
    pos = np.zeros((2 * T, 2, 3))
    pos[:T, 0], pos[T:, 0] = p1, p1[-1]
    pos[:T, 1], pos[T:, 1] = p2[0], p2
    pos[:, 1, 0] += 0.4  # apart from the first object
    eef = np.concatenate([e1, e2])
    eef[T:] += (0.4, 0, 0)
    ap = np.concatenate([a1, a2])
    upper = np.stack([np.r_[u1, np.full(T, u1[-1])], np.r_[np.zeros(T, bool), u2]], 1)
    return pos, eef, ap, upper


def test_a_pushed_second_object_counts_and_an_unpushed_one_does_not():
    r = C.compaction(*_two(True))
    assert (r["placed"], r["compactable"], r["compacted"]) == (2, 1, 1)
    np.testing.assert_allclose(r["shift_cm"], [5.0], atol=0.4)
    r = C.compaction(*_two(False))
    assert (r["placed"], r["compactable"], r["compacted"], r["first_pushed"]) == (2, 1, 0, 0)


def test_summary_rate():
    per = {"a": C.compaction(*_two(True)), "b": C.compaction(*_two(False))}
    s = C.summarise(per)
    assert s["compaction_rate"] == 0.5 and s["episodes"] == 2
