"""Placement-loss attribution on hand-built traces (scripts/diagnose_placement_loss.py)."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "diagnose_placement_loss", Path(__file__).parents[1] / "scripts/diagnose_placement_loss.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC  # noqa: E402

CONTAINS = DEFAULT_SPEC.upper_shelf.contains
SHELF = np.array([-0.06, -0.06, 0.97])  # resting on the upper deck
FLOOR = np.array([-0.06, -0.30, 0.20])
FAR = np.array([0.6, -0.4, 1.3])


def _trace(steps, obj_path, left_path, right_path, other=None):
    objects = np.zeros((steps, 1 if other is None else 2, 3))
    objects[:, 0] = obj_path
    if other is not None:
        objects[:, 1] = other
    eef = np.stack([left_path, right_path], 1)
    return objects, eef


def _path(*segments):
    return np.concatenate([np.repeat(np.asarray(p)[None], n, 0) for p, n in segments])


def test_never_released_while_held_in_the_band():
    obj = _path((SHELF, 5), (FLOOR, 5))
    left = _path((SHELF + [0, 0, 0.14], 5), (FLOOR + [0, 0, 0.14], 5))
    case, _ = D.classify(*_trace(10, obj, left, _path((FAR, 10))), 0, CONTAINS)
    assert case == "a_never_released"


def test_fell_with_no_arm_near():
    obj = _path((SHELF, 6), (FLOOR, 4))
    case, info = D.classify(*_trace(10, obj, _path((FAR, 10)), _path((FAR, 10))), 0, CONTAINS)
    assert case == "b_fell_after_release" and info["t_rest"] == 0


def test_pushed_off_by_the_compactor():
    obj = _path((SHELF, 6), (FLOOR, 4))
    right = _path((FAR, 4), (SHELF + [0.1, 0, 0.05], 6))
    case, info = D.classify(*_trace(10, obj, _path((FAR, 10)), right), 0, CONTAINS)
    assert case == "c_compactor" and info["nearest_arm"] == "right"


def test_disturbed_by_the_mover_carrying_another_object():
    obj = _path((SHELF, 6), (FLOOR, 4))
    left = _path((FAR, 4), (SHELF + [0.08, 0, 0.12], 6))
    carried = left - [0, 0, 0.14]
    case, info = D.classify(*_trace(10, obj, left, _path((FAR, 10)), other=carried), 0, CONTAINS)
    assert case == "d_mover_carrying" and info["mover_carrying"]


def test_lift_order_follows_lifted_step():
    events = {"a": {"lifted_step": 30}, "b": {"lifted_step": 5}, "c": {"lifted_step": None}}
    assert D.lift_order(events) == ["b", "a"]
