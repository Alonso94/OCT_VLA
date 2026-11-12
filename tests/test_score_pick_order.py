"""Pick-order scoring for the PFN-VLA interventions (scripts/score_pick_order.py)."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "score_pick_order", Path(__file__).parents[1] / "scripts/score_pick_order.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

X = {"a": -0.3, "b": 0.0, "c": 0.25}


def _episode(lifts, initial_x=X):
    return {"initial_x": initial_x,
            "events": {t: {"lifted_step": lifts.get(t)} for t in initial_x}}


def test_rule_orders_match_the_managers_selectors():
    assert S.rule_order(X, "leftmost") == ["a", "b", "c"]
    assert S.rule_order(X, "rightmost") == ["c", "b", "a"]
    # middle, then leftmost of the two left (the manager's even-count fallback)
    assert S.rule_order(X, "middle") == ["b", "a", "c"]


def test_scoring_first_picks_and_full_orders():
    rightmost = _episode({"c": 10, "b": 90, "a": 200})
    partial = _episode({"a": 15, "b": 120})  # two lifts, leftmost so far
    never = _episode({})
    old = {"events": {"a": {"lifted_step": 3}}}  # recorded before initial_x
    r = S.score([rightmost, partial, never, old])
    assert (r["episodes"], r["scored"], r["lifted_any"], r["skipped_no_initial_x"]) == (4, 3, 2, 1)
    assert r["rightmost"] == {"first": 1, "full_order": 1}
    assert r["leftmost"] == {"first": 1, "full_order": 0}  # partial order is not a full one
    assert r["middle"]["first"] == 0
