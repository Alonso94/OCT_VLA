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


def test_served_scores_count_a_scene_without_a_lift_as_a_miss():
    done_right = dict(_episode({"c": 10, "b": 90, "a": 200}), success=True)
    order_wrong = dict(_episode({"a": 10, "b": 90, "c": 200}), success=True)
    first_only = dict(_episode({"c": 12}), success=False)
    never = dict(_episode({}), success=False)
    r = S.served([done_right, order_wrong, first_only, never], "rightmost")
    assert r["scenes"] == 4 and r["lift_rate"] == 0.75
    assert r["S_pick"] == 0.5                      # never-lifted counts against it
    assert abs(r["S_pick_given_lift"] - 2 / 3) < 1e-9
    assert r["S_task"] == 0.5 and r["S_order"] == 0.25 and r["S_joint"] == 0.25


def test_served_rule_is_read_from_the_intervention():
    assert S.served_rule({"future_intervention": {"oracle_rule": "rightmost"}}) == "rightmost"
    assert S.served_rule({}) is None
