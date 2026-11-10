"""Transfer rules: each selector's order, and the rule/instruction contract."""

from __future__ import annotations

import dataclasses

import pytest

from oct_vla.core.frames import Pose
from oct_vla.core.objects import ObjectScene, ObjectState
from oct_vla.tasks.shelf_restock.manager import SELECTORS, ShelfRestockManager
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC, RULE_INSTRUCTIONS, spec_for_rule


def _scene(xs):
    return ObjectScene(0.0, tuple(
        ObjectState(f"obj_{i}", Pose((x, -0.24, 0.78), (0.0, 0.0, 0.0, 1.0)), (0.05, 0.05, 0.06), 1.0, 1.0)
        for i, x in enumerate(xs)
    ))


def _order(rule, xs):
    scene, remaining, out = _scene(xs), [f"obj_{i}" for i in range(len(xs))], []
    while remaining:
        pick = SELECTORS[rule](tuple(remaining), scene)
        out.append(pick)
        remaining.remove(pick)
    return out


def test_each_rule_orders_three_objects_as_named():
    xs = (-0.10, -0.40, -0.25)            # obj_1 left, obj_2 middle, obj_0 right
    assert _order("leftmost", xs) == ["obj_1", "obj_2", "obj_0"]
    assert _order("rightmost", xs) == ["obj_0", "obj_2", "obj_1"]
    assert _order("middle", xs) == ["obj_2", "obj_1", "obj_0"]


def test_rules_disagree_on_the_first_object_of_every_three_object_scene():
    xs = (-0.30, -0.15, -0.45)
    firsts = {rule: _order(rule, xs)[0] for rule in SELECTORS}
    assert len(set(firsts.values())) == 3


def test_spec_for_rule_sets_a_matching_instruction_and_the_manager_follows_it():
    spec = spec_for_rule("rightmost")
    assert spec.instruction == RULE_INSTRUCTIONS["rightmost"] != DEFAULT_SPEC.instruction
    scene = _scene((-0.10, -0.40, -0.25))
    ctx = ShelfRestockManager(spec).next_context(scene)
    assert ctx.target_track_id == "obj_0" and ctx.instruction == spec.instruction
    assert ShelfRestockManager(DEFAULT_SPEC).next_context(scene).target_track_id == "obj_1"


def test_a_rule_without_its_instruction_is_refused():
    with pytest.raises(ValueError, match="does not describe rule"):
        dataclasses.replace(DEFAULT_SPEC, rule="rightmost")
    with pytest.raises(ValueError, match="rule must be one of"):
        spec_for_rule("largest")


def test_leftmost_instruction_is_unchanged_from_earlier_corpora():
    assert DEFAULT_SPEC.rule == "leftmost"
    assert DEFAULT_SPEC.instruction == (
        "Restock the leftmost object from the lower shelf to the upper shelf, "
        "then compact it toward the previous neighbor if one exists."
    )
