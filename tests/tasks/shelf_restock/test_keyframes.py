"""The shelf keyframe oracle's state machine, on the real spec and planners."""

import numpy as np

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene, ObjectState
from oct_vla.core.state import ArmState, EEFState
from oct_vla.data.entity_tokens import ENTITY_TOKEN_DIM
from oct_vla.data.future_keyframes import FUTURE_EVENTS, FUTURE_ROLES
from oct_vla.tasks.shelf_restock.keyframes import ShelfKeyframeOracle
from oct_vla.tasks.shelf_restock.oracle.grasps import generate_top_down_grasps
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC, spec_for_rule

IDENTITY = (0.0, 0.0, 0.0, 1.0)
SIZE = (0.05, 0.05, 0.08)
LOWER_Z = DEFAULT_SPEC.lower_shelf.top_z + SIZE[2] / 2
UPPER_Z = DEFAULT_SPEC.upper_shelf.top_z + SIZE[2] / 2
LOWER_Y = DEFAULT_SPEC.lower_shelf.center_xyz[1]
UPPER_Y = DEFAULT_SPEC.upper_shelf.center_xyz[1]
HOME = {"left": (-0.4, -0.2, 1.2), "right": (0.4, -0.2, 1.2)}
EVENT_COL = ENTITY_TOKEN_DIM + len(FUTURE_ROLES)


def _scene(**positions):
    return ObjectScene(0.0, tuple(ObjectState(k, Pose(p, IDENTITY, WORKCELL_FRAME), SIZE, 1.0, 1.0)
                                  for k, p in sorted(positions.items())))


def _eef(left=HOME["left"]):
    return EEFState(ArmState(Pose(left, IDENTITY, WORKCELL_FRAME), 0.8),
                    ArmState(Pose(HOME["right"], IDENTITY, WORKCELL_FRAME), 0.8))


def _above(position, height=0.14):
    return (position[0], position[1], position[2] + height)


def _event(rows, mask):
    return FUTURE_EVENTS[int(np.argmax(rows[0, EVENT_COL:]))] if mask.all() else None


def test_grasp_then_place_then_next_object_then_nothing():
    a, b = (-0.25, LOWER_Y, LOWER_Z), (0.05, LOWER_Y, LOWER_Z)
    a_up, b_up = (-0.2, UPPER_Y, UPPER_Z), (-0.1, UPPER_Y, UPPER_Z)
    carried = (-0.25, LOWER_Y, LOWER_Z + 0.2)
    oracle = ShelfKeyframeOracle(spec_for_rule("leftmost"))
    steps = [
        (_scene(a=a, b=b), _eef()),                          # grasp a (leftmost)
        (_scene(a=carried, b=b), _eef(_above(carried))),     # lifted: place a
        (_scene(a=a_up, b=b), _eef(_above(a_up))),           # on the shelf, still held
        (_scene(a=a_up, b=b), _eef()),                       # let go: grasp b
        (_scene(a=a_up, b=b_up), _eef()),                    # b placed, arm away: done
    ]
    events = [_event(*oracle(scene, eef)) for scene, eef in steps]
    assert events == ["grasp", "place", "place", "grasp", None]
    assert [t for _, t in oracle.history] == ["a", "a", "a", "b", None]


def test_rightmost_rule_selects_the_other_object():
    a, b = (-0.25, LOWER_Y, LOWER_Z), (0.05, LOWER_Y, LOWER_Z)
    oracle = ShelfKeyframeOracle(spec_for_rule("rightmost"))
    oracle(_scene(a=a, b=b), _eef())
    assert oracle.target == "b"


def test_grasp_rows_are_the_experts_first_candidate():
    a = (-0.25, LOWER_Y, LOWER_Z)
    rows, mask = ShelfKeyframeOracle(spec_for_rule("leftmost"))(_scene(a=a), _eef())
    expected = generate_top_down_grasps(_scene(a=a).get("a"), gripper_max_width=0.08,
                                        standoff=0.10)[0]
    np.testing.assert_allclose(rows[0, :3], a, atol=1e-6)
    np.testing.assert_allclose(rows[1, :3], expected.grasp_pose.position, atol=1e-6)
    assert rows[1, ENTITY_TOKEN_DIM - 3] == 1.0          # left_gripper type: the mover
    assert DEFAULT_SPEC.upper_shelf.contains(tuple(rows[2, :3]))  # goal on the upper shelf
    np.testing.assert_array_equal(rows[:, ENTITY_TOKEN_DIM:EVENT_COL], np.eye(3))


def test_a_dropped_target_is_regrasped_where_it_lies():
    a, b = (-0.25, LOWER_Y, LOWER_Z), (0.05, LOWER_Y, LOWER_Z)
    dropped = (-0.15, LOWER_Y, LOWER_Z)
    oracle = ShelfKeyframeOracle(spec_for_rule("leftmost"))
    oracle(_scene(a=a, b=b), _eef())
    oracle(_scene(a=(-0.25, LOWER_Y, LOWER_Z + 0.2), b=b), _eef())
    rows, mask = oracle(_scene(a=dropped, b=b), _eef())
    assert _event(rows, mask) == "grasp" and oracle.target == "a"
    np.testing.assert_allclose(rows[0, :3], dropped, atol=1e-6)
