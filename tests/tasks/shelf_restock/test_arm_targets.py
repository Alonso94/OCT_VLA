"""The paired script oracle's per-arm via-points (tasks/shelf_restock/arm_targets.py)."""

import numpy as np

from oct_vla.tasks.shelf_restock.arm_targets import (
    CLOSED,
    OPEN,
    REACH,
    VIA,
    ShelfArmTargetOracle,
)
from oct_vla.tasks.shelf_restock.oracle.grasps import generate_top_down_grasps
from oct_vla.tasks.shelf_restock.spec import spec_for_rule

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene, ObjectState
from oct_vla.core.state import ArmState, EEFState
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC

IDENTITY = (0.0, 0.0, 0.0, 1.0)
SIZE = (0.05, 0.05, 0.08)
LOWER_Z = DEFAULT_SPEC.lower_shelf.top_z + SIZE[2] / 2
UPPER_Z = DEFAULT_SPEC.upper_shelf.top_z + SIZE[2] / 2
LOWER_Y = DEFAULT_SPEC.lower_shelf.center_xyz[1]
UPPER_Y = DEFAULT_SPEC.upper_shelf.center_xyz[1]
HOME = {"left": (-0.4, -0.2, 1.2), "right": (0.4, -0.2, 1.2)}


def _scene(**positions):
    return ObjectScene(0.0, tuple(ObjectState(k, Pose(p, IDENTITY, WORKCELL_FRAME), SIZE, 1.0, 1.0)
                                  for k, p in sorted(positions.items())))


def _eef(left=HOME["left"]):
    return EEFState(ArmState(Pose(left, IDENTITY, WORKCELL_FRAME), 0.8),
                    ArmState(Pose(HOME["right"], IDENTITY, WORKCELL_FRAME), 0.8))


def _vias(out, side):
    vias, mask = out
    return vias[side][mask[side]]


def test_grasp_stage_scripts_the_whole_transfer_for_the_mover_only():
    a = (-0.25, LOWER_Y, LOWER_Z)
    oracle = ShelfArmTargetOracle(spec_for_rule("leftmost"))
    out = oracle(_scene(a=a), _eef())
    assert oracle.stage == "grasp" and out[0].shape == (2, VIA, 8)
    left, right = _vias(out, 0), _vias(out, 1)
    assert len(right) == 0  # the compactor idles
    grasp = generate_top_down_grasps(_scene(a=a).get("a"), gripper_max_width=0.08,
                                     standoff=0.10)[0]
    np.testing.assert_allclose(left[0, :3], grasp.pregrasp_pose.position, atol=1e-6)
    np.testing.assert_allclose(left[1, :3], grasp.grasp_pose.position, atol=1e-6)
    # pregrasp open, grasp closes, lift/preplace closed, place opens, back off open.
    assert left[:, 7].tolist() == [OPEN, CLOSED, CLOSED, CLOSED, OPEN, OPEN]
    assert left[4, 1] > LOWER_Y  # the place is on the upper shelf


def test_reached_via_points_are_dropped_and_the_stage_follows_the_scene():
    a = (-0.25, LOWER_Y, LOWER_Z)
    oracle = ShelfArmTargetOracle(spec_for_rule("leftmost"))
    first = _vias(oracle(_scene(a=a), _eef()), 0)
    at_pregrasp = tuple(first[0, :3] + np.array([0.0, 0.0, REACH / 2]))
    second = _vias(oracle(_scene(a=a), _eef(at_pregrasp)), 0)
    np.testing.assert_allclose(second[0], first[1])  # pregrasp latched off
    carried = (-0.25, LOWER_Y, LOWER_Z + 0.2)
    oracle(_scene(a=carried), _eef((carried[0], carried[1], carried[2] + 0.14)))
    assert oracle.stage == "carry"


def test_a_resting_placed_object_starts_release_then_retreat():
    a = (-0.25, LOWER_Y, LOWER_Z)
    oracle = ShelfArmTargetOracle(spec_for_rule("leftmost"))
    plan = _vias(oracle(_scene(a=a), _eef()), 0)
    place_obj = (plan[4, 0], plan[4, 1], UPPER_Z)  # the object under the place pose
    at_place = tuple(plan[4, :3])
    carried = (-0.25, LOWER_Y, LOWER_Z + 0.2)
    oracle(_scene(a=carried), _eef((carried[0], carried[1], carried[2] + 0.14)))
    out = oracle(_scene(a=place_obj), _eef(at_place))
    assert oracle.stage == "release"
    left = _vias(out, 0)
    assert (left[:, 7] == OPEN).all() and left[-1, 2] > at_place[2] + 0.05  # back off, up
    high = (at_place[0], at_place[1], at_place[2] + 0.05)
    oracle(_scene(a=place_obj), _eef(high))
    assert oracle.stage in ("retreat", "grasp", "none", "park_mover")
