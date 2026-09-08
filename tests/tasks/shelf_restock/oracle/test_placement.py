import pytest

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
from oct_vla.tasks.shelf_restock.oracle.grasps import GRASP_TCP_OFFSET
from oct_vla.tasks.shelf_restock.oracle.placement import (
    COMPACTED_GAP,
    PLACEMENT_GAP,
    PlacementError,
    plan_placement,
    resting_z,
)
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC
from oct_vla.tasks.shelf_restock.success import check_placement


def obj(track_id="target", position=(0.0, -0.02, 0.96), size=(0.04, 0.04, 0.05)) -> ObjectState:
    return ObjectState(track_id, Pose(position, (0, 0, 0, 1)), size, 1.0, 1.0)


def test_resting_z_puts_the_object_on_the_deck_not_inside_it():
    target = obj(size=(0.04, 0.04, 0.05))
    assert resting_z(DEFAULT_SPEC, target) == pytest.approx(DEFAULT_SPEC.upper_shelf.top_z + 0.025)


def test_without_a_neighbour_the_object_goes_to_the_middle_of_the_deck():
    placement = plan_placement(DEFAULT_SPEC, obj(), 0.0, None, standoff=0.1)
    shelf = DEFAULT_SPEC.upper_shelf
    assert placement.object_pose.position[:2] == pytest.approx(shelf.center_xyz[:2])
    assert placement.needs_compaction is False
    assert placement.compact_pose is None


def test_placement_lands_on_the_upper_shelf():
    placement = plan_placement(DEFAULT_SPEC, obj(), 0.0, None, standoff=0.1)
    assert DEFAULT_SPEC.upper_shelf.contains(placement.object_pose.position)


def test_with_a_neighbour_the_object_is_set_down_a_gap_away_then_compacted_closer():
    neighbour = obj("neighbor", position=(-0.05, -0.02, 0.96))
    placement = plan_placement(DEFAULT_SPEC, obj(), 0.0, neighbour, standoff=0.1)

    assert placement.needs_compaction is True
    reach = 0.02 + 0.02  # both horizontal radii
    placed_gap = abs(placement.object_pose.position[0] - neighbour.pose.position[0]) - reach
    compacted_gap = (
        abs(placement.compacted_object_pose.position[0] - neighbour.pose.position[0]) - reach
    )
    assert placed_gap == pytest.approx(PLACEMENT_GAP)
    assert compacted_gap == pytest.approx(COMPACTED_GAP)
    assert compacted_gap < placed_gap, "compaction must actually close the gap"


def test_compaction_moves_toward_the_neighbour_not_away():
    neighbour = obj("neighbor", position=(-0.05, -0.02, 0.96))
    placement = plan_placement(DEFAULT_SPEC, obj(), 0.0, neighbour, standoff=0.1)
    before = abs(placement.object_pose.position[0] - neighbour.pose.position[0])
    after = abs(placement.compacted_object_pose.position[0] - neighbour.pose.position[0])
    assert after < before


def test_placement_satisfies_the_success_check_only_after_compaction():
    """Placement is deliberately wider than the threshold so that the
    compaction step has something to do."""
    neighbour = obj("neighbor", position=(-0.05, -0.02, 0.96))
    placement = plan_placement(DEFAULT_SPEC, obj(), 0.0, neighbour, standoff=0.1)
    context = TaskContext(
        instruction="restock", target_track_id="target", previous_neighbor_track_id="neighbor"
    )

    placed = ObjectState("target", placement.object_pose, obj().size_xyz, 1.0, 1.0)
    before = check_placement(ObjectScene(0.0, (placed, neighbour)), context, DEFAULT_SPEC)
    assert before.target_on_upper_shelf is True
    assert before.compacted_to_neighbor is False

    compacted = ObjectState("target", placement.compacted_object_pose, obj().size_xyz, 1.0, 1.0)
    after = check_placement(ObjectScene(0.0, (compacted, neighbour)), context, DEFAULT_SPEC)
    assert after.compacted_to_neighbor is True
    assert after.success is True


def test_object_is_placed_on_whichever_side_of_the_neighbour_has_room():
    shelf = DEFAULT_SPEC.upper_shelf
    near_high_edge = obj("neighbor", position=(shelf.center_xyz[0] + 0.15, -0.02, 0.96))
    placement = plan_placement(DEFAULT_SPEC, obj(), 0.0, near_high_edge, standoff=0.1)
    # No room to the +x side, so it must go to -x.
    assert placement.object_pose.position[0] < near_high_edge.pose.position[0]
    assert shelf.contains(placement.object_pose.position)


def test_raises_when_the_neighbour_leaves_no_room_at_all():
    tiny = DEFAULT_SPEC.upper_shelf
    huge = obj("neighbor", position=(tiny.center_xyz[0], -0.02, 0.96), size=(0.4, 0.4, 0.05))
    with pytest.raises(PlacementError):
        plan_placement(DEFAULT_SPEC, obj(), 0.0, huge, standoff=0.1)


def test_commanded_pose_accounts_for_the_tcp_offset_and_object_height():
    height, depth = 0.05, 0.015
    placement = plan_placement(
        DEFAULT_SPEC, obj(size=(0.04, 0.04, height)), 0.0, None, standoff=0.09
    )
    pads_z = placement.object_pose.position[2] + height / 2 - depth
    assert placement.place_pose.position[2] == pytest.approx(pads_z + GRASP_TCP_OFFSET)
    assert placement.preplace_pose.position[2] == pytest.approx(
        placement.place_pose.position[2] + 0.09
    )
    assert placement.place_pose.frame == WORKCELL_FRAME
