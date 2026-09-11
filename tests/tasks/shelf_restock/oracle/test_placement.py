from math import pi

import pytest

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.geometry import exp, rotate
from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
from oct_vla.tasks.shelf_restock.oracle.grasps import (
    DEFAULT_GRASP_DEPTH,
    GRASP_TCP_OFFSET,
    holding_tcp_pose,
)
from oct_vla.tasks.shelf_restock.oracle.placement import (
    COMPACTED_GAP,
    CONTACT_CLEARANCE,
    PLACEMENT_GAP,
    PUSH_PATH_CONSTRAINT,
    PUSHER_HALF_WIDTH,
    PlacementError,
    plan_placement,
    plan_push,
    push_axis_half_extent,
    resting_z,
)
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC
from oct_vla.tasks.shelf_restock.success import check_placement, horizontal_radius


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
    assert placement.compacted_object_pose is None


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
    huge = obj("neighbor", position=(tiny.center_xyz[0], -0.02, 0.96), size=(0.6, 0.6, 0.05))
    with pytest.raises(PlacementError):
        plan_placement(DEFAULT_SPEC, obj(), 0.0, huge, standoff=0.1)


def test_with_no_neighbour_and_side_right_the_row_starts_one_step_left_of_centre():
    """A fresh row must start inside the overlap of both arms' reach, not at
    a deck edge: the right arm measured as unable to plan to the deck's outer
    thirds, so starting a (compaction-clustered) row at an edge parks it
    where the opposite arm cannot reach it."""
    shelf = DEFAULT_SPEC.upper_shelf
    target = obj()
    placement = plan_placement(DEFAULT_SPEC, target, 0.0, None, standoff=0.1, side="right")
    step = 2 * horizontal_radius(target) + PLACEMENT_GAP
    expected_x = shelf.center_xyz[0] - step
    assert placement.object_pose.position[0] == pytest.approx(expected_x)


def test_with_no_neighbour_and_side_left_the_row_starts_one_step_right_of_centre():
    shelf = DEFAULT_SPEC.upper_shelf
    target = obj()
    placement = plan_placement(DEFAULT_SPEC, target, 0.0, None, standoff=0.1, side="left")
    step = 2 * horizontal_radius(target) + PLACEMENT_GAP
    expected_x = shelf.center_xyz[0] + step
    assert placement.object_pose.position[0] == pytest.approx(expected_x)


def test_with_no_neighbour_and_no_side_it_still_goes_to_the_deck_centre():
    shelf = DEFAULT_SPEC.upper_shelf
    placement = plan_placement(DEFAULT_SPEC, obj(), 0.0, None, standoff=0.1, side=None)
    assert placement.object_pose.position[0] == pytest.approx(shelf.center_xyz[0])


def test_with_a_neighbour_side_right_prefers_plus_x_even_when_minus_x_has_more_room():
    """The roomier-side heuristic alone would send this back toward the
    middle, which would eventually stack a later object on an earlier one; a
    row only works if it grows consistently toward the compacting arm."""
    neighbour = obj("neighbor", position=(0.1, -0.02, 0.96))
    placement = plan_placement(DEFAULT_SPEC, obj(), 0.0, neighbour, standoff=0.1, side="right")
    assert placement.object_pose.position[0] > neighbour.pose.position[0]


def test_with_a_neighbour_side_falls_back_when_the_preferred_side_has_no_room():
    neighbour = obj("neighbor", position=(0.25, -0.02, 0.96))
    placement = plan_placement(DEFAULT_SPEC, obj(), 0.0, neighbour, standoff=0.1, side="right")
    assert placement.object_pose.position[0] < neighbour.pose.position[0]


def test_freshly_placed_gap_stays_above_compaction_threshold_while_compacted_gap_stays_below():
    """The property that keeps compaction non-vacuous: a freshly placed
    object must not already read as compacted, but after compaction it must."""
    neighbour = obj("neighbor", position=(-0.05, -0.02, 0.96))
    placement = plan_placement(DEFAULT_SPEC, obj(), 0.0, neighbour, standoff=0.1)
    reach = horizontal_radius(obj()) + horizontal_radius(neighbour)
    placed_gap = abs(placement.object_pose.position[0] - neighbour.pose.position[0]) - reach
    compacted_gap = (
        abs(placement.compacted_object_pose.position[0] - neighbour.pose.position[0]) - reach
    )
    assert placed_gap > DEFAULT_SPEC.compaction_distance
    assert compacted_gap < DEFAULT_SPEC.compaction_distance


def test_a_three_object_chain_stays_within_reach_of_the_deck_centre():
    """Regression guard for the reach limit a live run found: the right arm
    could not plan to x=-0.124 on the upper deck. Compaction clusters a row
    near wherever it started rather than letting it march across the deck,
    so every placed and compacted pose in a chain built the way the oracle
    builds one (place beside the previous object, compact it, use the
    compacted pose as the next neighbour) must stay close to the deck centre
    -- not just the row's starting point."""
    shelf = DEFAULT_SPEC.upper_shelf
    size = (0.054, 0.048, 0.077)

    first_target = obj("obj-0", size=size)
    first = plan_placement(DEFAULT_SPEC, first_target, 0.0, None, standoff=0.1, side="right")
    xs = [first.object_pose.position[0]]

    neighbour = ObjectState("obj-0", first.object_pose, size, 1.0, 1.0)
    for i in range(1, 3):
        target = obj(f"obj-{i}", size=size)
        placement = plan_placement(DEFAULT_SPEC, target, 0.0, neighbour, standoff=0.1, side="right")
        xs.append(placement.object_pose.position[0])
        xs.append(placement.compacted_object_pose.position[0])
        neighbour = ObjectState(f"obj-{i}", placement.compacted_object_pose, size, 1.0, 1.0)

    for x in xs:
        assert abs(x - shelf.center_xyz[0]) <= 0.16


def test_commanded_pose_accounts_for_the_tcp_offset_and_object_height():
    height, depth = 0.05, DEFAULT_GRASP_DEPTH
    placement = plan_placement(
        DEFAULT_SPEC, obj(size=(0.04, 0.04, height)), 0.0, None, standoff=0.09
    )
    pads_z = placement.object_pose.position[2] + height / 2 - depth
    assert placement.place_pose.position[2] == pytest.approx(pads_z + GRASP_TCP_OFFSET)
    assert placement.preplace_pose.position[2] == pytest.approx(
        placement.place_pose.position[2] + 0.09
    )
    assert placement.place_pose.frame == WORKCELL_FRAME


def test_push_poses_all_share_the_objects_own_y():
    target = obj(position=(-0.004, -0.02, 0.96))
    push = plan_push(target, -0.062, standoff=0.1)
    for pose in (push.approach_pose, push.start_pose, push.end_pose, push.retreat_pose):
        assert pose.position[1] == pytest.approx(target.pose.position[1])


def test_start_and_end_poses_put_the_pads_at_the_objects_own_centre_height():
    """Pads sit at the object's own centre height (not a fixed offset from the
    world origin) -- that is what keeps a sideways shove from tipping the
    object, per plan_push's docstring."""
    target = obj(position=(-0.004, -0.02, 0.96))
    push = plan_push(target, -0.062, standoff=0.1)
    expected_z = target.pose.position[2] + GRASP_TCP_OFFSET
    assert push.start_pose.position[2] == pytest.approx(expected_z)
    assert push.end_pose.position[2] == pytest.approx(expected_z)


def test_push_orientation_approach_axis_is_vertical_for_every_pose():
    """A horizontal approach (fingers along x) is what failed to plan on the
    live run; every push pose must point straight down instead, the only
    orientation this workcell plans reliably."""
    target = obj(position=(-0.004, -0.02, 0.96))
    push = plan_push(target, -0.062, standoff=0.1)
    for pose in (push.approach_pose, push.start_pose, push.end_pose, push.retreat_pose):
        assert rotate(pose.orientation, (1, 0, 0)) == pytest.approx((0, 0, -1))


def test_pushing_toward_plus_x_starts_on_the_objects_minus_x_side_and_ends_further_plus_x():
    target = obj(position=(-0.004, -0.02, 0.96))
    push = plan_push(target, 0.05, standoff=0.1)
    assert push.start_pose.position[0] < target.pose.position[0]
    assert push.end_pose.position[0] > push.start_pose.position[0]


def test_pushing_toward_minus_x_starts_on_the_objects_plus_x_side_and_ends_further_minus_x():
    target = obj(position=(-0.004, -0.02, 0.96))
    push = plan_push(target, -0.062, standoff=0.1)
    assert push.start_pose.position[0] > target.pose.position[0]
    assert push.end_pose.position[0] < push.start_pose.position[0]


def test_the_gripper_travels_the_objects_distance_plus_the_contact_clearance():
    target = obj(position=(-0.004, -0.02, 0.96))
    target_x = -0.062
    push = plan_push(target, target_x, standoff=0.1)
    direction = -1.0
    assert push.end_pose.position[0] - push.start_pose.position[0] == pytest.approx(
        (target_x - target.pose.position[0]) + direction * CONTACT_CLEARANCE
    )


def test_start_pose_clears_the_objects_real_face_by_exactly_contact_clearance():
    """The descent must clear the object's actual face -- accounting for both
    its own extent and the gripper's own half-width -- by exactly
    CONTACT_CLEARANCE, not by whatever margin `horizontal_radius` happens to
    leave for a rotated object."""
    target = obj(position=(-0.004, -0.02, 0.96))
    target_x = -0.062
    push = plan_push(target, target_x, standoff=0.1)
    extent = push_axis_half_extent(target)
    clearance = abs(push.start_pose.position[0] - target.pose.position[0]) - extent
    clearance -= PUSHER_HALF_WIDTH
    assert clearance == pytest.approx(CONTACT_CLEARANCE)


def test_approach_and_retreat_poses_are_standoff_directly_above_start_and_end():
    target = obj(position=(-0.004, -0.02, 0.96))
    target_x = -0.062
    standoff = 0.1
    push = plan_push(target, target_x, standoff=standoff)
    assert push.approach_pose.position[0] == pytest.approx(push.start_pose.position[0])
    assert push.approach_pose.position[1] == pytest.approx(push.start_pose.position[1])
    assert push.approach_pose.position[2] == pytest.approx(push.start_pose.position[2] + standoff)
    assert push.retreat_pose.position[0] == pytest.approx(push.end_pose.position[0])
    assert push.retreat_pose.position[1] == pytest.approx(push.end_pose.position[1])
    assert push.retreat_pose.position[2] == pytest.approx(push.end_pose.position[2] + standoff)


def test_pushing_to_the_objects_current_x_raises():
    target = obj(position=(-0.004, -0.02, 0.96))
    with pytest.raises(PlacementError):
        plan_push(target, target.pose.position[0], standoff=0.1)


def test_end_pose_lands_the_pushing_face_exactly_on_the_objects_target_face_minus_x():
    target = obj(position=(-0.004, -0.02, 0.96))
    target_x = -0.062
    push = plan_push(target, target_x, standoff=0.1)
    extent = push_axis_half_extent(target)
    # Pushing toward -x: the gripper is on the object's +x side, so its
    # pushing face is PUSHER_HALF_WIDTH in the -x direction from the
    # commanded pose.
    pushing_face = push.end_pose.position[0] - PUSHER_HALF_WIDTH
    assert pushing_face == pytest.approx(target_x + extent)


def test_end_pose_lands_the_pushing_face_exactly_on_the_objects_target_face_plus_x():
    target = obj(position=(-0.004, -0.02, 0.96))
    target_x = 0.05
    push = plan_push(target, target_x, standoff=0.1)
    extent = push_axis_half_extent(target)
    # Pushing toward +x: the gripper is on the object's -x side, so its
    # pushing face is PUSHER_HALF_WIDTH in the +x direction from the
    # commanded pose.
    pushing_face = push.end_pose.position[0] + PUSHER_HALF_WIDTH
    assert pushing_face == pytest.approx(target_x - extent)


def test_push_axis_half_extent_is_half_the_x_size_when_unrotated():
    target = obj(size=(0.06, 0.03, 0.05))
    assert push_axis_half_extent(target) == pytest.approx(0.03)


def test_push_axis_half_extent_is_half_the_y_size_when_yawed_a_quarter_turn():
    position = (0.0, -0.02, 0.96)
    yawed = ObjectState(
        "target", Pose(position, exp((0.0, 0.0, pi / 2))), (0.06, 0.03, 0.05), 1.0, 1.0
    )
    assert push_axis_half_extent(yawed) == pytest.approx(0.015)


def test_push_path_constraint_frees_rotation_and_pins_two_translations():
    """Rotational freedom is not incidental: every variant that pins the
    rotations as well as both translations is refused outright by cuRobo."""
    assert len(PUSH_PATH_CONSTRAINT) == 6
    assert PUSH_PATH_CONSTRAINT[0:3] == (0.0, 0.0, 0.0)
    assert sum(PUSH_PATH_CONSTRAINT[3:6]) == pytest.approx(2.0)
    assert PUSH_PATH_CONSTRAINT[3:6].count(0.0) == 1


def test_push_path_constraint_frees_exactly_the_axis_the_push_travels_along():
    """Derivation, not restatement: find which local axis actually carries the
    travel under the push's own orientation, and check the constraint frees
    that one and pins the other two translations. Leaving a non-travel
    translation free is what let a logged push wander 43mm toward the shelf's
    front edge. This is the test that would catch someone changing the push
    orientation without re-deriving the constraint."""
    target = obj(position=(-0.004, -0.02, 0.96))
    push = plan_push(target, -0.062, standoff=0.1)
    orientation = push.start_pose.orientation
    travel = tuple(
        e - s for s, e in zip(push.start_pose.position, push.end_pose.position, strict=True)
    )

    travel_axes = []
    for i, local_axis in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        world_axis = rotate(orientation, local_axis)
        if abs(sum(a * b for a, b in zip(world_axis, travel, strict=True))) > 1e-9:
            travel_axes.append(i)

    assert len(travel_axes) == 1
    for i in range(3):
        expected = 0.0 if i == travel_axes[0] else 1.0
        assert PUSH_PATH_CONSTRAINT[3 + i] == pytest.approx(expected)


def test_plan_push_populates_the_push_with_the_path_constraint():
    target = obj(position=(-0.004, -0.02, 0.96))
    push = plan_push(target, -0.062, standoff=0.1)
    assert push.constraint == PUSH_PATH_CONSTRAINT


def test_push_axis_half_extent_exceeds_horizontal_radius_for_a_box_yawed_45_degrees():
    """This is the case that caused the live bug: horizontal_radius returns a
    yaw-independent circumscribing radius that understates the real extent
    along the push axis once a non-square box is rotated."""
    position = (0.0, -0.02, 0.96)
    yawed = ObjectState(
        "target", Pose(position, exp((0.0, 0.0, pi / 4))), (0.054, 0.0479, 0.0774), 1.0, 1.0
    )
    assert push_axis_half_extent(yawed) > horizontal_radius(yawed)


def test_a_yawed_object_is_placed_upright_regardless_of_its_spawn_tilt():
    """The object is rigid with the gripper from grasp to release, so
    commanding the grasp's own wrist yaw at place time would set it down at
    whatever angle it spawned at. plan_placement instead cancels the
    object's spawn yaw out of the place wrist yaw so it always lands
    axis-aligned with the shelf."""
    spawn_yaw = 0.3265
    tilted = ObjectState(
        "target", Pose((0.0, -0.02, 0.96), exp((0.0, 0.0, spawn_yaw))), (0.04, 0.04, 0.05), 1.0, 1.0
    )
    grasp_wrist_yaw = spawn_yaw  # a real grasp candidate's wrist_yaw = object_yaw + offset
    placement = plan_placement(DEFAULT_SPEC, tilted, grasp_wrist_yaw, None, standoff=0.1)

    assert placement.object_pose.orientation == pytest.approx((0.0, 0.0, 0.0, 1.0))
    # The commanded wrist yaw must differ from the grasp's own by exactly the
    # object's spawn yaw -- that is what un-rotates the held object to upright.
    straight_pose = holding_tcp_pose(
        placement.object_pose.position, tilted.size_xyz[2], 0.0, grasp_depth=DEFAULT_GRASP_DEPTH
    )
    assert rotate(placement.place_pose.orientation, (0, 0, 1)) == pytest.approx(
        rotate(straight_pose.orientation, (0, 0, 1)), abs=1e-9
    )


def test_a_chained_placement_holds_the_rows_nominal_y_not_the_neighbours_observed_y():
    """A neighbour's re-observed y can drift a little from where it was
    commanded (real placement noise), and letting that drift carry into the
    next object's target would compound down a chain. The row instead always
    targets the shelf's own nominal y -- the same value the first,
    no-neighbour placement uses -- so every object in the row lands on one
    line."""
    shelf_y = DEFAULT_SPEC.upper_shelf.center_xyz[1]
    drifted_neighbour = obj("neighbor", position=(-0.05, shelf_y + 0.01, 0.96))
    placement = plan_placement(DEFAULT_SPEC, obj(), 0.0, drifted_neighbour, standoff=0.1)

    assert placement.object_pose.position[1] == pytest.approx(shelf_y)
    assert placement.compacted_object_pose.position[1] == pytest.approx(shelf_y)
