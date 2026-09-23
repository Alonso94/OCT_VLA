"""Ground-truth transfer stages: each failure mode lands in its own bucket."""

from oct_vla.core.frames import Pose
from oct_vla.core.objects import ObjectScene, ObjectState
from oct_vla.core.state import ArmState, EEFState
from oct_vla.tasks.shelf_restock.events import HOLD_RADIUS, OUTCOMES, TransferEvents
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC as SPEC

IDENTITY = (0.0, 0.0, 0.0, 1.0)
FAR = (5.0, 5.0, 5.0)


def _on(shelf):
    cx, cy, _ = shelf.center_xyz
    return (cx, cy, shelf.top_z + 0.04)


LOWER = _on(SPEC.lower_shelf)
UPPER = _on(SPEC.upper_shelf)
#: Lifted clear of the lower deck, not yet over the upper one.
AIR = (LOWER[0], LOWER[1], SPEC.lower_shelf.top_z + 0.15)
assert not SPEC.upper_shelf.contains(AIR)
#: Above the upper deck's footprint, too high to count as resting on it.
OVER = (UPPER[0], UPPER[1], SPEC.upper_shelf.top_z + SPEC.upper_shelf.occupancy_height + 0.05)
FLOOR = (LOWER[0], LOWER[1], 0.0)


def _eef(near):
    """Both end-effectors far away, or the left one holding at `near`."""
    held = (near[0], near[1], near[2] + HOLD_RADIUS / 2) if near else FAR
    return EEFState(ArmState(Pose(held, IDENTITY), 0.0), ArmState(Pose(FAR, IDENTITY), 0.0))


def _run(trajectory):
    """`trajectory` is a list of (object position, held?) per step."""
    events = TransferEvents(SPEC)
    for step, (position, held) in enumerate(trajectory, start=1):
        scene = ObjectScene(0.0, (ObjectState("obj_0", Pose(position, IDENTITY), (0.05,) * 3, 1, 1),))
        events.update(step, scene, _eef(position if held else None))
    return events.report()["obj_0"]


def test_each_failure_mode_is_distinguished():
    cases = {
        "never_lifted": [(LOWER, False)] * 3,
        "dropped_before_shelf": [(LOWER, True), (AIR, True), (LOWER, False)],
        "carried_not_reached": [(LOWER, True), (AIR, True), (AIR, True)],
        "reached_not_placed_held": [(AIR, True), (OVER, True), (OVER, True)],
        "reached_not_placed_dropped": [(AIR, True), (OVER, True), (FLOOR, False)],
        "placed_then_lost": [(AIR, True), (UPPER, False), (FLOOR, False)],
        "placed": [(AIR, True), (OVER, True), (UPPER, False)],
    }
    assert set(cases) == set(OUTCOMES)
    for expected, trajectory in cases.items():
        assert _run(trajectory)["outcome"] == expected, expected


def test_stage_steps_and_hold_fraction():
    report = _run([(LOWER, True), (AIR, True), (AIR, False), (OVER, True), (UPPER, False)])
    assert report["lifted_step"] == 2
    assert report["over_upper_step"] == 4
    assert report["placed_step"] == 5
    assert report["dropped_step"] is None
    # Airborne on steps 2-4 -- resting on the upper deck is above the lift
    # height but not carried -- and in hand on 2 and 4.
    assert report["held_fraction"] == round(2 / 3, 3)
