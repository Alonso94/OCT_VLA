from math import pi

import pytest

from oct_vla.core.frames import WORKCELL_FRAME, Pose, Transform
from oct_vla.core.geometry import exp
from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
from oct_vla.core.observation import RGBFrame
from oct_vla.robots.robotwin.backend import ArmReading, Contact, Reading, Trajectory, decode_pose
from oct_vla.tasks.shelf_restock.manager import ShelfRestockManager
from oct_vla.tasks.shelf_restock.oracle.expert import (
    CLOSED,
    OPEN,
    ExpertError,
    ShelfRestockExpert,
)
from oct_vla.tasks.shelf_restock.oracle.grasps import generate_top_down_grasps
from oct_vla.tasks.shelf_restock.oracle.motion import MotionError
from oct_vla.tasks.shelf_restock.oracle.placement import plan_placement, plan_push
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC, ShelfRegion, ShelfRestockSpec

# Nontrivial transform (as in test_motion.py) so nothing passes on an
# accidental identity mapping between world and workcell.
WORLD_TO_WORKCELL = Transform("world", WORKCELL_FRAME, (0.0, 0.0, 0.74), exp((0.0, 0.0, pi / 2)))

LEFT_JOINTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
RIGHT_JOINTS = (-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7)

OBJECT_SIZE = (0.04, 0.04, 0.05)

TARGET_BODY = "restock_target_body"
NEIGHBOR_BODY = "restock_neighbor_body"

HOME_POSES = {
    "left": Pose((-0.3, -0.2, 1.0), (0.0, 0.0, 0.0, 1.0)),
    "right": Pose((0.3, -0.2, 1.0), (0.0, 0.0, 0.0, 1.0)),
}


def _frame() -> RGBFrame:
    return RGBFrame(1, 1, bytes(3))


def target_obj(track_id="target", position=(0.0, -0.25, 0.78), size=OBJECT_SIZE) -> ObjectState:
    return ObjectState(track_id, Pose(position, (0.0, 0.0, 0.0, 1.0)), size, 1.0, 1.0)


def neighbor_obj(track_id="neighbor", position=(-0.05, -0.06, 0.96), size=OBJECT_SIZE):
    return ObjectState(track_id, Pose(position, (0.0, 0.0, 0.0, 1.0)), size, 1.0, 1.0)


class FakePort:
    """Records what was actually planned/commanded, keyed by move_to call
    order, so tests can check per-phase behaviour (ignore_object,
    allow_contact_with, commanded pose) without touching expert.py internals.
    """

    dt = 1.0 / 250.0

    def __init__(self, contacts_by_index: dict[int, tuple[Contact, ...]] | None = None) -> None:
        self.ignored_object: str | None = None
        # One entry per move_to call, in call order: (side, pose_wxyz, constraint, ignored_object).
        self.plan_calls: list[tuple[str, tuple[float, ...], tuple, str | None]] = []
        self.commands: list[tuple[str, tuple[float, ...], tuple[float, ...], float]] = []
        self.ticks = 0
        self._contacts_by_index = contacts_by_index or {}
        self._call_index = -1

    def contacts(self) -> tuple[Contact, ...]:
        return self._contacts_by_index.get(self._call_index, ())

    def reset(self, seed: int) -> None: ...

    def read(self) -> Reading:
        left = ArmReading((-0.3, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0), 0.5, LEFT_JOINTS)
        right = ArmReading((0.3, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0), 0.5, RIGHT_JOINTS)
        return Reading(left, right, (_frame(), _frame(), _frame()))

    def plan(self, side, pose_wxyz, constraint=None) -> Trajectory:
        self._call_index += 1
        self.plan_calls.append((side, tuple(pose_wxyz), constraint, self.ignored_object))
        return Trajectory(((0.0,) * 7,), ((0.0,) * 7,))

    def command(self, side, q, qdot, gripper) -> None:
        self.commands.append((side, tuple(q), tuple(qdot), gripper))

    def tick(self) -> None:
        self.ticks += 1

    def hold(self) -> None: ...

    def close(self) -> None: ...


def decoded_pose(pose_wxyz) -> Pose:
    return WORLD_TO_WORKCELL.apply_pose(decode_pose(pose_wxyz))


class QueuedObserve:
    """Returns scenes from a queue, one per call, holding the last scene once
    the queue is drained -- for run()/re-observation tests where the number
    of observe() calls is known but callers should not have to count exactly.
    """

    def __init__(self, scenes) -> None:
        self._scenes = list(scenes)

    def __call__(self) -> ObjectScene:
        if len(self._scenes) > 1:
            return self._scenes.pop(0)
        return self._scenes[0]


def make_expert(port, observe, *, spec=DEFAULT_SPEC, body_names=None, **kwargs):
    body_names = body_names if body_names is not None else {"target": TARGET_BODY}
    return ShelfRestockExpert(
        spec, port, observe, WORLD_TO_WORKCELL, body_names, HOME_POSES, **kwargs
    )


NO_COMPACTION_PHASES = (
    "open",
    "pregrasp",
    "descend",
    "close",
    "lift",
    "preplace",
    "place",
    "release",
    "retreat",
)
COMPACTION_PHASES = (
    "park mover",
    "compact close",
    "compact approach",
    "compact contact",
    "compact push",
    "compact retreat",
    "park compactor",
)


def test_a_transfer_with_no_neighbour_runs_exactly_the_nine_phases_in_order():
    target = target_obj()
    scene = ObjectScene(0.0, (target,))
    port = FakePort()
    expert = make_expert(port, lambda: scene)
    context = TaskContext(instruction=DEFAULT_SPEC.instruction, target_track_id="target")

    record = expert.transfer(context)

    assert tuple(phase for phase, _ in record.motions) == NO_COMPACTION_PHASES
    assert record.compacted is False


def test_a_transfer_with_a_neighbour_additionally_runs_the_compaction_phases_in_order():
    target = target_obj()
    neighbor = neighbor_obj()
    scene = ObjectScene(0.0, (target, neighbor))
    port = FakePort()
    expert = make_expert(port, lambda: scene)
    context = TaskContext(
        instruction=DEFAULT_SPEC.instruction,
        target_track_id="target",
        previous_neighbor_track_id="neighbor",
    )

    record = expert.transfer(context)

    phases = tuple(phase for phase, _ in record.motions)
    assert phases == NO_COMPACTION_PHASES + COMPACTION_PHASES
    assert record.compacted is True


def test_compaction_phases_run_on_the_other_arm_than_the_transfer_phases():
    target = target_obj()
    neighbor = neighbor_obj()
    scene = ObjectScene(0.0, (target, neighbor))
    port = FakePort()
    expert = make_expert(port, lambda: scene)
    context = TaskContext(
        instruction=DEFAULT_SPEC.instruction,
        target_track_id="target",
        previous_neighbor_track_id="neighbor",
    )

    record = expert.transfer(context)
    by_phase = dict(record.motions)

    for phase in NO_COMPACTION_PHASES:
        assert by_phase[phase].side == "left"
    for phase in (
        "compact close",
        "compact approach",
        "compact contact",
        "compact push",
        "compact retreat",
    ):
        assert by_phase[phase].side == "right"
    assert by_phase["park mover"].side == "left"
    assert by_phase["park compactor"].side == "right"


def test_gripper_stays_closed_for_every_compaction_phase_except_parking_the_mover():
    target = target_obj()
    neighbor = neighbor_obj()
    scene = ObjectScene(0.0, (target, neighbor))
    port = FakePort()
    expert = make_expert(port, lambda: scene)
    context = TaskContext(
        instruction=DEFAULT_SPEC.instruction,
        target_track_id="target",
        previous_neighbor_track_id="neighbor",
    )

    record = expert.transfer(context)
    by_phase = dict(record.motions)

    # "compact close" ramps the gripper shut (set_gripper interpolates from
    # its current opening), so it never *equals* CLOSED at every tick -- but
    # it must end there and never tick back open.
    closing = by_phase["compact close"].gripper
    assert closing[-1] == pytest.approx(CLOSED)
    assert all(g1 >= g2 for g1, g2 in zip(closing[:-1], closing[1:], strict=True))

    # Every move_to phase that follows commands one fixed gripper value for
    # its whole trajectory -- "park mover" releases the *placing* arm, not
    # the compactor, so everything else touching the compactor must stay shut.
    for phase in (
        "compact approach",
        "compact contact",
        "compact push",
        "compact retreat",
        "park compactor",
    ):
        assert set(by_phase[phase].gripper) == {CLOSED}
    assert set(by_phase["park mover"].gripper) == {OPEN}


def test_push_is_computed_from_the_reobserved_pose_not_the_planned_one():
    target = target_obj()
    neighbor = neighbor_obj()
    candidate = generate_top_down_grasps(target, gripper_max_width=0.08, standoff=0.1)[0]
    placement = plan_placement(
        DEFAULT_SPEC, target, candidate.wrist_yaw, neighbor, standoff=0.1, side="right"
    )
    predicted = placement.object_pose

    # Simulate the object landing away from its intended pose: a push aimed
    # at `predicted` would therefore miss.
    actual_position = (predicted.position[0] + 0.03, predicted.position[1], predicted.position[2])
    placed = ObjectState(
        "target", Pose(actual_position, predicted.orientation), OBJECT_SIZE, 1.0, 1.0
    )

    scene_before_place = ObjectScene(0.0, (target, neighbor))
    scene_after_place = ObjectScene(0.0, (placed, neighbor))
    observe = QueuedObserve([scene_before_place, scene_after_place])

    port = FakePort()
    expert = make_expert(port, observe)
    context = TaskContext(
        instruction=DEFAULT_SPEC.instruction,
        target_track_id="target",
        previous_neighbor_track_id="neighbor",
    )
    expert.transfer(context)

    expected_push = plan_push(placed, placement.compacted_object_pose.position[0], standoff=0.1)
    naive_push = plan_push(
        ObjectState("target", predicted, OBJECT_SIZE, 1.0, 1.0),
        placement.compacted_object_pose.position[0],
        standoff=0.1,
    )

    # Every move_to call, in the order it happens -- matches port.plan_calls
    # 1:1. "compact close" is excluded: it is a set_gripper call, not a
    # move_to, so it never reaches port.plan.
    move_phases = (
        NO_COMPACTION_PHASES[1:3]
        + NO_COMPACTION_PHASES[4:7]
        + (NO_COMPACTION_PHASES[8],)
        + tuple(phase for phase in COMPACTION_PHASES if phase != "compact close")
    )
    assert len(move_phases) == len(port.plan_calls)
    contact_index = move_phases.index("compact contact")
    push_index = move_phases.index("compact push")

    contact_pose = decoded_pose(port.plan_calls[contact_index][1])
    push_pose = decoded_pose(port.plan_calls[push_index][1])

    assert contact_pose.position == pytest.approx(expected_push.start_pose.position, abs=1e-6)
    # Travel axis (x) comes from the re-observed object, which is the point of
    # this test. y and z deliberately do not: they are re-aimed at the arm's
    # own achieved pose so the straight-push constraint is satisfiable from
    # the first waypoint (see _straight_push_to).
    assert push_pose.position[0] == pytest.approx(expected_push.end_pose.position[0], abs=1e-6)
    assert contact_pose.position != pytest.approx(naive_push.start_pose.position, abs=1e-6)


def test_the_push_goal_ends_on_the_row_line_at_the_arms_achieved_height():
    """A closed gripper is a narrow nub against a much wider object face, so
    an off-centre contact torques the object and it walks toward the shelf's
    front edge as it slides -- measured at ~20mm over a ~100mm push. Ending
    the push on the row's own y turns what lateral motion the blade has into
    a correction toward the line the row sits on. Height still comes from the
    arm, which is the authority on where it actually is."""
    target = target_obj()
    neighbor = neighbor_obj()
    scene = ObjectScene(0.0, (target, neighbor))
    port = FakePort()
    expert = make_expert(port, lambda: scene)
    context = TaskContext(
        instruction=DEFAULT_SPEC.instruction,
        target_track_id="target",
        previous_neighbor_track_id="neighbor",
    )
    expert.transfer(context)

    move_phases = (
        NO_COMPACTION_PHASES[1:3]
        + NO_COMPACTION_PHASES[4:7]
        + (NO_COMPACTION_PHASES[8],)
        + tuple(phase for phase in COMPACTION_PHASES if phase != "compact close")
    )
    push_pose = decoded_pose(port.plan_calls[move_phases.index("compact push")][1])
    achieved = WORLD_TO_WORKCELL.apply_pose(decode_pose(port.read().right.pose_wxyz))

    assert push_pose.position[1] == pytest.approx(DEFAULT_SPEC.upper_shelf.center_xyz[1], abs=1e-6)
    assert push_pose.position[2] == pytest.approx(achieved.position[2], abs=1e-6)


def test_ignore_object_is_the_targets_track_id_on_carrying_phases():
    target = target_obj()
    scene = ObjectScene(0.0, (target,))
    port = FakePort()
    expert = make_expert(port, lambda: scene)
    context = TaskContext(instruction=DEFAULT_SPEC.instruction, target_track_id="target")
    expert.transfer(context)

    # move_to call order: pregrasp, descend, lift, preplace, place, retreat.
    ignored = [call[3] for call in port.plan_calls]
    assert ignored == [None, "target", "target", "target", "target", "target"]


def test_allow_contact_with_uses_the_body_name_not_the_track_id():
    """Body name and track_id are deliberately different strings here; if the
    expert built allow_contact_with from the track_id instead of the body
    name, check_contacts would reject the (very real, expected) contact with
    the held object's body and the transfer would raise MotionError."""
    target = target_obj()
    scene = ObjectScene(0.0, (target,))
    # A contact with the target's *body* is present from "descend" onward
    # (indices 1-5: descend, lift, preplace, place, retreat), mirroring an
    # arm that is now genuinely touching/holding the object.
    holding_contact = (Contact("left", "panda_leftfinger", TARGET_BODY, 0.05),)
    contacts_by_index = {i: holding_contact for i in range(1, 6)}
    port = FakePort(contacts_by_index=contacts_by_index)
    expert = make_expert(port, lambda: scene)
    context = TaskContext(instruction=DEFAULT_SPEC.instruction, target_track_id="target")

    record = expert.transfer(context)  # must not raise
    assert record.compacted is False


def test_missing_body_names_entry_raises_expert_error_naming_the_track_id():
    target = target_obj()
    scene = ObjectScene(0.0, (target,))
    port = FakePort()
    expert = make_expert(port, lambda: scene, body_names={})
    context = TaskContext(instruction=DEFAULT_SPEC.instruction, target_track_id="target")

    with pytest.raises(ExpertError, match="target"):
        expert.transfer(context)
    assert port.plan_calls == []


def test_unreachable_placement_raises_expert_error():
    """A shelf pushed far enough toward the right that even the no-neighbour
    starting placement lands inside the left (placing) arm's measured
    cross-body-unreachable region."""
    cross_body_spec = ShelfRestockSpec(
        lower_shelf=DEFAULT_SPEC.lower_shelf,
        upper_shelf=ShelfRegion("upper_shelf", (0.3, -0.02, 0.92), (0.3, 0.06, 0.015)),
        object_variation=DEFAULT_SPEC.object_variation,
    )
    target = target_obj()
    scene = ObjectScene(0.0, (target,))
    port = FakePort()
    expert = make_expert(port, lambda: scene, spec=cross_body_spec)
    context = TaskContext(instruction=DEFAULT_SPEC.instruction, target_track_id="target")

    with pytest.raises(ExpertError, match="left"):
        expert.transfer(context)


def test_absent_target_raises_expert_error_naming_the_track_id():
    scene = ObjectScene(0.0, ())
    port = FakePort()
    expert = make_expert(port, lambda: scene)
    context = TaskContext(instruction=DEFAULT_SPEC.instruction, target_track_id="ghost")

    with pytest.raises(ExpertError, match="ghost"):
        expert.transfer(context)


def test_no_fitting_grasp_raises_expert_error_naming_the_size():
    target = target_obj(size=(0.2, 0.09, 0.05))
    scene = ObjectScene(0.0, (target,))
    port = FakePort()
    expert = make_expert(port, lambda: scene)
    context = TaskContext(instruction=DEFAULT_SPEC.instruction, target_track_id="target")

    with pytest.raises(ExpertError, match=r"0\.2"):
        expert.transfer(context)


class RecordingManager(ShelfRestockManager):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.recorded: list[str] = []

    def record_placement(self, track_id: str) -> None:
        self.recorded.append(track_id)
        super().record_placement(track_id)


def test_run_drives_the_manager_until_none_and_records_one_placement_per_transfer():
    obj_a_lower = target_obj("obj_a", position=(-0.1, -0.25, 0.78))
    obj_b_lower = target_obj("obj_b", position=(0.1, -0.25, 0.78))
    obj_a_placed = target_obj("obj_a", position=(0.0, -0.06, 0.96))
    obj_b_placed = target_obj("obj_b", position=(0.25, -0.06, 0.96))
    obj_b_final = target_obj("obj_b", position=(0.2, -0.06, 0.96))

    scene1 = ObjectScene(0.0, (obj_a_lower, obj_b_lower))
    scene2 = ObjectScene(0.0, (obj_a_placed, obj_b_lower))
    scene3 = ObjectScene(0.0, (obj_a_placed, obj_b_placed))
    scene4 = ObjectScene(0.0, (obj_a_placed, obj_b_final))
    observe = QueuedObserve([scene1, scene1, scene2, scene2, scene3, scene4])

    port = FakePort()
    body_names = {"obj_a": "body_a", "obj_b": "body_b"}
    expert = make_expert(port, observe, body_names=body_names)
    manager = RecordingManager(DEFAULT_SPEC)

    records = expert.run(manager)

    assert [r.context.target_track_id for r in records] == ["obj_a", "obj_b"]
    assert manager.recorded == ["obj_a", "obj_b"]
    assert records[0].compacted is False
    assert records[1].compacted is True


def test_run_propagates_a_motion_error_rather_than_returning_a_short_episode():
    target = target_obj("obj_a")
    scene = ObjectScene(0.0, (target,))
    observe = QueuedObserve([scene])
    # "pregrasp" is the first move_to call (index 0); an unexcused contact
    # there must blow up the whole run, not just be swallowed as a failed step.
    port = FakePort(contacts_by_index={0: (Contact("left", "panda_link6", "upper_shelf", 0.9),)})
    body_names = {"obj_a": "body_a"}
    expert = make_expert(port, observe, body_names=body_names)
    manager = ShelfRestockManager(DEFAULT_SPEC)

    with pytest.raises(MotionError):
        expert.run(manager)
