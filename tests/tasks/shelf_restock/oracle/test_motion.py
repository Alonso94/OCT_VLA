from math import pi

import pytest

from oct_vla.core.frames import WORKCELL_FRAME, Pose, Transform
from oct_vla.core.geometry import exp
from oct_vla.core.observation import RGBFrame
from oct_vla.robots.robotwin.backend import ArmReading, Contact, Reading, Trajectory
from oct_vla.tasks.shelf_restock.oracle.motion import (
    ZERO_VELOCITY,
    ExecutedMotion,
    MotionError,
    check_contacts,
    move_to,
    set_gripper,
)

# Nontrivial transform, as in the backend tests, so nothing passes by accident
# on an identity mapping between world and workcell.
WORLD_TO_WORKCELL = Transform("world", WORKCELL_FRAME, (0.0, 0.0, 0.74), exp((0.0, 0.0, pi / 2)))

LEFT_JOINTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
RIGHT_JOINTS = (-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7)


def _frame() -> RGBFrame:
    return RGBFrame(1, 1, bytes(3))


class FakePort:
    dt = 1.0 / 250.0

    def __init__(
        self, rows: int = 3, fail_plan: bool = False, contacts: tuple[Contact, ...] = ()
    ) -> None:
        self.plan_calls: list[tuple[str, tuple[float, ...]]] = []
        self.commands: list[tuple[str, tuple[float, ...], tuple[float, ...], float]] = []
        self.ticks = 0
        self.rows = rows
        self.fail_plan = fail_plan
        self._contacts = contacts

    def contacts(self) -> tuple[Contact, ...]:
        return self._contacts

    def reset(self, seed: int) -> None: ...

    def read(self) -> Reading:
        left = ArmReading((0.4, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0), 0.5, LEFT_JOINTS)
        right = ArmReading((0.4, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0), 0.9, RIGHT_JOINTS)
        return Reading(left, right, (_frame(), _frame(), _frame()))

    def plan(self, side: str, pose_wxyz: tuple[float, ...]) -> Trajectory:
        self.plan_calls.append((side, tuple(pose_wxyz)))
        if self.fail_plan:
            raise RuntimeError("plan failed")
        q = tuple(tuple(float(i) + j for j in range(7)) for i in range(self.rows))
        qdot = tuple(tuple(0.5 for _ in range(7)) for _ in range(self.rows))
        return Trajectory(q, qdot)

    def command(self, side, q, qdot, gripper) -> None:
        self.commands.append((side, tuple(q), tuple(qdot), gripper))

    def tick(self) -> None:
        self.ticks += 1

    def hold(self) -> None: ...

    def close(self) -> None: ...


def target(position=(0.0, -0.05, 0.95)) -> Pose:
    return Pose(position, (0.0, 0.0, 0.0, 1.0), WORKCELL_FRAME)


def test_plans_exactly_once_for_the_whole_motion():
    port = FakePort(rows=5)
    move_to(port, "left", target(), WORLD_TO_WORKCELL, settle_ticks=3)
    assert len(port.plan_calls) == 1
    assert port.plan_calls[0][0] == "left"


def test_executes_every_planned_row_in_order_then_settles_on_the_last():
    port = FakePort(rows=4)
    motion = move_to(port, "left", target(), WORLD_TO_WORKCELL, settle_ticks=3)

    assert motion.ticks == 4 + 3
    assert port.ticks == 4 + 3
    moving = [c for c in port.commands if c[0] == "left"]
    assert [q for _, q, _, _ in moving[:4]] == [
        tuple(float(i) + j for j in range(7)) for i in range(4)
    ]
    # Settle ticks hold the final planned row with zero velocity.
    for _, q, qdot, _ in moving[4:]:
        assert q == tuple(3.0 + j for j in range(7))
        assert qdot == ZERO_VELOCITY


def test_other_arm_holds_its_pre_motion_joints_every_tick():
    port = FakePort(rows=3)
    move_to(port, "left", target(), WORLD_TO_WORKCELL, settle_ticks=2)

    holding = [c for c in port.commands if c[0] == "right"]
    assert len(holding) == 5
    for _, q, qdot, gripper in holding:
        assert q == RIGHT_JOINTS
        assert qdot == ZERO_VELOCITY
        assert gripper == 0.9


def test_gripper_defaults_to_the_moving_arm_s_current_value_and_is_held():
    port = FakePort(rows=2)
    motion = move_to(port, "left", target(), WORLD_TO_WORKCELL, settle_ticks=1)
    assert set(motion.gripper) == {0.5}


def test_explicit_gripper_overrides_the_current_value():
    port = FakePort(rows=2)
    motion = move_to(port, "left", target(), WORLD_TO_WORKCELL, gripper=0.0, settle_ticks=1)
    assert set(motion.gripper) == {0.0}


def test_target_is_converted_out_of_the_workcell_frame_before_planning():
    port = FakePort(rows=1)
    move_to(port, "left", target((0.0, 0.0, 0.74)), WORLD_TO_WORKCELL, settle_ticks=0)
    (_, pose_wxyz) = port.plan_calls[0]
    # Workcell (0,0,0.74) maps back to world (0,0,0) under this transform.
    assert pose_wxyz[:3] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_plan_failure_propagates_and_commands_nothing():
    port = FakePort(fail_plan=True)
    with pytest.raises(RuntimeError, match="plan failed"):
        move_to(port, "left", target(), WORLD_TO_WORKCELL)
    assert port.commands == []
    assert port.ticks == 0


def test_cross_body_target_is_refused_before_planning():
    port = FakePort()
    with pytest.raises(MotionError, match="cross-body"):
        move_to(port, "left", target((0.15, 0.0, 0.95)), WORLD_TO_WORKCELL)
    assert port.plan_calls == []
    assert port.commands == []


def test_rejects_an_invalid_side():
    with pytest.raises(ValueError):
        move_to(FakePort(), "middle", target(), WORLD_TO_WORKCELL)


def test_set_gripper_ramps_to_the_target_while_arms_hold():
    port = FakePort()
    motion = set_gripper(port, "left", 0.0, ticks=5)

    assert motion.ticks == 5
    assert motion.gripper[-1] == pytest.approx(0.0)
    assert motion.gripper[0] == pytest.approx(0.4)  # from 0.5, one fifth of the way
    assert all(g1 >= g2 for g1, g2 in zip(motion.gripper[:-1], motion.gripper[1:], strict=True))
    for _, q, qdot, _ in (c for c in port.commands if c[0] == "left"):
        assert q == LEFT_JOINTS
        assert qdot == ZERO_VELOCITY


def test_set_gripper_holds_the_other_arm_untouched():
    port = FakePort()
    set_gripper(port, "left", 1.0, ticks=3)
    holding = [c for c in port.commands if c[0] == "right"]
    assert len(holding) == 3
    assert all(gripper == 0.9 and q == RIGHT_JOINTS for _, q, _, gripper in holding)


def test_set_gripper_rejects_nonpositive_ticks():
    with pytest.raises(ValueError):
        set_gripper(FakePort(), "left", 0.0, ticks=0)


def test_executed_motion_ticks_matches_logged_rows():
    motion = ExecutedMotion("left", ((0.0,) * 7,) * 3, ((0.0,) * 7,) * 3, (0.5, 0.5, 0.5))
    assert motion.ticks == 3


def test_move_to_raises_when_it_ends_in_unexpected_contact():
    port = FakePort(rows=2, contacts=(Contact("left", "panda_link6", "upper_shelf", 0.52),))
    with pytest.raises(MotionError, match="upper_shelf"):
        move_to(port, "left", target(), WORLD_TO_WORKCELL, settle_ticks=1)


def test_move_to_accepts_contact_with_an_allowed_body():
    port = FakePort(
        rows=2, contacts=(Contact("left", "panda_leftfinger", "restock_object_0", 0.11),)
    )
    motion = move_to(
        port,
        "left",
        target(),
        WORLD_TO_WORKCELL,
        settle_ticks=1,
        allow_contact_with=("restock_object_0",),
    )
    assert motion.ticks == 3


def test_move_to_reports_the_worst_contact_and_the_count():
    port = FakePort(
        rows=1,
        contacts=(
            Contact("left", "panda_link7", "upper_shelf", 0.26),
            Contact("left", "panda_link6", "upper_shelf", 0.52),
        ),
    )
    with pytest.raises(MotionError, match=r"left/panda_link6 <-> upper_shelf .*0\.5200.*2 contact"):
        move_to(port, "left", target(), WORLD_TO_WORKCELL, settle_ticks=0)


def test_check_contacts_passes_when_the_robot_touches_nothing():
    check_contacts(FakePort())


def test_set_gripper_does_not_check_contacts_since_closing_is_contact():
    port = FakePort(contacts=(Contact("left", "panda_leftfinger", "restock_object_0", 0.11),))
    motion = set_gripper(port, "left", 0.0, ticks=2)
    assert motion.ticks == 2


def test_a_collision_with_the_other_arm_is_not_mistaken_for_a_self_collision():
    """Both arms load the same Panda URDF, so link names alone cannot say
    which arm a link is on; the other arm must be reported qualified."""
    port = FakePort(
        rows=1, contacts=(Contact("right", "panda_rightfinger", "left/panda_hand", 0.008),)
    )
    with pytest.raises(MotionError, match="right/panda_rightfinger <-> left/panda_hand"):
        move_to(port, "right", target(), WORLD_TO_WORKCELL, settle_ticks=0)


def test_carrying_an_object_does_not_excuse_hitting_the_other_arm():
    port = FakePort(
        rows=1,
        contacts=(
            Contact("right", "panda_leftfinger", "restock_object_1", 0.11),
            Contact("right", "panda_rightfinger", "left/panda_hand", 0.008),
        ),
    )
    with pytest.raises(MotionError, match="left/panda_hand"):
        move_to(
            port,
            "right",
            target(),
            WORLD_TO_WORKCELL,
            settle_ticks=0,
            allow_contact_with=("restock_object_1",),
        )
