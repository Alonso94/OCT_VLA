"""Oracle motion primitives: plan once, then execute that plan's trajectory.

Deliberately not routed through `RobotBackend.step()`. That interface takes
bounded per-step Cartesian deltas because a *policy* emits them, and it
re-plans on every step. Driving a long motion through it means repeatedly
re-planning from scratch toward a moving intermediate waypoint -- the "dense
waypoint IK used as trajectory planning" anti-pattern the research plan warns
against, and the thing that drove the arm into configurations whose next
waypoint could not be planned even when both endpoints planned fine
(docs/architecture.md).

The oracle is simulation-only and has no reason to imitate the policy's
interface: it asks cuRobo for one trajectory and executes the interpolated
trajectory that plan produced, which is what the plan was computed for.

These primitives are written against the `NativePort` protocol, not against
SAPIEN, so they are unit-testable with a fake port.
"""

from collections.abc import Collection
from dataclasses import dataclass

from oct_vla.core.frames import Pose, Transform
from oct_vla.robots.robotwin.backend import NativePort, encode_pose
from oct_vla.tasks.shelf_restock.oracle.arms import Side, is_cross_body_limited

ZERO_VELOCITY = (0.0,) * 7
SETTLE_TICKS = 50
GRIPPER_TICKS = 60


class MotionError(RuntimeError):
    """Raised instead of silently continuing when a motion cannot be executed."""


@dataclass(frozen=True)
class ExecutedMotion:
    """What was actually commanded, tick by tick, for the expert plan record."""

    side: Side
    q: tuple[tuple[float, ...], ...]
    qdot: tuple[tuple[float, ...], ...]
    gripper: tuple[float, ...]

    @property
    def ticks(self) -> int:
        return len(self.q)


def _other(side: Side) -> Side:
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
    return "right" if side == "left" else "left"


def _arm(reading, side: Side):
    return reading.left if side == "left" else reading.right


def check_contacts(port: NativePort, allow_contact_with: Collection[str] = ()) -> None:
    """Raise unless every current robot contact is with an allowed body.

    A cuRobo `Success` is not evidence of a collision-free trajectory --
    trajopt collision avoidance is a soft cost, and executing a "successful"
    plan has been observed to leave the wrist jammed against the shelf
    (docs/architecture.md). Checking afterwards is the only hard guarantee.

    This inspects the state at the moment it is called, so it catches
    sustained contact (a jam) but not a transient brush that has already
    separated.
    """
    allowed = set(allow_contact_with)
    offending = [contact for contact in port.contacts() if contact.other not in allowed]
    if not offending:
        return
    worst = max(offending, key=lambda contact: contact.impulse)
    raise MotionError(
        f"motion ended in contact: {worst.qualified_link} <-> {worst.other} "
        f"(impulse {worst.impulse:.4f}); {len(offending)} contact(s) not in "
        f"allow_contact_with={sorted(allowed)}"
    )


def move_to(
    port: NativePort,
    side: Side,
    target: Pose,
    world_to_workcell: Transform,
    *,
    gripper: float | None = None,
    settle_ticks: int = SETTLE_TICKS,
    allow_contact_with: Collection[str] = (),
) -> ExecutedMotion:
    """Plan once to `target` (workcell frame) and execute the whole trajectory.

    The other arm holds at the joint positions it had before the motion, with
    zero velocity, for every tick -- the same hold semantics the canonical
    backend uses, so a stationary arm never drifts.

    Raises `MotionError` if the arm ends in contact with anything outside
    `allow_contact_with` (pass the held object's name when carrying one).
    """
    other = _other(side)
    if is_cross_body_limited(side, target.position):
        raise MotionError(
            f"{side} arm cannot reach {tuple(round(v, 3) for v in target.position)}: "
            "measured cross-body limit (see oracle/arms.py)"
        )

    reading = port.read()
    holding = _arm(reading, other)
    if gripper is None:
        gripper = _arm(reading, side).gripper

    world_target = encode_pose(world_to_workcell.inverse().apply_pose(target))
    trajectory = port.plan(side, world_target)

    q_log: list[tuple[float, ...]] = []
    qdot_log: list[tuple[float, ...]] = []
    gripper_log: list[float] = []

    def tick(q: tuple[float, ...], qdot: tuple[float, ...]) -> None:
        port.command(side, q, qdot, gripper)
        port.command(other, holding.joints, ZERO_VELOCITY, holding.gripper)
        port.tick()
        q_log.append(tuple(q))
        qdot_log.append(tuple(qdot))
        gripper_log.append(gripper)

    for q, qdot in zip(trajectory.q, trajectory.qdot, strict=True):
        tick(q, qdot)
    for _ in range(settle_ticks):
        tick(trajectory.q[-1], ZERO_VELOCITY)

    check_contacts(port, allow_contact_with)
    return ExecutedMotion(side, tuple(q_log), tuple(qdot_log), tuple(gripper_log))


def set_gripper(
    port: NativePort, side: Side, target: float, *, ticks: int = GRIPPER_TICKS
) -> ExecutedMotion:
    """Ramp one gripper to `target` while both arms hold their joint positions.

    Ramped rather than commanded in one step: RoboTwin's own `set_gripper`
    rate-limits how far a drive target may move per call, so a single large
    command would be silently truncated instead of reaching `target`.

    Deliberately does not check contacts: closing on an object *is* contact.
    Call `check_contacts` explicitly afterwards, passing the object being
    grasped, if you want to assert nothing else is being touched.
    """
    if ticks <= 0:
        raise ValueError("ticks must be positive")
    other = _other(side)
    reading = port.read()
    moving, holding = _arm(reading, side), _arm(reading, other)
    start = moving.gripper

    q_log: list[tuple[float, ...]] = []
    qdot_log: list[tuple[float, ...]] = []
    gripper_log: list[float] = []
    for index in range(ticks):
        value = start + (target - start) * (index + 1) / ticks
        port.command(side, moving.joints, ZERO_VELOCITY, value)
        port.command(other, holding.joints, ZERO_VELOCITY, holding.gripper)
        port.tick()
        q_log.append(tuple(moving.joints))
        qdot_log.append(ZERO_VELOCITY)
        gripper_log.append(value)

    return ExecutedMotion(side, tuple(q_log), tuple(qdot_log), tuple(gripper_log))
