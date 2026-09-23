"""Where a transfer fails, from simulator ground truth.

The scores the server reports -- lifted, transferred, success -- say *that* the
lift-to-transfer stage is the weakest in every arm, not *why*. The candidate
explanations call for different fixes: an object that slips out of the gripper
in transit points at gripper timing (a classification head); one carried
correctly but never brought over the upper shelf points at the reach, where a
gripper head would change nothing; one held over the shelf and never let go
points at the release. This tracker tells them apart.

Diagnostic only. It reads the same ground-truth poses the scorer reads and is
never shown to the policy. Pure Python, so it is tested without a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from oct_vla.core.objects import ObjectScene
from oct_vla.core.state import EEFState
from oct_vla.tasks.shelf_restock.manager import LIFT_CLEARANCE
from oct_vla.tasks.shelf_restock.spec import ShelfRestockSpec

#: An object's centre within this distance of either end-effector pose counts
#: as in hand. The reported pose is not the fingertip: measured over 60 oracle
#: clips, an object in the air and on neither shelf sits 0.140-0.152 m from the
#: nearest end-effector (643 frames), while resting objects sit at a median of
#: 0.24-0.40 m. So "near" means within a few centimetres of that offset.
HOLD_RADIUS = 0.17

#: Per-object outcomes, in the order a transfer progresses. Each names the last
#: stage the object reached and how it left it.
OUTCOMES = (
    "never_lifted",
    "dropped_before_shelf",
    "carried_not_reached",
    "reached_not_placed_held",
    "reached_not_placed_dropped",
    "placed_then_lost",
    "placed",
)


@dataclass
class _Track:
    lifted_step: int | None = None
    over_upper_step: int | None = None
    placed_step: int | None = None
    #: The first step an object that had been lifted was no longer lifted and
    #: not on the upper shelf either -- a drop.
    dropped_step: int | None = None
    #: Steps airborne -- above the lift height and not resting on the upper
    #: deck -- and how many of those in hand.
    held_steps: int = 0
    lifted_steps: int = 0
    final: dict = field(default_factory=dict)


def _over_upper(position, spec: ShelfRestockSpec) -> bool:
    """Above the upper deck's footprint, at or above its surface -- where an
    object has to be before it can be put down there."""
    shelf = spec.upper_shelf
    cx, cy, _ = shelf.center_xyz
    hx, hy, _ = shelf.half_extent_xyz
    x, y, z = position
    return abs(x - cx) <= hx and abs(y - cy) <= hy and z >= shelf.top_z - shelf.occupancy_tolerance


def _grip_distance(position, eef: EEFState) -> float:
    return min(
        sum((a - b) ** 2 for a, b in zip(position, arm.pose.position, strict=True)) ** 0.5
        for arm in (eef.left, eef.right)
    )


class TransferEvents:
    """Accumulates, per object, the first step each transfer stage was reached."""

    def __init__(self, spec: ShelfRestockSpec) -> None:
        self._spec = spec
        self._tracks: dict[str, _Track] = {}

    def update(self, step: int, scene: ObjectScene, eef: EEFState) -> None:
        spec = self._spec
        lift_height = spec.lower_shelf.top_z + LIFT_CLEARANCE
        for obj in scene.objects:
            track = self._tracks.setdefault(obj.track_id, _Track())
            position = obj.pose.position
            lifted = position[2] > lift_height
            placed = spec.upper_shelf.contains(position)
            distance = _grip_distance(position, eef)
            held = distance < HOLD_RADIUS
            if lifted and not placed:
                track.lifted_steps += 1
                track.held_steps += held
            if lifted:
                if track.lifted_step is None:
                    track.lifted_step = step
            if track.over_upper_step is None and _over_upper(position, spec):
                track.over_upper_step = step
            if track.placed_step is None and placed:
                track.placed_step = step
            if (
                track.lifted_step is not None
                and track.dropped_step is None
                and not lifted
                and not placed
            ):
                track.dropped_step = step
            track.final = {
                "lifted": lifted,
                "placed": placed,
                "on_lower": spec.lower_shelf.contains(position),
                "held": held,
                "grip_distance": round(distance, 4),
                "z": round(position[2], 4),
            }

    def outcome(self, track_id: str) -> str:
        track = self._tracks[track_id]
        final = track.final
        if final["placed"]:
            return "placed"
        if track.placed_step is not None:
            return "placed_then_lost"
        if track.lifted_step is None:
            return "never_lifted"
        if track.over_upper_step is not None:
            return "reached_not_placed_held" if final["lifted"] else "reached_not_placed_dropped"
        return "carried_not_reached" if final["lifted"] else "dropped_before_shelf"

    def report(self) -> dict[str, dict]:
        """One entry per object: its outcome, the step each stage was first
        reached, and where it ended up."""
        return {
            track_id: {
                "outcome": self.outcome(track_id),
                "lifted_step": track.lifted_step,
                "over_upper_step": track.over_upper_step,
                "placed_step": track.placed_step,
                "dropped_step": track.dropped_step,
                # Of the steps it spent in the air, the share it spent in hand:
                # low means it was airborne but not carried (knocked, tossed).
                "held_fraction": round(track.held_steps / track.lifted_steps, 3)
                if track.lifted_steps
                else None,
                "final": track.final,
            }
            for track_id, track in sorted(self._tracks.items())
        }
