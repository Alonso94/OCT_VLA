"""Role-based arm assignment for the shelf-restock oracle.

Which arm the expert uses is an oracle concern, not a policy-facing one: the
canonical action carries both arms every step, and the policy learns which to
move (the stationary arm holds). So arm choice lives here, not in
`TaskContext`.

Arms have fixed roles rather than being picked per target: the **left** arm
grasps and places, the **right** arm only compacts. A fixed split keeps each
arm's job identical across episodes, which is what makes the demonstrations
learnable, and it removes the freedom that let both arms end up over the same
narrow shelf at once.

It does mean the left arm now places across the whole deck, including the
`x >= CROSS_BODY_X` region its own cross-body limit covers -- the upper
shelf sits at `y = -0.02`, inside the limited band. `move_to` checks
`is_cross_body_limited` and raises with a kinematic explanation, so such a
placement fails loudly instead of as an opaque planner `Fail`. Grasping is
unaffected: lower-shelf objects sit at `y` around -0.25, well below
`CROSS_BODY_Y`, so the left arm reaches the full width there.

The limit itself was measured because cross-body reaches are unplannable
in part of this workcell. Direct planner queries over a 90-pose grid (both
arms; x in -0.15/0.0/0.15, y in -0.20..0.00, z in 0.85/0.95/1.05) returned
`Fail` for exactly one symmetric region: the left arm could not plan to
`x=+0.15` at `y` of -0.05 or 0.00 (at every height tried), and the right arm
could not plan to `x=-0.15` at those same `y`. Every same-side pose, and
every pose on the `x=0` centreline, planned successfully -- and the result
was identical at `num_trajopt_seeds` of 1 and 4, so this is a kinematic
limit, not planner flakiness (docs/architecture.md (at tag stageA-2026-09-23)).

The measured boundary is only known to lie between the sampled points:
failures start somewhere in `y` between -0.10 (success) and -0.05 (failure),
and in `x` between 0.0 (success) and 0.15 (failure). The constants below
record where failure was actually observed, not an interpolated edge.
"""

from typing import Literal

from oct_vla.core.geometry import finite_values

Side = Literal["left", "right"]
Role = Literal["grasp", "place", "compact"]

#: Smallest |x| at which a cross-body reach was observed to fail.
CROSS_BODY_X = 0.15
#: Smallest y at which those cross-body failures were observed.
CROSS_BODY_Y = -0.05

#: Fixed role split. The left arm carries an object from the lower shelf to
#: the upper one; the right arm only pushes an already-placed object toward
#: its neighbour.
ROLE_ARMS: dict[str, Side] = {"grasp": "left", "place": "left", "compact": "right"}


def arm_for(role: Role) -> Side:
    """The arm that performs `role`."""
    try:
        return ROLE_ARMS[role]
    except KeyError:
        raise ValueError(f"role must be one of {sorted(ROLE_ARMS)}; got {role!r}") from None


def is_cross_body_limited(side: Side, position_xyz: tuple[float, float, float]) -> bool:
    """Whether this target falls in the arm's measured unreachable region.

    With arms assigned by role rather than by which side the target is on,
    this is a real possibility rather than a defensive check, so it exists to
    fail loudly with a kinematic explanation instead of surfacing as an
    opaque planner `Fail`.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
    x, y, _ = finite_values(position_xyz, 3)
    if y < CROSS_BODY_Y:
        return False
    return x >= CROSS_BODY_X if side == "left" else x <= -CROSS_BODY_X
