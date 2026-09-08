"""Side-aware arm selection for the shelf-restock oracle.

Which arm the expert uses is an oracle concern, not a policy-facing one: the
canonical action carries both arms every step, and the policy learns which to
move (the stationary arm holds). So arm choice lives here, not in
`TaskContext`.

Selection is same-side because cross-body reaches are measurably unplannable
in part of this workcell. Direct planner queries over a 90-pose grid (both
arms; x in -0.15/0.0/0.15, y in -0.20..0.00, z in 0.85/0.95/1.05) returned
`Fail` for exactly one symmetric region: the left arm could not plan to
`x=+0.15` at `y` of -0.05 or 0.00 (at every height tried), and the right arm
could not plan to `x=-0.15` at those same `y`. Every same-side pose, and
every pose on the `x=0` centreline, planned successfully -- and the result
was identical at `num_trajopt_seeds` of 1 and 4, so this is a kinematic
limit, not planner flakiness (docs/architecture.md).

The measured boundary is only known to lie between the sampled points:
failures start somewhere in `y` between -0.10 (success) and -0.05 (failure),
and in `x` between 0.0 (success) and 0.15 (failure). The constants below
record where failure was actually observed, not an interpolated edge.
"""

from typing import Literal

from oct_vla.core.geometry import finite_values

Side = Literal["left", "right"]

#: Smallest |x| at which a cross-body reach was observed to fail.
CROSS_BODY_X = 0.15
#: Smallest y at which those cross-body failures were observed.
CROSS_BODY_Y = -0.05


def select_arm(position_xyz: tuple[float, float, float]) -> Side:
    """The arm on the same side of the workcell midline as the target.

    Ties at `x == 0` go to the left arm; both arms plan centreline poses
    successfully, so the choice is arbitrary but must stay deterministic.
    """
    x, _, _ = finite_values(position_xyz, 3)
    return "right" if x > 0.0 else "left"


def is_cross_body_limited(side: Side, position_xyz: tuple[float, float, float]) -> bool:
    """Whether this target falls in the arm's measured unreachable region.

    `select_arm` never produces such a pairing; this exists so code that
    chooses an arm for other reasons can fail loudly with a kinematic
    explanation instead of discovering it as an opaque planner `Fail`.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right'; got {side!r}")
    x, y, _ = finite_values(position_xyz, 3)
    if y < CROSS_BODY_Y:
        return False
    return x >= CROSS_BODY_X if side == "left" else x <= -CROSS_BODY_X
