# Coordinate frames

`core/frames.py` defines `Pose`, `Transform`, and `WORKCELL_FRAME = "workcell"`.
All values are immutable; sequence inputs are copied into finite float tuples.
The core uses only the standard library and runs in Python 3.10–3.12.

Positions and translations use meters in a right-handed frame. Orientations
use active unit quaternions in **xyzw** order, mapping tool/object coordinates
into the pose's reference frame. A `Pose` carries that reference frame explicitly.
Canonical `ArmState` accepts only workcell poses; backend adapters must convert
their world/base measurements before creating canonical EEF states.

The workcell is attached to the shelf/workspace. Its physical origin and axis
alignment must be supplied by scene geometry or hardware calibration; this core
does not assume the simulator's world origin equals the workcell origin.

## Transform direction and composition

`Transform(source="base", target="workcell", ...)` is `T_workcell_base`:

```text
p_workcell = R_workcell_base p_base + t_workcell_base
R_workcell_tool = R_workcell_base R_base_tool
```

`apply_point` includes translation. `apply_vector` rotates free vectors without
translation, including Cartesian position increments and spatial rotation
vectors. `apply_pose` checks its input frame and returns a pose in the target
frame. `inverse` reverses the transform.

`a.compose(b)` applies b first, then a. It requires `b.target == a.source`.
This makes accidental base/world/workcell chain mismatches explicit errors.
Point/vector arguments themselves have no frame tag: callers must supply them
in the transform's source frame. Use labelled poses whenever possible.

## Rotation conventions

`geometry.exp(dr)` maps a rotation vector to a quaternion.
`geometry.log(q)` returns the principal rotation vector with norm in [0, pi].
`multiply(a, b)` applies b first, then a. Increment composition is spatial:

```text
dr = Log(R_next R_current^T)
R_next = Exp(dr) R_current
```

Rotation vectors are not Euler angles. To change the reference frame of a
spatial increment, rotate it using `Transform.apply_vector`, just like dp.
There is no translation cross-term here: these are finite differences of the
same EEF origin, not an SE(3) body-twist representation.

Input quaternion norms must be within 1e-6 of one. Accepted values are normalized
to remove numerical drift; zero, nonfinite, and substantially non-unit inputs
are rejected. Quaternions are sign-canonicalized by positive w; at exactly w=0,
the first nonzero xyz component is positive. Thus q and -q represent the same
stored rotation. The principal logarithm necessarily has an axis discontinuity
at pi; compare physical rotations, not rotation-vector components across that
boundary. Pose/action round trips recover orientation, not quaternion sign or
rotation winding beyond pi.

These operations do not establish physical calibration accuracy, reachability,
collision clearance, or robot tracking performance.
