# Canonical action and EEF state

`core/action.py` owns the sole learned action layout (`ACTION_DIM = 14`):

| Indices | Meaning | Units |
| --- | --- | --- |
| 0–2 | Left EEF position increment | meters per policy step |
| 3–5 | Left spatial rotation vector increment | radians per policy step |
| 6 | Left absolute gripper target | normalized [0, 1] |
| 7–9 | Right EEF position increment | meters per policy step |
| 10–12 | Right spatial rotation vector increment | radians per policy step |
| 13 | Right absolute gripper target | normalized [0, 1] |

All pose increments are expressed in the workcell frame. Gripper 0 means closed
and 1 means open. These are per-step displacements, not velocities; do not
multiply them by a control timestep at execution. Backend calibration determines
the physical gripper mapping.

`Action` contains left and right immutable `ArmAction` values. `from_vector`
rejects wrong lengths, nested shapes, nonfinite values, and out-of-range grippers.
`to_vector` returns a flat tuple suitable for JSON encoding or conversion to a
library tensor at an adapter boundary. Core construction does not clip values.

`EEFState` contains two `ArmState` values: workcell `Pose` plus normalized gripper.
It does not contain joint targets. Timestamps belong to future observation and
episode containers; no timestamp or 15-Hz sampling claim is implied by this type.

```python
from oct_vla.core.action import Action, action_between, apply_action
from oct_vla.core.frames import Pose
from oct_vla.core.state import ArmState, EEFState

pose = Pose((0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0))
current = EEFState(ArmState(pose, 0.3), ArmState(pose, 0.8))
left = [0.01, 0, 0, 0, 0, 0, 0.3]
right = [0, 0, 0, 0, 0, 0, 0.8]
command = Action.from_vector(left + right)
following = apply_action(current, command)
reconstructed = action_between(current, following)
hold = Action.hold(following)
```

`apply_action` adds translation, left-composes rotation, and replaces the gripper
target. `action_between` computes `p_next - p_current` and
`Log(R_next R_current^T)`, retaining the next absolute gripper values. Training
conversion and deployment should call these same helpers. This implementation
handles one dual-arm step; batch/resampling adapters follow with dataset work.

`Action.hold(state)` retains both grippers with zero pose increments. An all-zero
14-D vector instead commands both grippers closed. To move only one arm, preserve
the other arm's hold command. No global active-arm mask is part of the contract.

The core validates representation, not robot feasibility. Workspace limits,
per-step displacement bounds, collision checking, interpolation, and tracking
belong to backend execution. Legacy commanded EEF targets and measured next
poses remain distinct data sources; conversion must explicitly identify which
is being used. See [coordinate conventions](coordinate_frames.md).
