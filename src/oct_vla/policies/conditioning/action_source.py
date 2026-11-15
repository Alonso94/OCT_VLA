"""Kinematic bridge: an interaction keyframe -> a coarse eeabs action chunk.

PFN-informed conditional-source flow (paper/5_time_row_pfn_design.md; plan rev. 8).
The keyframe (oracle or prior; ``data/future_keyframes.py``) names where the moving
gripper must go next. This module turns it, with the grippers' current poses, into a
40-step chunk in the eeabs action encoding, which the flow then starts from
(``source_flow.py``). Everything is deterministic and in raw workcell units; the same
function serves training, rollouts and the offline diagnostic, so the three cannot
disagree.

**eeabs action** (``data/lerobot_export.py``): 16 columns, left then right arm, each
``x, y, z, qx, qy, qz, qw, gripper``; row k is the commanded pose for frame t+k+1;
the gripper column is binary (1 open, 0 closed); quaternions have w >= 0.

**The proposal.**
- The **mover** is the arm whose gripper type the keyframe's gripper row carries. It
  moves in a straight line at a constant ``speed``: current -> approach waypoint (row 3,
  when present and unmasked) -> event pose (row 1), its rotation slerped from current
  to the event's over the whole path. Its gripper keeps its current binary state until
  it arrives, then takes the event's: closed for grasp/regrasp, open for place/release.
- **Goal-only rows** (the gripper row masked, the object row present) get an event pose
  from the oracle's own grasp geometry (``oracle/grasps.holding_tcp_pose``) above the
  object (grasp) or the goal (place); the mover is then given by ``goal_only_mover``.
- The **idle** arm holds its current pose and gripper.
- No keyframe (row 0 masked): ``valid`` is False and the caller uses its fallback
  (pure Gaussian source).
"""

from __future__ import annotations

import math

import torch

from oct_vla.data.entity_tokens import ENTITY_TOKEN_DIM, ENTITY_TYPES
from oct_vla.data.future_keyframes import FUTURE_EVENTS, FUTURE_LAYOUTS

#: Columns of an eeabs action row.
ACTION_DIM = 16
ARM_COLUMNS = {"left": slice(0, 8), "right": slice(8, 16)}
POS, QUAT, GRIP = slice(0, 3), slice(3, 7), 7
#: Measured aperture at or above which the gripper counts as open
#: (core/gripper.GRIPPER_OPEN_THRESHOLD, which the exporter binarizes with).
GRIPPER_OPEN_THRESHOLD = 0.822522
#: Entity-token columns: position, 6D rotation, size, aperture, type one-hot.
TOK_POS, TOK_ROT, TOK_SIZE, TOK_APERTURE = slice(0, 3), slice(3, 9), slice(9, 12), 12
TYPE_START = ENTITY_TOKEN_DIM - len(ENTITY_TYPES)
LEFT_TYPE = TYPE_START + ENTITY_TYPES.index("left_gripper")
RIGHT_TYPE = TYPE_START + ENTITY_TYPES.index("right_gripper")
#: The binary gripper command after each event.
EVENT_GRIPPER = {"grasp": 0.0, "regrasp": 0.0, "place": 1.0, "release": 1.0}
HORIZON = 40
DT = 1.0 / 15.0


def rotation6d_to_quat(r6: torch.Tensor) -> torch.Tensor:
    """[..., 6] (the first two rotation-matrix columns, entity_tokens._rotation6) ->
    [..., 4] xyzw unit quaternion with w >= 0 (core/geometry.unit_quaternion's sign)."""
    a, b = r6[..., 0:3], r6[..., 3:6]
    c0 = torch.nn.functional.normalize(a, dim=-1)
    b = b - (c0 * b).sum(-1, keepdim=True) * c0
    c1 = torch.nn.functional.normalize(b, dim=-1)
    c2 = torch.cross(c0, c1, dim=-1)
    m = torch.stack([c0, c1, c2], -1)  # columns -> [..., 3, 3], m[..., row, col]
    return _matrix_to_quat(m)


def _matrix_to_quat(m: torch.Tensor) -> torch.Tensor:
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    trace = m00 + m11 + m22
    cands = torch.stack([
        torch.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0],
                     m[..., 1, 0] - m[..., 0, 1], 1 + trace], -1),
        torch.stack([1 + m00 - m11 - m22, m[..., 0, 1] + m[..., 1, 0],
                     m[..., 0, 2] + m[..., 2, 0], m[..., 2, 1] - m[..., 1, 2]], -1),
        torch.stack([m[..., 0, 1] + m[..., 1, 0], 1 - m00 + m11 - m22,
                     m[..., 1, 2] + m[..., 2, 1], m[..., 0, 2] - m[..., 2, 0]], -1),
        torch.stack([m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1],
                     1 - m00 - m11 + m22, m[..., 1, 0] - m[..., 0, 1]], -1),
    ], -2)  # [..., 4 candidates, xyzw]; pick the best-conditioned
    choice = torch.stack([trace, m00, m11, m22], -1).argmax(-1)
    q = cands.gather(-2, choice[..., None, None].expand(*choice.shape, 1, 4))[..., 0, :]
    q = torch.nn.functional.normalize(q, dim=-1)
    return canonical_quat(q)


def canonical_quat(q: torch.Tensor) -> torch.Tensor:
    """The sign rule of core/geometry.unit_quaternion: first non-zero of (w, x, y, z) > 0."""
    order = q[..., [3, 0, 1, 2]]
    nonzero = order.abs() > 1e-12
    first = nonzero.float().argmax(-1, keepdim=True)
    sign = torch.sign(order.gather(-1, first))
    return q * torch.where(sign == 0, torch.ones_like(sign), sign)


def slerp(q0: torch.Tensor, q1: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """[..., 4] x [..., 4] x [...] -> [..., 4], shortest arc, canonical sign."""
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    s = torch.sin(theta)
    f = f[..., None]
    small = s < 1e-6
    w0 = torch.where(small, 1 - f, torch.sin((1 - f) * theta) / s.clamp_min(1e-6))
    w1 = torch.where(small, f, torch.sin(f * theta) / s.clamp_min(1e-6))
    return canonical_quat(torch.nn.functional.normalize(w0 * q0 + w1 * q1, dim=-1))


def gripper_states(tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Current left/right EEF pose and binary gripper from the entity tokens.

    tokens [B, N, 17], mask [B, N] -> {"left": [B, 8], "right": [B, 8]} in the eeabs
    arm layout. Exactly one valid token of each gripper type is required: the gripper
    rows are the same `eef.<side>.pose` the state and action are built from."""
    out = {}
    for side, col in (("left", LEFT_TYPE), ("right", RIGHT_TYPE)):
        is_side = (tokens[..., col] > 0.5) & mask.bool()
        count = is_side.sum(-1)
        if not bool((count == 1).all()):
            raise ValueError(f"every sample needs exactly one {side}_gripper token; got {count}")
        tok = (tokens * is_side[..., None]).sum(1)
        grip = (tok[:, TOK_APERTURE] >= GRIPPER_OPEN_THRESHOLD).to(tokens.dtype)
        out[side] = torch.cat([tok[:, TOK_POS], rotation6d_to_quat(tok[:, TOK_ROT]),
                               grip[:, None]], -1)
    return out


def _layout(dim: int) -> tuple[tuple[str, ...], slice]:
    if dim not in FUTURE_LAYOUTS:
        raise ValueError(f"keyframe rows {dim} wide match no layout {sorted(FUTURE_LAYOUTS)}")
    roles = FUTURE_LAYOUTS[dim]
    start = ENTITY_TOKEN_DIM + len(roles)
    return roles, slice(start, start + len(FUTURE_EVENTS))


def kinematic_proposal(tokens: torch.Tensor, entity_mask: torch.Tensor,
                       keyframe: torch.Tensor, keyframe_mask: torch.Tensor, *,
                       speed: float, horizon: int = HORIZON, dt: float = DT,
                       goal_only_mover: str = "left") -> tuple[torch.Tensor, torch.Tensor]:
    """Raw eeabs chunks [B, horizon, 16] and `valid` [B] (False: no keyframe).

    tokens [B, N, 17] / entity_mask [B, N]: the current scene (grippers included);
    keyframe [B, R, D] / keyframe_mask [B, R]: 3 x 24 or 4 x 25 rows."""
    if speed <= 0:
        raise ValueError("speed must be positive")
    B, R, D = keyframe.shape
    roles, ev_cols = _layout(D)
    if R != len(roles):
        raise ValueError(f"{R} keyframe rows, the {D}-wide layout has {len(roles)}")
    kmask = keyframe_mask.bool()
    current = gripper_states(tokens, entity_mask)
    chunk = torch.cat([current["left"], current["right"]], -1)[:, None].repeat(1, horizon, 1)
    valid = kmask[:, 0].clone()
    event = keyframe[:, 0, ev_cols].argmax(-1)  # [B]
    g = roles.index("gripper_keyframe")
    has_grip = kmask[:, g]
    target_pos = keyframe[:, g, TOK_POS].clone()
    target_q = rotation6d_to_quat(keyframe[:, g, TOK_ROT])
    is_right = keyframe[:, g, RIGHT_TYPE] > 0.5
    goal_only = valid & ~has_grip
    if bool(goal_only.any()):
        pos, q = _goal_only_event_pose(keyframe, roles, event, goal_only)
        target_pos = torch.where(goal_only[:, None], pos, target_pos)
        target_q = torch.where(goal_only[:, None], q, target_q)
        is_right = torch.where(goal_only, torch.full_like(is_right, goal_only_mover == "right"),
                               is_right)
    if "gripper_waypoint" in roles:
        w = roles.index("gripper_waypoint")
        has_wp = kmask[:, w] & has_grip
        waypoint = torch.where(has_wp[:, None], keyframe[:, w, TOK_POS], target_pos)
    else:
        has_wp = torch.zeros_like(valid)
        waypoint = target_pos
    for side, right in (("left", False), ("right", True)):
        cols = ARM_COLUMNS[side]
        mover = valid & (is_right == right)
        if not bool(mover.any()):
            continue
        arm = chunk[:, :, cols]
        start = arm[:, 0, POS]
        seg1 = (waypoint - start).norm(dim=-1)
        seg2 = (target_pos - waypoint).norm(dim=-1)
        total = (seg1 + seg2).clamp_min(1e-9)
        k = torch.arange(1, horizon + 1, device=keyframe.device, dtype=keyframe.dtype)
        s = torch.minimum(speed * dt * k[None], total[:, None])  # [B, H] arclength
        on1 = s <= seg1[:, None]
        f1 = (s / seg1.clamp_min(1e-9)[:, None]).clamp(0, 1)
        f2 = ((s - seg1[:, None]) / seg2.clamp_min(1e-9)[:, None]).clamp(0, 1)
        p1 = start[:, None] + f1[..., None] * (waypoint - start)[:, None]
        p2 = waypoint[:, None] + f2[..., None] * (target_pos - waypoint)[:, None]
        pos = torch.where(on1[..., None], p1, p2)
        quat = slerp(arm[:, :, QUAT], target_q[:, None].expand(-1, horizon, -1), s / total[:, None])
        arrived = s >= total[:, None] - 1e-6
        after = torch.tensor([EVENT_GRIPPER[e] for e in FUTURE_EVENTS],
                             device=keyframe.device, dtype=keyframe.dtype)[event]
        grip = torch.where(arrived, after[:, None], arm[:, :, GRIP])
        moved = torch.cat([pos, quat, grip[..., None]], -1)
        chunk[:, :, cols] = torch.where(mover[:, None, None], moved, arm)
    return chunk, valid


def _goal_only_event_pose(keyframe, roles, event, rows):
    """Event pose from the object (grasp) or goal (place) row, by the oracle's geometry.

    Wrist yaw is the object's yaw; per-sample Python, used only by goal-only sources."""
    from oct_vla.tasks.shelf_restock.oracle.grasps import holding_tcp_pose

    pos = torch.zeros(len(keyframe), 3, dtype=keyframe.dtype)
    quat = torch.zeros(len(keyframe), 4, dtype=keyframe.dtype)
    quat[:, 3] = 1.0
    obj_row, goal_row = roles.index("object_keyframe"), roles.index("object_goal")
    for i in torch.nonzero(rows).flatten().tolist():
        name = FUTURE_EVENTS[int(event[i])]
        row = keyframe[i, goal_row if name in ("place", "release") else obj_row].double()
        yaw = math.atan2(float(row[TOK_ROT][1]), float(row[TOK_ROT][0]))
        pose = holding_tcp_pose(tuple(row[TOK_POS].tolist()), float(row[TOK_SIZE][2]), yaw)
        pos[i] = torch.tensor(pose.position, dtype=keyframe.dtype)
        quat[i] = torch.tensor(pose.orientation, dtype=keyframe.dtype)
    return pos.to(keyframe.device), canonical_quat(quat).to(keyframe.device)


def normalize_actions(raw: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """GR00T's action min-max normalisation (GrootN17PackInputsStep._min_max_norm):
    2 (x - min) / (max - min) - 1, 0 where max == min, clipped to [-1, 1]."""
    span = hi - lo
    out = torch.where(span == 0, torch.zeros_like(raw),
                      2 * (raw - lo) / torch.where(span == 0, torch.ones_like(span), span) - 1)
    return out.clamp(-1, 1)


def via_point_proposal(tokens: torch.Tensor, entity_mask: torch.Tensor,
                       targets: torch.Tensor, target_mask: torch.Tensor, *,
                       speed: float | tuple[float, float], dwell: int = 0,
                       horizon: int = HORIZON, dt: float = DT) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw eeabs chunks [B, horizon, 16] from per-arm via-points, and `valid` [B].

    targets [B, 2, V, 8] (left, right; position, xyzw quaternion, gripper command on
    arrival) with a prefix mask [B, 2, V] (tasks/shelf_restock/arm_targets.py). Each
    arm moves through its own via-points in order at constant speed (`speed`, or one
    per arm), its rotation slerped within each segment toward the via-point's. Where a
    via-point changes the gripper command, the arm switches on arrival and then waits
    `dwell` steps (the expert's close/release takes ~4 frames). A zero-length segment
    (a via-point at the current pose) is reached at once, which is how "close in place"
    is expressed. An arm with no via-points holds. `valid` is False when neither arm
    has one."""
    speeds = (speed, speed) if isinstance(speed, (int, float)) else tuple(speed)
    if min(speeds) <= 0 or dwell < 0:
        raise ValueError("speed must be positive and dwell non-negative")
    B, A, V, C = targets.shape
    if A != 2 or C != 8:
        raise ValueError(f"via-points must be [B, 2, V, 8]; got {tuple(targets.shape)}")
    tmask = target_mask.bool()
    if bool((tmask[..., 1:] & ~tmask[..., :-1]).any()):
        raise ValueError("via-point masks must be prefixes (no gap before a valid point)")
    current = gripper_states(tokens, entity_mask)
    chunk = torch.cat([current["left"], current["right"]], -1)[:, None].repeat(1, horizon, 1)
    valid = tmask.any(-1).any(-1)
    dev, dtype = targets.device, targets.dtype
    k = torch.arange(1, horizon + 1, device=dev, dtype=dtype)
    for a, side in enumerate(("left", "right")):
        cols = ARM_COLUMNS[side]
        n = tmask[:, a].sum(-1)
        if not bool((n > 0).any()):
            continue
        cur = current[side]
        last = (n - 1).clamp_min(0)
        idx = torch.minimum(torch.arange(V, device=dev)[None], last[:, None])
        pts = targets[:, a].gather(1, idx[..., None].expand(B, V, 8))
        # Gripper command before each via-point (the current one, then each previous).
        before = torch.cat([cur[:, None, 7], pts[:, :-1, 7]], 1)
        switch = (pts[..., 7] != before) & tmask[:, a]
        # Segments in time: travel to via k, then a dwell at it (zero motion).
        travel = (pts[..., :3] - torch.cat([cur[:, None, :3], pts[:, :-1, :3]], 1)).norm(dim=-1)
        rate = speeds[a] * dt  # metres per step
        pause = switch.to(dtype) * dwell * rate  # in metres-equivalent of time
        lengths = torch.stack([travel, pause], -1).reshape(B, 2 * V)
        ends = torch.cat([cur[:, None, :7], pts[..., :7]], 1)  # [B, V+1, 7]
        node_from = torch.stack([ends[:, :-1], ends[:, 1:]], 2).reshape(B, 2 * V, 7)
        node_to = torch.stack([ends[:, 1:], ends[:, 1:]], 2).reshape(B, 2 * V, 7)
        cum = torch.cat([torch.zeros(B, 1, dtype=dtype, device=dev), lengths.cumsum(-1)], -1)
        s = torch.minimum(rate * k[None], cum[:, -1:])
        j = torch.searchsorted(cum[:, 1:].contiguous(), s.contiguous()).clamp(max=2 * V - 1)
        start, length = cum.gather(1, j), lengths.gather(1, j)
        f = torch.where(length < 1e-9, torch.ones_like(s), ((s - start) / length.clamp_min(1e-9)))
        f = f.clamp(0, 1)
        p0 = node_from.gather(1, j[..., None].expand(B, horizon, 7))
        p1 = node_to.gather(1, j[..., None].expand(B, horizon, 7))
        pos = p0[..., :3] + f[..., None] * (p1[..., :3] - p0[..., :3])
        quat = slerp(p0[..., 3:7], p1[..., 3:7], f)
        # Gripper: the command of the last via-point arrived at (end of its travel).
        arrive = cum[:, 1::2]  # [B, V]: end of each travel segment
        n_reached = (arrive[:, None, :] <= s[..., None] + 1e-9).sum(-1)
        cmd = pts[..., 7].gather(1, (n_reached - 1).clamp_min(0))
        grip = torch.where(n_reached > 0, cmd, cur[:, None, 7].expand(B, horizon))
        moved = torch.cat([pos, quat, grip[..., None]], -1)
        chunk[:, :, cols] = torch.where((n > 0)[:, None, None], moved, chunk[:, :, cols])
    return chunk, valid
