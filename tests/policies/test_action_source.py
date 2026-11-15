"""The kinematic bridge from a keyframe to an eeabs action chunk (conditioning/action_source.py)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oct_vla.core.frames import WORKCELL_FRAME, Pose  # noqa: E402
from oct_vla.core.geometry import unit_quaternion  # noqa: E402
from oct_vla.data.entity_tokens import _rotation6, _token  # noqa: E402
from oct_vla.data.future_keyframes import (  # noqa: E402
    FUTURE_DIM,
    FUTURE_DIM_WP,
    FUTURE_ROLES,
    FUTURE_ROLES_WP,
    _row,
)
from oct_vla.policies.conditioning.action_source import (  # noqa: E402
    GRIPPER_OPEN_THRESHOLD,
    gripper_states,
    kinematic_proposal,
    normalize_actions,
    rotation6d_to_quat,
)

DOWN = (-0.5, 0.5, 0.5, 0.5)


def _quat(rng):
    q = rng.normal(size=4)
    return unit_quaternion(q / np.linalg.norm(q))


def test_rotation6d_round_trips_to_the_canonical_quaternion():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = _quat(rng)
        r6 = torch.tensor(_rotation6(Pose((0, 0, 0), q, WORKCELL_FRAME)), dtype=torch.float64)
        got = rotation6d_to_quat(r6).numpy()
        np.testing.assert_allclose(got, q, atol=1e-7)


def test_normalisation_is_grootss_min_max():
    raw = torch.randn(5, 40, 16) * 2
    lo, hi = torch.full((16,), -1.5), torch.full((16,), 1.0)
    lo[3], hi[3] = 0.2, 0.2  # a constant column
    want = (2 * (raw - lo) / torch.where(hi - lo == 0, torch.ones(16), hi - lo) - 1)
    want = torch.where(hi - lo != 0, want, torch.zeros_like(want)).clamp(-1, 1)
    torch.testing.assert_close(normalize_actions(raw, lo, hi), want)


def _scene(left=(-0.3, -0.1, 1.2), right=(0.4, -0.2, 1.2), left_open=True):
    aperture = 0.83 if left_open else 0.6
    toks = [_token(Pose((0.0, 0.1, 0.9), (0, 0, 0, 1), WORKCELL_FRAME), (0.05, 0.05, 0.08), 0.0,
                   "movable"),
            _token(Pose(left, DOWN, WORKCELL_FRAME), (0, 0, 0), aperture, "left_gripper"),
            _token(Pose(right, DOWN, WORKCELL_FRAME), (0, 0, 0), 0.83, "right_gripper")]
    return torch.tensor([toks], dtype=torch.float64), torch.ones(1, 3, dtype=torch.bool)


def _keyframe(event="grasp", target=(0.0, 0.1, 1.04), waypoint=None, side="left"):
    roles = FUTURE_ROLES_WP if waypoint is not None else FUTURE_ROLES
    obj = _token(Pose((0.0, 0.1, 0.9), (0, 0, 0, 1), WORKCELL_FRAME), (0.05, 0.05, 0.08), 0.0,
                 "movable")
    yawed = unit_quaternion((0.0, 0.0, np.sin(0.2), np.cos(0.2)))
    from oct_vla.core.geometry import multiply

    q = unit_quaternion(multiply(yawed, DOWN))
    grip = _token(Pose(target, q, WORKCELL_FRAME), (0, 0, 0), 0.6, f"{side}_gripper")
    rows = [_row(obj, "object_keyframe", event, roles), _row(grip, "gripper_keyframe", event, roles),
            _row(obj, "object_goal", event, roles)]
    if waypoint is not None:
        way = _token(Pose(waypoint, q, WORKCELL_FRAME), (0, 0, 0), 1.0, f"{side}_gripper")
        rows.append(_row(way, "gripper_waypoint", event, roles))
    return (torch.tensor(np.stack(rows)[None], dtype=torch.float64),
            torch.ones(1, len(rows), dtype=torch.bool), q)


def test_mover_passes_the_waypoint_ends_at_the_event_and_closes_on_arrival():
    tokens, emask = _scene()
    kf, kmask, q = _keyframe(waypoint=(0.0, 0.1, 1.14))
    chunk, valid = kinematic_proposal(tokens, emask, kf, kmask, speed=0.3)
    assert valid.all() and chunk.shape == (1, 40, 16)
    left = chunk[0, :, :8]
    np.testing.assert_allclose(left[-1, :3], (0.0, 0.1, 1.04), atol=1e-9)
    np.testing.assert_allclose(left[-1, 3:7], q, atol=1e-7)
    # It reaches the waypoint before descending: some step is at the waypoint.
    assert (torch.linalg.norm(left[:, :3] - torch.tensor([0.0, 0.1, 1.14], dtype=torch.float64),
                              dim=-1) < 0.3 / 15 + 1e-9).any()
    # Constant speed: no step moves farther than speed * dt.
    steps = torch.linalg.norm(torch.diff(torch.cat([torch.tensor([[-0.3, -0.1, 1.2]],
                                                                  dtype=torch.float64),
                                                     left[:, :3]]), dim=0), dim=-1)
    assert (steps <= 0.3 / 15 + 1e-9).all()
    grip = left[:, 7]
    assert grip[0] == 1.0 and grip[-1] == 0.0 and int((torch.diff(grip) != 0).sum()) == 1
    # The idle arm holds its pose and gripper.
    right = chunk[0, :, 8:]
    assert torch.allclose(right, right[:1].expand_as(right))
    np.testing.assert_allclose(right[0, :3], (0.4, -0.2, 1.2), atol=1e-9)


def test_place_opens_and_a_right_mover_moves_the_right_arm():
    tokens, emask = _scene()
    kf, kmask, _ = _keyframe(event="place", target=(0.35, 0.0, 1.1), side="right")
    chunk, _ = kinematic_proposal(tokens, emask, kf, kmask, speed=1.0)
    assert torch.allclose(chunk[0, :, :8], chunk[0, :1, :8].expand(40, 8))  # left idle
    np.testing.assert_allclose(chunk[0, -1, 8:11], (0.35, 0.0, 1.1), atol=1e-9)
    assert chunk[0, -1, 15] == 1.0


def test_an_empty_keyframe_is_flagged_and_holds_both_arms():
    tokens, emask = _scene()
    kf, kmask, _ = _keyframe()
    chunk, valid = kinematic_proposal(tokens, emask, torch.zeros_like(kf),
                                      torch.zeros_like(kmask), speed=0.3)
    assert not valid.any()
    assert torch.allclose(chunk, chunk[:, :1].expand_as(chunk))


def test_goal_only_rows_give_a_top_down_pose_above_the_object():
    from oct_vla.tasks.shelf_restock.oracle.grasps import GRASP_TCP_OFFSET

    tokens, emask = _scene()
    kf, kmask, _ = _keyframe()
    kmask[:, 1] = False  # a goal-only prior: no gripper row
    kf[:, 1] = 0
    chunk, valid = kinematic_proposal(tokens, emask, kf, kmask, speed=1.0)
    end = chunk[0, -1, :3].numpy()
    assert valid.all()
    np.testing.assert_allclose(end[:2], (0.0, 0.1), atol=1e-6)
    assert 0.9 + 0.04 < end[2] < 0.9 + 0.04 + GRASP_TCP_OFFSET + 0.01


def test_gripper_state_reads_the_measured_aperture_and_refuses_missing_grippers():
    tokens, emask = _scene(left_open=False)
    states = gripper_states(tokens, emask)
    assert states["left"][0, 7] == 0.0 and states["right"][0, 7] == 1.0
    assert 0.83 >= GRIPPER_OPEN_THRESHOLD > 0.79
    emask[0, 2] = False
    with pytest.raises(ValueError, match="right_gripper"):
        gripper_states(tokens, emask)


def test_both_layouts_are_accepted_and_others_refused():
    tokens, emask = _scene()
    kf3, m3, _ = _keyframe()
    assert kf3.shape[-1] == FUTURE_DIM
    kinematic_proposal(tokens, emask, kf3, m3, speed=0.3)
    kf4, m4, _ = _keyframe(waypoint=(0, 0.1, 1.2))
    assert kf4.shape[-1] == FUTURE_DIM_WP
    kinematic_proposal(tokens, emask, kf4, m4, speed=0.3)
    with pytest.raises(ValueError):
        kinematic_proposal(tokens, emask, kf4[..., :23], m4, speed=0.3)


def _vias(points, side=0, V=6):
    """points: [(xyz, quat, grip)] for one arm -> [1, 2, V, 8] and mask."""
    t = torch.zeros(1, 2, V, 8, dtype=torch.float64)
    m = torch.zeros(1, 2, V, dtype=torch.bool)
    for i, (xyz, q, g) in enumerate(points):
        t[0, side, i] = torch.tensor([*xyz, *q, g], dtype=torch.float64)
        m[0, side, i] = True
    return t, m


def test_via_points_are_visited_in_order_and_grippers_switch_on_arrival():
    from oct_vla.policies.conditioning.action_source import via_point_proposal

    tokens, emask = _scene()
    t, m = _vias([((-0.3, -0.1, 1.3), DOWN, 1.0), ((0.0, 0.1, 1.3), DOWN, 1.0),
                  ((0.0, 0.1, 1.05), DOWN, 0.0)])
    chunk, valid = via_point_proposal(tokens, emask, t, m, speed=0.6)
    left = chunk[0, :, :8]
    assert valid.all()
    np.testing.assert_allclose(left[-1, :3], (0.0, 0.1, 1.05), atol=1e-9)
    # Goes up first (x unchanged), then across, then down; closes only at the end.
    first_x_move = int(torch.nonzero(left[:, 0] > -0.3 + 1e-6)[0])
    assert (left[:first_x_move, 2] > 1.2).all()  # rising, x fixed, before going across
    corner = torch.tensor([-0.3, -0.1, 1.3], dtype=torch.float64)
    assert torch.linalg.norm(left[:, :3] - corner, dim=-1).min() <= 0.6 / 15 + 1e-9
    grip = left[:, 7]
    assert grip[0] == 1.0 and grip[-1] == 0.0 and int((torch.diff(grip) != 0).sum()) == 1
    assert torch.allclose(chunk[0, :, 8:], chunk[0, :1, 8:].expand(40, 8))  # right holds


def test_a_via_point_at_the_current_pose_closes_in_place_at_once():
    from oct_vla.policies.conditioning.action_source import via_point_proposal

    tokens, emask = _scene()
    t, m = _vias([((0.4, -0.2, 1.2), DOWN, 0.0), ((0.4, 0.0, 1.0), DOWN, 0.0)], side=1)
    chunk, _ = via_point_proposal(tokens, emask, t, m, speed=0.3)
    assert chunk[0, 0, 15] == 0.0  # closed from the first step
    moved = np.linalg.norm(chunk[0, 0, 8:11].numpy() - (0.4, -0.2, 1.2))
    assert moved == pytest.approx(0.3 / 15, abs=1e-9)  # then on toward the next point


def test_via_masks_must_be_prefixes_and_none_means_hold():
    from oct_vla.policies.conditioning.action_source import via_point_proposal

    tokens, emask = _scene()
    t, m = _vias([((0, 0, 1.2), DOWN, 1.0), ((0, 0.1, 1.2), DOWN, 1.0)])
    m[0, 0, 0] = False
    with pytest.raises(ValueError, match="prefix"):
        via_point_proposal(tokens, emask, t, m, speed=0.3)
    chunk, valid = via_point_proposal(tokens, emask, t, torch.zeros_like(m), speed=0.3)
    assert not valid.any() and torch.allclose(chunk, chunk[:, :1].expand_as(chunk))


def test_dwell_holds_the_arm_after_a_gripper_switch_and_speeds_are_per_arm():
    from oct_vla.policies.conditioning.action_source import via_point_proposal

    tokens, emask = _scene()
    t, m = _vias([((-0.3, -0.1, 1.1), DOWN, 0.0), ((-0.3, -0.1, 1.3), DOWN, 0.0)])
    chunk, _ = via_point_proposal(tokens, emask, t, m, speed=(0.3, 0.1), dwell=4)
    z = chunk[0, :, 2]
    arrive = int(torch.nonzero((z - 1.1).abs() < 1e-9)[0])
    assert chunk[0, arrive, 7] == 0.0  # closed on arrival
    assert torch.allclose(z[arrive:arrive + 5], torch.full((5,), 1.1, dtype=torch.float64))
    assert z[arrive + 5] > 1.1  # then moves on
    t2, m2 = _vias([((0.4, -0.2, 1.0), DOWN, 1.0)], side=1)
    slow, _ = via_point_proposal(tokens, emask, t2, m2, speed=(0.3, 0.1))
    assert (slow[0, 0, 10] - 1.2).abs() == pytest.approx(0.1 / 15, abs=1e-9)
