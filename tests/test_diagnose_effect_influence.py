"""Pure parts of scripts/diagnose_effect_influence.py."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "diagnose_effect_influence",
    Path(__file__).parents[1] / "scripts/diagnose_effect_influence.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def _tokens():
    from oct_vla.core.frames import WORKCELL_FRAME, Pose
    from oct_vla.data.entity_tokens import _token

    up = (0.0, 0.0, 0.0, 1.0)
    toks = [_token(Pose((x, -0.25, 0.79), up, WORKCELL_FRAME), (0.05, 0.05, 0.1), 0.0,
                   "movable") for x in (-0.25, 0.0, 0.25)]
    toks.append(_token(Pose((-0.3, -0.1, 1.2), up, WORKCELL_FRAME), (0, 0, 0), 0.83,
                       "left_gripper"))
    return np.array(toks), np.ones(4)


def _z(tokens, k):
    from oct_vla.data.effects import yaw_token

    z = np.zeros((3, 24))
    z[0, :17] = yaw_token(tokens[k, :3], (0, 0, 0, 1), tokens[k, 9:12])
    z[2, :17] = yaw_token((-0.2, -0.06, 0.97), (0, 0, 0, 1), tokens[k, 9:12])
    z[0, 17], z[2, 19] = 1, 1
    return z, np.array([1.0, 0.0, 1.0])


def test_approach_case_names_the_true_and_nearest_other_object():
    tokens, mask = _tokens()
    z, zm = _z(tokens, 0)
    true_k, alt_k, idx, pos = D.approach_case(tokens, mask, z, zm, np.array([-0.3, -0.1, 1.2]))
    assert idx[true_k] == 0 and idx[alt_k] == 1
    # Engaged with an object (mover within HOLD_DIST): not an approach.
    assert D.approach_case(tokens, mask, z, zm, tokens[0, :3] + (0, 0, 0.1)) is None
    # No effect: nothing to score.
    assert D.approach_case(tokens, mask, z, np.zeros(3), np.array([-0.3, -0.1, 1.2])) is None


def test_alt_rows_swap_the_object_and_keep_the_goal_position():
    tokens, _ = _tokens()
    z, _ = _z(tokens, 0)
    alt = D.alt_rows(z, tokens, 1)
    np.testing.assert_allclose(alt[0, :3], tokens[1, :3])
    np.testing.assert_allclose(alt[2, :3], z[2, :3])
    assert alt[0, 17:].tolist() == z[0, 17:].tolist()


def test_heads_to_picks_the_object_the_path_closes_on():
    eef = np.array([0.0, 0.0, 1.2])
    a, b = np.array([-0.3, 0.0, 0.8]), np.array([0.3, 0.0, 0.8])
    toward_b = np.linspace(eef, b + (0, 0, 0.3), 25)
    assert D.heads_to(toward_b, eef, a, b) == 1
    assert D.heads_to(toward_b[::-1] * 0 + np.linspace(eef, a, 25), eef, a, b) == 0


def test_perturbed_moves_only_the_goal_along_the_row():
    tokens, _ = _tokens()
    z, _ = _z(tokens, 1)
    for sign in (1.0, -1.0):
        out = D.perturbed(z, sign)
        np.testing.assert_array_equal(out[0], z[0])
        np.testing.assert_allclose(out[2, 0] - z[2, 0], sign * D.PERTURB)
        np.testing.assert_array_equal(out[2, 1:], z[2, 1:])
