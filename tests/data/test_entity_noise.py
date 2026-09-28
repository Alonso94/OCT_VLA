"""Object-state noise touches only movable rows and keeps rotations rotations."""

import math

import numpy as np
import pytest

from oct_vla.data.entity_noise import EntityNoise, corrupt


def _scene():
    """Two movables, two grippers, one support, one padding row."""
    rows = []
    for kind in (0, 0, 1, 2, 3):
        theta = 0.3 * (len(rows) + 1)
        c0 = [math.cos(theta), math.sin(theta), 0.0]
        c1 = [-math.sin(theta), math.cos(theta), 0.0]
        one_hot = [1.0 if i == kind else 0.0 for i in range(4)]
        rows.append([0.1 * len(rows), -0.2, 0.9, *c0, *c1, 0.05, 0.06, 0.07, 0.0, *one_hot])
    rows.append([0.0] * 17)
    return np.array(rows), np.array([True] * 5 + [False])


def test_zero_noise_is_the_identity():
    tokens, mask = _scene()
    out, out_mask = corrupt(tokens, mask, EntityNoise(), np.random.default_rng(0))
    assert np.array_equal(out, tokens) and np.array_equal(out_mask, mask)


def test_only_movable_rows_change():
    tokens, mask = _scene()
    out, out_mask = corrupt(tokens, mask, EntityNoise(position_mm=5, rotation_deg=5),
                            np.random.default_rng(0))
    assert not np.allclose(out[:2], tokens[:2])
    assert np.array_equal(out[2:], tokens[2:]) and np.array_equal(out_mask, mask)


def test_rotation_noise_keeps_the_columns_orthonormal_and_near_the_truth():
    tokens, mask = _scene()
    out, _ = corrupt(tokens, mask, EntityNoise(rotation_deg=2), np.random.default_rng(1))
    for row in range(2):
        c0, c1 = out[row, 3:6], out[row, 6:9]
        assert np.isclose(np.linalg.norm(c0), 1) and np.isclose(np.linalg.norm(c1), 1)
        assert np.isclose(c0 @ c1, 0, atol=1e-12)
        angle = math.degrees(math.acos(np.clip(c0 @ tokens[row, 3:6], -1, 1)))
        assert angle < 15


def test_position_noise_has_the_requested_scale():
    tokens, mask = _scene()
    rng = np.random.default_rng(2)
    errors = [corrupt(tokens, mask, EntityNoise(position_mm=5), rng)[0][:2, :3] - tokens[:2, :3]
              for _ in range(2000)]
    assert np.isclose(np.std(np.concatenate(errors)) * 1000, 5, rtol=0.05)


def test_dropout_masks_and_zeroes_movables_only():
    tokens, mask = _scene()
    rng = np.random.default_rng(3)
    dropped = 0
    for _ in range(1000):
        out, out_mask = corrupt(tokens, mask, EntityNoise(dropout=0.25), rng)
        assert out_mask[2:5].all() and np.array_equal(out[2:], tokens[2:])
        for row in range(2):
            if not out_mask[row]:
                dropped += 1
                assert not out[row].any()
    assert np.isclose(dropped / 2000, 0.25, atol=0.03)


def test_inputs_are_not_modified_and_bad_levels_are_refused():
    tokens, mask = _scene()
    before = tokens.copy()
    corrupt(tokens, mask, EntityNoise(position_mm=10, dropout=0.5), np.random.default_rng(4))
    assert np.array_equal(tokens, before)
    with pytest.raises(ValueError):
        EntityNoise(dropout=1.0)
    assert EntityNoise(position_mm=5, rotation_deg=2, dropout=0.1).tag == "pos5_rot2_drop10"
