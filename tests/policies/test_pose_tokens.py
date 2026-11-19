"""Stage-3 executor conditioning: pose tokens, banks, the residual codec, the K/V branch."""

import numpy as np
import pytest
import torch

from oct_vla.policies.conditioning import pose_tokens as P
from oct_vla.prior.pose_window import fixtures as F


def _window(H=16, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros((H, 2, 7), np.float32)
    w[..., :3] = rng.normal(0, 0.1, (H, 2, 3)) + [0, -0.15, 1.0]
    q = rng.normal(size=(H, 2, 4))
    w[..., 3:] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return w


def _cfg():
    from oct_vla.policies.control_act.configuration_control_act import ControlACTConfig

    return ControlACTConfig(object_conditioning="kv", object_entity_normalizer=None)


def test_window_tokens_round_trip_and_layout():
    w = _window()
    t = P.window_tokens(w)
    assert t.shape == (32, P.POSE_TOKEN_DIM)
    assert t[0, 7 + P.ROLES.index("window_left")] == 1
    assert t[1, 7 + P.ROLES.index("window_right")] == 1
    assert t[0, 13] == 0 and t[-1, 13] == 1
    np.testing.assert_allclose(P.tokens_to_window(torch.tensor(t)[None])[0].numpy(), w)


def test_support_tokens_carry_both_wrists_and_every_object():
    ep = F.restock_episode([-0.2, 0.0, 0.2], ["red", "blue", "red"], "red")
    t = P.support_tokens(ep, rows=8)
    roles = t[:, 7:13].argmax(1)
    assert (roles == P.ROLES.index("left_wrist")).sum() == 8
    assert (roles == P.ROLES.index("movable")).sum() == 2 * 3
    colours = t[roles == P.ROLES.index("movable")][:, 14:19].argmax(1)
    assert sorted(set(colours.tolist())) == [1, 2]             # red and blue reach the tokens


def test_bank_selection_is_entry_zero_at_eval_and_skips_empty_entries_in_training():
    a, b = P.window_tokens(_window(seed=1)), P.window_tokens(_window(seed=2))
    toks, mask = P.bank([a, b], rows=32)
    toks, mask = torch.tensor(toks)[None].repeat(64, 1, 1), torch.tensor(mask)[None].repeat(64, 1)
    t, _ = P.select_bank(toks, mask, 2, training=False)
    np.testing.assert_allclose(t[0].numpy(), a)
    t, _ = P.select_bank(toks, mask, 2, training=True, generator=torch.Generator().manual_seed(0))
    drawn = {bool(torch.allclose(t[i], torch.tensor(b))) for i in range(64)}
    assert drawn == {False, True}                               # both entries get drawn
    toks2, mask2 = P.bank([a, a[:0]], rows=32)                  # entry 1 is empty
    _, m = P.select_bank(torch.tensor(toks2)[None].repeat(32, 1, 1),
                         torch.tensor(mask2)[None].repeat(32, 1), 2, training=True)
    assert bool((m.sum(-1) == 32).all())


@pytest.mark.parametrize("mode", ["MEAN_STD", "MIN_MAX"])
def test_residual_codec_round_trips_and_is_zero_on_the_window(mode):
    rng = np.random.default_rng(0)
    stats = {"mean": rng.normal(0, 0.1, 16).tolist(), "std": rng.uniform(0.05, 0.3, 16).tolist(),
             "min": rng.uniform(-1, -0.5, 16).tolist(), "max": rng.uniform(0.5, 1, 16).tolist()}
    codec = P.ActionCodec.from_stats(stats, mode)
    w = torch.tensor(_window())[None]
    # An expert chunk equal to the window (pose columns, the last pose held past H) has
    # zero pose residual; grippers stay absolute.
    raw = torch.zeros(1, 20, 16)
    idx = torch.clamp(torch.arange(20), max=15)
    raw[..., 0:7], raw[..., 8:15] = w[0, idx, 0], w[0, idx, 1]
    raw[..., 7], raw[..., 15] = 0.3, 0.9
    norm = (raw - codec.offset) / codec.scale
    res = codec.encode(norm, w)
    assert res[..., list(P.POSE_COLS)].abs().max() < 1e-5
    torch.testing.assert_close(res[..., list(P.GRIP_COLS)], norm[..., list(P.GRIP_COLS)])
    # Decode inverts encode on any expert chunk, also against a sign-flipped window: the
    # same positions and grippers, the same rotations (a quaternion up to its sign).
    expert = raw.clone()
    expert[..., :3] += 0.02 * torch.randn(1, 20, 3)
    expert[..., 3:7] = torch.nn.functional.normalize(expert[..., 3:7] + 0.1 * torch.randn(1, 20, 4), dim=-1)
    expert[..., 11:15] = -expert[..., 11:15]                    # an arbitrary label sign
    other = (expert - codec.offset) / codec.scale
    for win in (w, -w * torch.tensor([1, 1, 1, -1, -1, -1, -1]).neg()):
        back = codec.decode(codec.encode(other, win), win) * codec.scale + codec.offset
        cols = [0, 1, 2, 7, 8, 9, 10, 15]
        torch.testing.assert_close(back[..., cols], expert[..., cols], atol=1e-5, rtol=1e-5)
        for q in (slice(3, 7), slice(11, 15)):
            dot = (back[..., q] * expert[..., q]).sum(-1).abs()
            torch.testing.assert_close(dot, torch.ones_like(dot), atol=1e-5, rtol=1e-5)
    # GR00T pads past the 16 eeabs columns: padding passes through untouched.
    padded = torch.cat([other, torch.randn(1, 20, 16)], -1)
    torch.testing.assert_close(codec.encode(padded, w)[..., 16:], padded[..., 16:])


def test_pose_token_branch_starts_inert_masks_padding_and_adds_nothing_when_empty():
    branch = P.PoseTokenAttention(_cfg(), 64, heads=4)
    q = torch.randn(2, 4, 5, 16)
    toks = torch.tensor(P.window_tokens(_window()))[None].repeat(2, 1, 1)
    mask = torch.ones(2, 32)
    assert branch(q, toks, mask, output_weight=torch.randn(64, 64)).abs().max() == 0
    torch.nn.init.normal_(branch.to_v.weight)
    torch.nn.init.normal_(branch.to_k.weight)
    half = mask.clone()
    half[:, 16:] = 0
    garbage = toks.clone()
    garbage[:, 16:] = 1e6                                       # anything in a padded slot
    torch.testing.assert_close(branch(q, garbage, half, output_weight=torch.eye(64)),
                               branch(q, toks, half, output_weight=torch.eye(64)))
    assert not torch.allclose(branch(q, toks, mask, output_weight=torch.eye(64)),
                              branch(q, toks, half, output_weight=torch.eye(64)))
    empty = branch(q, toks, torch.zeros(2, 32), output_weight=torch.eye(64))
    assert empty.abs().max() == 0


def test_config_validates_pose_kinds_banks_and_residual():
    from oct_vla.policies.control_act.configuration_control_act import ControlACTConfig

    ok = ControlACTConfig(object_conditioning="kv", object_entity_normalizer=None,
                          object_future=True, object_future_kind="pose_window",
                          object_future_rows=32, object_future_dim=19, object_future_bank=4)
    assert ok.object_future_bank == 4
    with pytest.raises(ValueError, match="19 wide"):
        ControlACTConfig(object_conditioning="kv", object_entity_normalizer=None,
                         object_future=True, object_future_kind="pose_window",
                         object_future_rows=32, object_future_dim=24)
    with pytest.raises(ValueError, match="residual_stats"):
        ControlACTConfig(object_conditioning="kv", object_entity_normalizer=None,
                         object_future=True, object_future_kind="pose_window",
                         object_future_rows=32, object_future_dim=19, action_residual=True)
    with pytest.raises(ValueError, match="kind pose_window"):
        ControlACTConfig(object_conditioning="kv", object_entity_normalizer=None,
                         object_future=True, object_future_kind="support_table",
                         object_future_rows=40, object_future_dim=19, action_residual=True,
                         action_residual_stats={k: [0.0] * 16 for k in ("mean", "std", "min", "max")})
