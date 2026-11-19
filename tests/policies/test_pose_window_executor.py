"""ControlACT with the BimanualPFN pose window: training on residual targets, decoding at
inference, bank selection, and the refusals that keep those paths from silently failing."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
from lerobot.configs.types import FeatureType, PolicyFeature

from oct_vla.policies.conditioning import pose_tokens as P
from oct_vla.policies.conditioning.future import FUTURE_KEYFRAME, FUTURE_MASK
from oct_vla.policies.control_act import ControlACTPolicy

_h = importlib.util.spec_from_file_location(
    "act_helpers", Path(__file__).resolve().parent / "test_control_act_encoder_branches.py")
A = importlib.util.module_from_spec(_h)
_h.loader.exec_module(A)

H, BANK = 4, 2
ROWS = 2 * H
STATS = {"mean": [0.0] * 16, "std": [0.5] * 16, "min": [-1.0] * 16, "max": [1.0] * 16}


def config(**flags):
    cfg = A.config(object_conditioning="kv", object_future=True, object_future_kind="pose_window",
                   object_future_rows=ROWS, object_future_dim=P.POSE_TOKEN_DIM,
                   object_future_bank=BANK, action_residual_stats=STATS, **flags)
    cfg.input_features[FUTURE_KEYFRAME] = PolicyFeature(type=FeatureType.ENV, shape=(BANK * ROWS, P.POSE_TOKEN_DIM))
    cfg.input_features[FUTURE_MASK] = PolicyFeature(type=FeatureType.ENV, shape=(BANK * ROWS,))
    return cfg


def windows(seed):
    rng = np.random.default_rng(seed)
    w = np.zeros((H, 2, 7), np.float32)
    w[..., :3] = rng.normal(0, 0.1, (H, 2, 3))
    w[..., 6] = 1.0
    return w


def batch(entry1_seed=2):
    torch.manual_seed(123)          # the same images / state / actions in every batch
    b = A.batch()
    toks, mask = P.bank([P.window_tokens(windows(1)), P.window_tokens(windows(entry1_seed))], ROWS)
    b[FUTURE_KEYFRAME] = torch.tensor(toks)[None].repeat(2, 1, 1)
    b[FUTURE_MASK] = torch.tensor(mask)[None].repeat(2, 1)
    return b


def test_training_on_residual_targets_moves_the_pose_window_branch():
    torch.manual_seed(0)
    policy = ControlACTPolicy(config(action_residual=True))
    seen = {}
    from lerobot.policies.act.modeling_act import ACTPolicy

    original = ACTPolicy.forward

    def spy(self, b):
        seen["action"] = b["action"].clone()
        return original(self, b)

    data = batch()
    ACTPolicy.forward = spy
    try:
        A.train(policy, data, steps=3)
    finally:
        ACTPolicy.forward = original
    pose = list(P.POSE_COLS)
    assert not torch.allclose(seen["action"][..., pose], data["action"][..., pose])  # residual
    torch.testing.assert_close(seen["action"][..., list(P.GRIP_COLS)],
                               data["action"][..., list(P.GRIP_COLS)])           # grips absolute
    assert policy.object_conditioning.future.values().__iter__().__next__().is_live


def test_prediction_is_the_residual_output_plus_the_window():
    torch.manual_seed(0)
    residual = ControlACTPolicy(config(action_residual=True)).eval()
    plain = ControlACTPolicy(config(action_residual=False)).eval()
    plain.load_state_dict(residual.state_dict())
    b = batch()
    with torch.no_grad():
        decoded, raw = residual.predict_action_chunk(b), plain.predict_action_chunk(b)
    codec = P.ActionCodec.from_stats(STATS, "MEAN_STD")
    window = torch.tensor(windows(1))[None].repeat(2, 1, 1, 1)       # bank entry 0
    torch.testing.assert_close(decoded, codec.decode(raw, window))


def test_evaluation_reads_bank_entry_zero_only():
    torch.manual_seed(0)
    policy = ControlACTPolicy(config(action_residual=True)).eval()
    torch.nn.init.normal_(next(iter(policy.object_conditioning.future.values())).to_v.weight)
    with torch.no_grad():
        a = policy.predict_action_chunk(batch(entry1_seed=2))
        b = policy.predict_action_chunk(batch(entry1_seed=7))
    torch.testing.assert_close(a, b)


def test_a_residual_policy_refuses_an_incomplete_window():
    policy = ControlACTPolicy(config(action_residual=True)).eval()
    b = batch()
    b[FUTURE_MASK] = b[FUTURE_MASK].clone()
    b[FUTURE_MASK][:, 0] = 0
    with pytest.raises(ValueError, match="complete PFN window"):
        policy.predict_action_chunk(b)
