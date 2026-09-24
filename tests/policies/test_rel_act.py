"""rel_act: the chunk-relative target round-trips, and uses the chunk-start state."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402

from oct_vla.policies.rel_act import RelACTConfig, RelACTPolicy  # noqa: E402

D, C = 16, 4


def _policy():
    torch.manual_seed(0)
    config = RelACTConfig(
        device="cpu",
        input_features={"observation.images.head": PolicyFeature(FeatureType.VISUAL, (3, 24, 32)),
                        "observation.state": PolicyFeature(FeatureType.STATE, (D,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (D,))},
        chunk_size=C, n_action_steps=2, dim_model=16, n_heads=2, dim_feedforward=32,
        n_encoder_layers=1, n_decoder_layers=1, latent_dim=4, n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
        state_mean=[0.1] * D, state_std=[2.0] * D,
        target_mean=list(np.linspace(-1, 1, C * D)), target_std=list(np.linspace(0.5, 2, C * D)),
    )
    return RelACTPolicy(config)


def test_the_transform_round_trips_and_offsets_only_positions():
    policy = _policy()
    state = torch.randn(3, D)
    action = torch.randn(3, C, D)
    target = policy._target(action, state)
    assert torch.allclose(policy._absolute(target, state), action, atol=1e-6)
    # Moving the chunk-start state moves the relative dims' target, not the rest.
    moved = state.clone()
    moved[:, 0] += 1.0
    moved[:, 5] += 1.0  # a quaternion component: absolute, so no effect
    changed = (policy._target(action, moved) - target).abs().sum(dim=(0, 1)) > 0
    assert changed.tolist() == [i in (0,) for i in range(D)]


def test_prediction_is_absolute_and_follows_the_state():
    policy = _policy().eval()
    batch = {"observation.images.head": torch.rand(1, 3, 24, 32), "observation.state": torch.zeros(1, D)}
    base = policy.predict_action_chunk(dict(batch))
    shifted = dict(batch)
    shifted["observation.state"] = batch["observation.state"].clone()
    shifted["observation.state"][:, 8] = 0.5
    out = policy.predict_action_chunk(shifted)
    # The state is also an input, so the offset is not exactly additive -- but
    # the relative dim must move by the state change plus the network's response.
    assert out.shape == (1, C, D)
    assert not torch.allclose(out, base)


def test_training_loss_runs_and_stats_are_required():
    policy = _policy().train()
    batch = {"observation.images.head": torch.rand(2, 3, 24, 32), "observation.state": torch.randn(2, D),
             "action": torch.randn(2, C, D), "action_is_pad": torch.zeros(2, C, dtype=torch.bool)}
    loss, _ = policy.forward(batch)
    loss.backward()
    config = policy.config
    config.target_std = config.target_std[:-1]
    with pytest.raises(ValueError, match="target_std"):
        RelACTPolicy(config)


def test_stats_are_per_step_and_fitted_on_what_the_loss_sees():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "stats", Path(__file__).resolve().parents[2] / "scripts/relative_action_stats.py")
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    # One episode moving +1 per step in dim 0: offset at step k is exactly k.
    length = 6
    state = np.zeros((length, D))
    state[:, 0] = np.arange(length)
    action = state + 0.0
    _, _, mean, std = stats.statistics([(state, action)], chunk=3, relative=[0])
    assert np.allclose(mean[:, 0], [0, 1, 2]) and np.allclose(std[:, 0], 1e-3)
    json.dumps(mean.tolist())
