# ruff: noqa: E402
"""Exercise the actual ACT decoder, both prediction entrypoints and checkpoint IO."""

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from oct_vla.policies.control_act import ControlACTConfig, ControlACTPolicy


def tiny_config():
    return ControlACTConfig(
        device="cpu",
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=2,
        chunk_size=4,
        n_action_steps=2,
        use_vae=False,
        dropout=0.0,
        pretrained_backbone_weights=None,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(3,)),
            "observation.entity_tokens": PolicyFeature(type=FeatureType.STATE, shape=(8, 17)),
            "observation.entity_mask": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
    )


def batch():
    return {
        "observation.state": torch.randn(2, 16),
        "observation.environment_state": torch.randn(2, 3),
        "observation.entity_tokens": torch.randn(2, 8, 17),
        "observation.entity_mask": torch.ones(2, 8, dtype=torch.bool),
        "action": torch.randn(2, 4, 16),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }


def test_actual_act_zero_identity_train_reload_and_select(tmp_path):
    config = tiny_config()
    policy = ControlACTPolicy(config)
    stock_fields = ACTConfig.__dataclass_fields__
    stock_config = ACTConfig(
        **{name: copy.deepcopy(getattr(config, name)) for name in stock_fields}
    )
    stock_config.input_features = {
        key: value
        for key, value in stock_config.input_features.items()
        if not key.startswith("observation.entity_")
    }
    stock = ACTPolicy(stock_config)
    stock.load_state_dict(
        {
            key: value
            for key, value in policy.state_dict().items()
            if "object_conditioning" not in key
        }
    )
    data = batch()
    torch.testing.assert_close(policy.predict_action_chunk(data), stock.predict_action_chunk(data))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    for _ in range(3):
        policy.train()
        loss, _ = policy(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert all(layer.is_live for layer in policy.object_conditioning.layers.values())
    expected = policy.predict_action_chunk(data)
    changed = dict(data, **{"observation.entity_tokens": data["observation.entity_tokens"] + 5})
    assert not torch.allclose(expected, policy.predict_action_chunk(changed))
    policy.save_pretrained(tmp_path)
    restored = ControlACTPolicy.from_pretrained(tmp_path)
    torch.testing.assert_close(expected, restored.predict_action_chunk(data))
    restored.reset()
    torch.testing.assert_close(restored.select_action(data), expected[:, 0])
    assert policy.object_conditioning._inputs is None


def test_entity_features_are_identity_but_not_flat_environment():
    config = tiny_config()
    config.validate_features()
    assert config.normalization_mapping[FeatureType.ENV] == NormalizationMode.IDENTITY
    assert config.env_state_feature.shape == (3,)
    del config.input_features["observation.environment_state"]
    assert config.env_state_feature is None
