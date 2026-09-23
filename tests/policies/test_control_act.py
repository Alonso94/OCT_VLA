# ruff: noqa: E402
"""Exercise the actual ACT decoder, both prediction entrypoints and checkpoint IO."""

import copy
import inspect

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from oct_vla.policies.control_act import ControlACTConfig, ControlACTPolicy


def tiny_config(*, flat_environment_state=False):
    """A camera-bearing entity arm, as production runs are.

    ACT requires at least one image *or* a flat environment state, and the
    entity tokens are neither. An earlier version of this fixture supplied
    `observation.environment_state` to satisfy that, which is the one
    combination `validate_features` now refuses -- entity tokens retype ENV to
    IDENTITY, and LeRobot resolves normalization per feature type, so a real
    flat env state riding alongside them would stop being normalized.
    """
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
        "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        "observation.entity_tokens": PolicyFeature(type=FeatureType.STATE, shape=(8, 17)),
        "observation.entity_mask": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    if flat_environment_state:
        features["observation.environment_state"] = PolicyFeature(
            type=FeatureType.ENV, shape=(3,)
        )
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
        input_features=features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
    )


def batch():
    return {
        "observation.state": torch.randn(2, 16),
        "observation.images.head": torch.rand(2, 3, 32, 32),
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
    # Entity tokens are typed ENV only to opt out of generic STATE
    # normalization; `env_state_feature` must never mistake them for ACT's
    # flat observation.environment_state input.
    assert config.env_state_feature is None
    config.validate_features()
    assert config.normalization_mapping[FeatureType.ENV] == NormalizationMode.IDENTITY
    assert config.input_features["observation.entity_tokens"].type == FeatureType.ENV
    assert config.env_state_feature is None


def test_entity_tokens_beside_a_flat_environment_state_are_refused():
    """LeRobot resolves a normalization mode per feature *type*, not per
    feature, so setting ENV to IDENTITY for the entity tokens would also stop
    normalizing a genuine observation.environment_state. Silently. Our exporter
    already refuses to write both, so this guards a path that should not exist
    -- but the failure it prevents is invisible, which is why it is loud."""
    config = tiny_config(flat_environment_state=True)
    assert config.env_state_feature.shape == (3,)
    with pytest.raises(ValueError, match="observation.environment_state"):
        config.validate_features()


@pytest.mark.parametrize(
    "policy_type",
    ["control_act", "control_pi05", "control_smolvla", "control_groot", "control_vla_jepa"],
)
def test_every_processor_factory_pins_entity_normalization_to_identity(policy_type):
    """The single point of failure between entity_v2 and a validation leak.

    LeRobot types any `observation.*` key that is not `environment_state` as
    STATE, every backbone here maps STATE to MEAN_STD, and the statistics come
    from `dataset.meta.stats` -- computed over the whole repo, train and
    validation together, because `export_episodes` writes both into one.

    `validate_features` is what retypes the tokens to ENV and pins ENV to
    IDENTITY. control_groot and control_smolvla called it in neither their
    model `__init__` nor their processor factory, so an entity_v2 run on either
    would have normalized the tokens with validation statistics and said
    nothing. This asserts the guard fires for every plugin, at the one call
    site that always runs.
    """
    import importlib

    from lerobot.configs.types import FeatureType, NormalizationMode
    from lerobot.configs.policies import PreTrainedConfig

    import oct_vla.policies  # noqa: F401  -- registers the types

    config = PreTrainedConfig._choice_registry[policy_type]()
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
        "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        "observation.entity_tokens": PolicyFeature(type=FeatureType.STATE, shape=(8, 17)),
        "observation.entity_mask": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    # Some backbones' own validate_features requires one; VLA-JEPA raises.
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))}

    module = importlib.import_module(
        type(config).__module__.replace("configuration_", "processor_")
    )
    factory = getattr(module, f"make_{policy_type}_pre_post_processors")
    assert callable(factory), f"{policy_type} has no processor factory"
    # The factory is reached by naming convention from LeRobot's
    # `make_pre_post_processors`; a rename raises there rather than silently
    # skipping, so resolving it the same way here is the honest check.
    source = inspect.getsource(factory)
    assert "validate_features" in source, (
        f"{policy_type}'s processor factory does not call validate_features, so its "
        "entity tokens would be MEAN_STD-normalized with whole-dataset statistics"
    )

    config.validate_features()
    assert config.normalization_mapping[FeatureType.ENV] == NormalizationMode.IDENTITY
    for key in ("observation.entity_tokens", "observation.entity_mask"):
        assert config.input_features[key].type == FeatureType.ENV, key


def test_the_object_branch_is_dropped_like_the_host_attention():
    """Both terms of the sum must be regularised alike.

    ACT applies `config.dropout` (default 0.1) to its native attention
    probabilities. An undropped object branch would be the less regularised of
    the two, which tilts an object-versus-RGB comparison toward conditioning --
    in the experiment built to measure it. Every earlier test and
    `validate_control_act.py` set dropout=0, so a real run would have been the
    first time the asymmetry mattered.
    """
    config = tiny_config()
    config.dropout = 0.5
    policy = ControlACTPolicy(config)
    layer = next(iter(policy.object_conditioning.layers.values()))
    with torch.no_grad():
        layer.to_v.weight.normal_(std=0.5)
        layer.to_k.weight.normal_(std=0.5)

    query = torch.randn(2, config.n_heads, 4, config.dim_model // config.n_heads)
    tokens = torch.randn(2, 8, 17)
    mask = torch.ones(2, 8, dtype=torch.bool)
    weight = torch.eye(config.dim_model)

    def run(dropout):
        return layer(query, tokens, mask, output_weight=weight, dropout=dropout)

    layer.train()
    torch.manual_seed(0)
    first = run(0.5)
    torch.manual_seed(1)
    assert not torch.equal(first, run(0.5)), "dropout never reached the object branch"
    torch.manual_seed(0)
    assert torch.equal(first, run(0.5)), "the same seed must reproduce the same mask"

    # Eval is deterministic, and dropout=0 is exactly the undropped path, so
    # every existing zero-dropout assertion still holds.
    layer.eval()
    torch.manual_seed(0)
    assert torch.equal(run(0.5), run(0.5))
    layer.train()
    torch.manual_seed(0)
    torch.testing.assert_close(run(0.0), run(0.0))


def test_the_act_hook_passes_the_host_dropout_through():
    """The wiring, not just the capability: ACT's decoder attention owns the
    rate, and the hook must read it from there rather than assume zero."""
    config = tiny_config()
    config.dropout = 0.25
    policy = ControlACTPolicy(config)
    for host in (layer.multihead_attn for layer in policy.model.decoder.layers):
        assert host.dropout == 0.25
    assert "dropout=host.dropout" in inspect.getsource(policy._make_hook)
