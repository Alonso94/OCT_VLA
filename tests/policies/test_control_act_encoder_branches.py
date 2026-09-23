# ruff: noqa: E402
"""ControlACT's encoder-side branches: AdaLN-Zero and in-context entity tokens.

Each test is the assertion that would catch the branch silently doing nothing,
or silently doing something at step 0 of a two-stage fine-tune.
"""

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from oct_vla.policies.control_act import ControlACTConfig, ControlACTPolicy
from oct_vla.policies.control_act.modeling_control_act import adaln_sites

ENCODER_LAYERS, DECODER_LAYERS = 2, 1
SITES = 2 * ENCODER_LAYERS + 3 * DECODER_LAYERS


def config(*, dropout=0.0, pre_norm=False, use_vae=False, side=32, **flags):
    return ControlACTConfig(
        device="cpu",
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=ENCODER_LAYERS,
        n_decoder_layers=DECODER_LAYERS,
        chunk_size=4,
        n_action_steps=2,
        use_vae=use_vae,
        dropout=dropout,
        pre_norm=pre_norm,
        object_attention_heads=2,
        pretrained_backbone_weights=None,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
            "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, side, side)),
            "observation.entity_tokens": PolicyFeature(type=FeatureType.STATE, shape=(8, 17)),
            "observation.entity_mask": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
        **flags,
    )


def batch(dtype=torch.float32, *, entities=3, side=32):
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :entities] = True
    return {
        "observation.state": torch.randn(2, 16, dtype=dtype),
        "observation.images.head": torch.rand(2, 3, side, side, dtype=dtype),
        "observation.entity_tokens": torch.randn(2, 8, 17, dtype=dtype),
        "observation.entity_mask": mask,
        "action": torch.randn(2, 4, 16, dtype=dtype),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }


def stock_twin(policy):
    """Stage-1 RGB ACT holding exactly the policy's non-object weights."""
    cfg = policy.config
    stock_config = ACTConfig(
        **{name: copy.deepcopy(getattr(cfg, name)) for name in ACTConfig.__dataclass_fields__}
    )
    stock_config.input_features = {
        key: value
        for key, value in stock_config.input_features.items()
        if not key.startswith("observation.entity_")
    }
    stock = ACTPolicy(stock_config)
    stock.load_state_dict(
        {k: v for k, v in policy.state_dict().items() if "object_conditioning" not in k}
    )
    return stock.to(next(policy.parameters()).dtype)


def train(policy, data, steps=3, lr=0.01):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    for _ in range(steps):
        loss, _ = policy(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@pytest.mark.parametrize("pre_norm", [False, True])
def test_adaln_is_exactly_stage1_at_init(pre_norm):
    torch.manual_seed(0)
    # float32 because ACT builds its latent in float32; identity is still
    # bitwise, since x * (1 + 0) + 0 is exact in any precision.
    policy = ControlACTPolicy(config(pre_norm=pre_norm, object_conditioning="kv_adaln"))
    stock = stock_twin(policy)
    data = batch()
    torch.testing.assert_close(
        policy.predict_action_chunk(data), stock.predict_action_chunk(data), rtol=0, atol=0
    )
    policy.train(), stock.train()
    torch.testing.assert_close(policy(data)[0], stock(data)[0], rtol=0, atol=0)


def test_adaln_sites_are_main_encoder_and_decoder_only():
    policy = ControlACTPolicy(config(use_vae=True, object_conditioning="kv_adaln"))
    sites = adaln_sites(policy.model)
    assert len(sites) == SITES
    assert policy.object_conditioning.adaln.sites == SITES
    vae = {id(module) for module in policy.model.vae_encoder.modules()}
    assert not any(id(module) in vae for pair in sites for module in pair)


def test_adaln_every_site_receives_gradient_and_only_the_projection_at_step0():
    torch.manual_seed(0)
    policy = ControlACTPolicy(config(object_conditioning="kv_adaln"))
    adaln = policy.object_conditioning.adaln
    policy.train()
    loss, _ = policy(batch())
    loss.backward()
    used = adaln.modulation.weight.grad.view(SITES, 3, 16, 16).flatten(2).abs().sum(-1) > 0
    # Every norm hook fires: a site whose hook never runs leaves its beta block
    # exactly zero.
    assert used[:, 2].all()
    # Every gate and scale too, with one principled exception. ACT's decoder
    # input is all zeros, so at init (zero MHA biases) the decoder self-attention
    # outputs exactly 0 and so does LN of it: alpha and gamma there have nothing
    # to scale. A trained stage-1 checkpoint has non-zero biases.
    decoder_self_attention = 2 * ENCODER_LAYERS
    others = [i for i in range(SITES) if i != decoder_self_attention]
    assert used[others, :2].all()
    assert not used[decoder_self_attention, :2].any()
    for name, parameter in adaln.named_parameters():
        if not name.startswith("modulation"):
            assert parameter.grad is None or not parameter.grad.any(), name
    policy.zero_grad()
    train(policy, batch(), steps=3)
    loss, _ = policy(batch())
    loss.backward()
    assert adaln.pool.in_proj_weight.grad.any()
    assert adaln.embedding.numeric_projection.weight.grad.any()


def test_adaln_live_policy_reads_objects_but_empty_scene_is_stage1():
    torch.manual_seed(0)
    policy = ControlACTPolicy(config(object_conditioning="kv_adaln"))
    data = batch()
    train(policy, data)
    assert policy.object_conditioning.adaln.is_live
    assert policy.object_conditioning.is_live
    policy.eval()
    moved = dict(data, **{"observation.entity_tokens": data["observation.entity_tokens"] + 5})
    assert not torch.allclose(policy.predict_action_chunk(data), policy.predict_action_chunk(moved))
    # With no entity at all, only the layerwise branch could act, and it too
    # returns an exact zero, so the policy must be its non-object weights.
    empty = dict(data, **{"observation.entity_mask": torch.zeros(2, 8, dtype=torch.bool)})
    torch.testing.assert_close(
        policy.predict_action_chunk(empty), stock_twin(policy).eval().predict_action_chunk(empty)
    )


def test_adaln_padded_slots_get_no_gradient():
    torch.manual_seed(0)
    policy = ControlACTPolicy(config(object_conditioning="kv_adaln"))
    data = batch()
    train(policy, data)
    tokens = data["observation.entity_tokens"].clone().requires_grad_(True)
    loss, _ = policy(dict(data, **{"observation.entity_tokens": tokens}))
    loss.backward()
    assert tokens.grad[:, :3].any()
    assert not tokens.grad[:, 3:].any()


def test_incontext_extends_encoder_only_and_stays_near_stage1():
    torch.manual_seed(0)
    # 96 px, not 32: at 32 px ResNet leaves one image token, so three
    # entities would face three native tokens, which no real run looks like.
    # The deviation falls with the native token count (2.6e-2 at 32 px,
    # 2.5e-3 at 96, 7.9e-4 at 160); production has ~240 image tokens.
    policy = ControlACTPolicy(config(object_conditioning="kv_tokens", side=96))
    stock = stock_twin(policy)
    lengths = {}

    def encoder_length(module, args, output):
        lengths["encoder"] = output.shape[0]

    def memory_length(module, args, kwargs, output):
        lengths["memory"] = kwargs["key"].shape[0]

    policy.model.encoder.layers[0].register_forward_hook(encoder_length)
    policy.model.decoder.layers[0].multihead_attn.register_forward_hook(
        memory_length, with_kwargs=True
    )
    data = batch(entities=3, side=96)
    ours, theirs = policy.predict_action_chunk(data), stock.predict_action_chunk(data)
    assert lengths["encoder"] == lengths["memory"] + 8
    deviation = ((ours - theirs).abs().mean() / theirs.abs().mean()).item()
    print(f"in-context step-0 relative L1 deviation: {deviation:.2e}")
    assert 0 < deviation < 1e-2


def test_incontext_gate_is_trainable_at_step0_and_padding_is_exact():
    torch.manual_seed(0)
    policy = ControlACTPolicy(config(object_conditioning="kv_tokens"))
    policy.train()
    loss, _ = policy(batch())
    loss.backward()
    assert policy.object_conditioning.incontext.gate.grad.abs() > 0
    train(policy, batch())
    assert policy.object_conditioning.incontext.is_live
    policy.eval()
    data = batch()
    empty = dict(data, **{"observation.entity_mask": torch.zeros(2, 8, dtype=torch.bool)})
    torch.testing.assert_close(
        policy.predict_action_chunk(empty), stock_twin(policy).eval().predict_action_chunk(empty)
    )


@pytest.mark.parametrize("arm", ["kv_adaln", "kv_tokens"])
def test_round_trip_and_closed_loop(arm, tmp_path):
    torch.manual_seed(0)
    policy = ControlACTPolicy(config(object_conditioning=arm))
    data = batch()
    train(policy, data)
    policy.eval()
    expected = policy.predict_action_chunk(data)
    policy.save_pretrained(tmp_path)
    restored = ControlACTPolicy.from_pretrained(tmp_path)
    assert restored.config.object_conditioning == arm
    torch.testing.assert_close(restored.predict_action_chunk(data), expected)
    restored.reset()
    torch.testing.assert_close(restored.select_action(data), expected[:, 0])
    assert restored.object_conditioning._inputs is None
    assert restored._adaln_cache is None


def test_default_checkpoint_has_neither_branch(tmp_path):
    policy = ControlACTPolicy(config())
    assert not hasattr(policy.object_conditioning, "adaln")
    assert not hasattr(policy.object_conditioning, "incontext")
    policy.save_pretrained(tmp_path)
    restored = ControlACTPolicy.from_pretrained(tmp_path)
    assert restored.config.object_conditioning == "kv"
