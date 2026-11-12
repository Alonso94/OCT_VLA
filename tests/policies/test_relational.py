"""The relational composer arm (conditioning/relational.py, ``kv_rel``) on GR00T's real DiT."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from torch import nn  # noqa: E402

from oct_vla.policies.conditioning.entity import TOKEN_DIM  # noqa: E402
from oct_vla.policies.conditioning.relational import (  # noqa: E402
    RelationalComposer,
    RelationalKVAttention,
)
from oct_vla.policies.conditioning.hosts import install_diffusers_kv  # noqa: E402
from oct_vla.policies.object_conditioning import ObjectConditioning  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_future_branch import _config, _scene  # noqa: E402


def _model(arm, seed=0):
    from lerobot.policies.groot.action_head.cross_attention_dit import DiT

    torch.manual_seed(seed)
    dit = DiT(num_attention_heads=2, attention_head_dim=4, output_dim=3, num_layers=2,
              dropout=0.0, cross_attention_dim=6, interleave_self_attention=True,
              positional_embeddings=None, final_dropout=False)
    owner = nn.Module()
    owner.model = dit
    config = _config(False)
    config.object_conditioning = arm
    owner.object_conditioning = ObjectConditioning(config, dit.inner_dim)
    install_diffusers_kv(dit, owner.object_conditioning, conditioning_owner=owner)
    return owner.eval()


def _tokens(batch=2, n=5):
    g = torch.Generator().manual_seed(1)
    tokens = torch.randn(batch, n, TOKEN_DIM, generator=g)
    tokens[..., 13:17] = 0.0
    tokens[..., 13] = 1.0  # movables
    tokens[:, 3, 13:17] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # a gripper
    mask = torch.ones(batch, n, dtype=torch.bool)
    mask[:, -1] = False
    tokens[:, -1] = 0.0  # padding
    return tokens, mask


def _run(owner, tokens, mask, scene):
    *_, hidden, context = scene
    control = owner.object_conditioning
    control.set_inputs(tokens, mask)
    try:
        return owner.model(hidden, context, timestep=torch.tensor([3, 7]))
    finally:
        control.clear()


def _trained(owner, seed=5):
    torch.manual_seed(seed)
    for branch in owner.object_conditioning.layers.values():
        for layer in (branch.to_k, branch.to_v):
            nn.init.normal_(layer.weight, std=0.3)


def test_kv_rel_builds_the_composer_and_starts_as_kv():
    kv, rel = _model("kv"), _model("kv_rel")
    assert all(isinstance(b, RelationalKVAttention)
               for b in rel.object_conditioning.layers.values())
    missing, unexpected = rel.load_state_dict(kv.state_dict(), strict=False)
    assert not unexpected and missing and all(".composer." in k for k in missing)
    _trained(kv)
    rel.load_state_dict(kv.state_dict(), strict=False)
    tokens, mask = _tokens()
    scene = _scene()
    assert torch.equal(_run(kv, tokens, mask, scene), _run(rel, tokens, mask, scene))


def test_a_trained_composer_makes_an_entity_depend_on_the_others():
    torch.manual_seed(0)
    from oct_vla.policies.conditioning.entity import EntityEmbedding

    embed, composer = EntityEmbedding(8, None), RelationalComposer(8)
    with torch.no_grad():
        composer.gamma.fill_(1.0)
    tokens, _ = _tokens(batch=1)
    moved = tokens.clone()
    moved[0, 2, :3] += 0.3  # move entity 2 only
    base, after = composer(embed(tokens), tokens), composer(embed(moved), moved)
    assert not torch.allclose(base[0, 0], after[0, 0])  # entity 0 sees entity 2's move
    # Without the composer, entity 0's embedding is its own business.
    torch.testing.assert_close(embed(tokens)[0, 0], embed(moved)[0, 0])
    # Padding never leaks: changing the padded row changes no valid entity.
    padded = tokens.clone()
    padded[0, -1, :3] = 5.0
    torch.testing.assert_close(composer(embed(padded), padded)[0, :4], base[0, :4])
    assert torch.isfinite(base).all()


def test_gamma_trains_once_the_branch_carries_signal():
    rel = _model("kv_rel").train()
    _trained(rel)
    tokens, mask = _tokens()
    _run(rel, tokens, mask, _scene()).square().sum().backward()
    gammas = [b.composer.gamma.grad for b in rel.object_conditioning.layers.values()]
    assert all(g is not None and g.abs().item() > 0 for g in gammas)


def test_only_groot_takes_kv_rel():
    from oct_vla.policies.control_groot import ControlGrootConfig

    ControlGrootConfig(object_conditioning="kv_rel")
    from oct_vla.policies.object_conditioning import CORE_ARMS

    assert "kv_rel" not in CORE_ARMS
