# ruff: noqa: E402
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
from torch import nn
from torch.nn import functional as F

from oct_vla.policies.conditioning.kv import KVAttention, packed_mha_query


def config():
    return SimpleNamespace(
        object_attention_heads=2, object_entity_normalizer=None
    )


def test_matches_dual_attention_and_bias_is_not_duplicated():
    torch.manual_seed(2)
    host = nn.MultiheadAttention(8, 2, batch_first=True)
    branch = KVAttention(config(), 8, heads=2)
    q, memory, tokens = torch.randn(2, 4, 8), torch.randn(2, 3, 8), torch.randn(2, 5, 17)
    valid = torch.ones(2, 5, dtype=torch.bool)
    native = host(q, memory, memory)[0]
    projected = packed_mha_query(host, q)
    assert torch.equal(
        branch(projected, tokens, valid, output_weight=host.out_proj.weight),
        torch.zeros_like(native),
    )
    nn.init.normal_(branch.to_k.weight)
    nn.init.normal_(branch.to_v.weight)
    z = branch.embedding(tokens)
    k = branch.to_k(z).view(2, 5, 2, 4).transpose(1, 2)
    v = branch.to_v(z).view(2, 5, 2, 4).transpose(1, 2)
    expected = F.scaled_dot_product_attention(projected, k, v)
    expected = expected.transpose(1, 2).reshape(2, 4, 8)
    expected = native + F.linear(expected, host.out_proj.weight)
    actual = native + branch(projected, tokens, valid, output_weight=host.out_proj.weight)
    torch.testing.assert_close(actual, expected)


def test_padding_permutation_empty_repeat_and_reload():
    branch = KVAttention(config(), 8, heads=2)
    nn.init.normal_(branch.to_k.weight)
    nn.init.normal_(branch.to_v.weight)
    q, tokens = torch.randn(4, 2, 3, 4), torch.randn(2, 5, 17)
    mask = torch.tensor([[True, True, False, False, False], [False] * 5])
    tokens[~mask] = float("nan")
    weight = torch.randn(6, 8)
    result = branch(q, tokens, mask, output_weight=weight)
    assert torch.isfinite(result).all()
    assert torch.equal(result[[1, 3]], torch.zeros_like(result[[1, 3]]))
    order = torch.tensor([3, 1, 4, 0, 2])
    torch.testing.assert_close(
        result, branch(q, tokens[:, order], mask[:, order], output_weight=weight)
    )
    explicit = branch(q, tokens.repeat(2, 1, 1), mask.repeat(2, 1), output_weight=weight)
    torch.testing.assert_close(result, explicit)
    restored = KVAttention(config(), 8, heads=2)
    restored.load_state_dict(branch.state_dict())
    torch.testing.assert_close(result, restored(q, tokens, mask, output_weight=weight))
    result.sum().backward()
    assert torch.isfinite(branch.to_k.weight.grad).all()


def test_zero_init_gradient_progression_and_bfloat16():
    branch = KVAttention(config(), 8, heads=2)
    q, tokens = torch.randn(2, 2, 3, 4), torch.randn(2, 4, 17)
    mask = torch.ones(2, 4, dtype=torch.bool)
    weight = torch.randn(8, 8)
    branch(q, tokens, mask, output_weight=weight).sum().backward()
    assert branch.to_v.weight.grad.abs().sum() > 0
    assert branch.to_k.weight.grad.abs().sum() == 0
    assert branch.embedding.numeric_projection.weight.grad.abs().sum() == 0
    with torch.no_grad():
        branch.to_v.weight.add_(branch.to_v.weight.grad, alpha=-0.01)
    branch.zero_grad()
    branch(q, tokens, mask, output_weight=weight).square().sum().backward()
    assert branch.to_k.weight.grad.abs().sum() > 0
    result = branch(q.bfloat16(), tokens, mask, output_weight=weight.bfloat16())
    assert result.dtype == torch.bfloat16 and torch.isfinite(result).all()


def test_object_width_makes_every_branch_low_rank_and_kv_still_starts_at_zero():
    """object_width embeds the entities narrow and projects them up to the host:
    the LoRA-style branch the LoRA-tuned VLAs use. KV must still add an exact
    zero at initialisation, and every projection into the host must be
    object_width -> host width."""
    from oct_vla.policies.conditioning.adaln import SceneAdaLN, SceneVector
    from oct_vla.policies.conditioning.tokens import EntityTokens

    narrow = SimpleNamespace(object_attention_heads=2, object_entity_normalizer=None,
                             object_width=4, object_tokens_gate_init=-8.0)
    host_width = 16
    kv = KVAttention(narrow, host_width, heads=2)
    assert (kv.to_k.in_features, kv.to_k.out_features) == (4, host_width)
    assert (kv.to_v.in_features, kv.to_v.out_features) == (4, host_width)
    host = nn.MultiheadAttention(host_width, 2, batch_first=True)
    q = torch.randn(2, 3, host_width)
    tokens, valid = torch.randn(2, 5, 17), torch.ones(2, 5, dtype=torch.bool)
    out = kv(packed_mha_query(host, q), tokens, valid, output_weight=host.out_proj.weight)
    assert torch.equal(out, torch.zeros_like(out))
    assert SceneAdaLN(narrow, host_width, sites=3).modulation.in_features == 4
    assert SceneVector(narrow, host_width, out_dim=7).projection.in_features == 4
    entity_tokens = EntityTokens(narrow, host_width)
    assert (entity_tokens.projection.in_features, entity_tokens.projection.out_features) == (4, host_width)
    assert entity_tokens.gate.item() == -8.0
    wide = KVAttention(config(), host_width, heads=2)  # object_width absent: the host's width
    assert wide.to_k.in_features == host_width
    assert sum(p.numel() for p in kv.parameters()) < sum(p.numel() for p in wide.parameters()) / 2
