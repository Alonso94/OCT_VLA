# ruff: noqa: E402
"""The VLA in-context and AdaLN branches, on the parts that run without a VLA."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

from oct_vla.policies.conditioning.adaln import SceneVector
from oct_vla.policies.conditioning.tokens import EntityTokens
from oct_vla.policies.conditioning.tokens import prepend_entities


class Config:
    object_entity_normalizer = None
    object_attention_heads = 2


class Control:
    def __init__(self, tokens, mask, width=8):
        self._inputs = (tokens, mask)
        self.incontext = EntityTokens(Config(), width)


def suffix(batch=2, chunk=3, width=8, causal=False):
    embs = torch.randn(batch, chunk, width)
    pad = torch.ones(batch, chunk, dtype=torch.bool)
    att = torch.tensor([1] * chunk if causal else [1] + [0] * (chunk - 1), dtype=torch.float32)
    return embs, pad, att[None].expand(batch, chunk)


@pytest.mark.parametrize("causal", [False, True], ids=["pi05", "smolvla"])
def test_actions_see_entities_and_entities_never_see_actions(causal):
    tokens = torch.randn(2, 4, 17)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    embs, pad, att = suffix(causal=causal)
    prefix_pad = torch.ones(2, 5, dtype=torch.bool)
    prefix_att = torch.zeros(2, 5)
    new_embs, new_pad, new_att = prepend_entities(Control(tokens, mask), embs, pad, att)
    assert new_embs.shape == (2, 4 + 3, 8)
    assert torch.equal(new_embs[:, -3:], embs), "actions stay last, untouched"
    full = make_att_2d_masks(torch.cat([prefix_pad, new_pad], 1), torch.cat([prefix_att, new_att], 1))
    entities, actions = slice(5, 9), slice(9, 12)
    # Row 0 has two real entities.
    assert full[0, actions][:, 5:7].all(), "actions attend to real entities"
    assert not full[0, actions][:, 7:9].any(), "never to padded ones"
    assert not full[0, entities][:, actions].any(), "entities never see noisy actions"
    assert full[0, 5:7][:, :5].all(), "entities see the prefix"
    assert not full[0, 7:9].any(), "padded entities attend to nothing"


def test_no_scene_leaves_the_suffix_exactly_alone():
    embs, pad, att = suffix()
    control = Control(torch.zeros(2, 0, 17), torch.zeros(2, 0, dtype=torch.bool))
    out = prepend_entities(control, embs, pad, att)
    assert all(a is b for a, b in zip(out, (embs, pad, att)))


def test_scene_vector_is_zero_at_init_and_for_an_empty_scene():
    scene = SceneVector(Config(), 8, 12)
    tokens = torch.randn(2, 4, 17)
    mask = torch.tensor([[True, True, False, False], [False] * 4])
    assert torch.equal(scene(tokens, mask, 2), torch.zeros(2, 12))
    with torch.no_grad():
        scene.projection.bias.fill_(1.0)
    out = scene(tokens, mask, 2)
    assert out[0].abs().sum() > 0 and torch.equal(out[1], torch.zeros(12))


def test_scene_vector_gets_gradient_through_its_zero_projection():
    scene = SceneVector(Config(), 8, 12)
    scene(torch.randn(2, 4, 17), torch.ones(2, 4, dtype=torch.bool), 2).sum().backward()
    assert scene.projection.weight.grad.abs().sum() > 0
