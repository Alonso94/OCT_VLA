"""Tensor-layout regressions for native-query layerwise hooks.

These use a tiny module with the same ``to_q``/``to_out`` contract as a
diffusers attention block, so they run without the optional policy packages.
"""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from oct_vla.policies.conditioning.hosts import install_diffusers_kv  # noqa: E402


class _Branch(nn.Module):
    def forward(self, query, *_inputs, output_weight=None, scaling=None):
        del output_weight, scaling
        return query.transpose(1, 2).flatten(2)


class _Control:
    def __init__(self):
        self._inputs = (torch.ones(1, 1, 1), torch.ones(1, 1, dtype=torch.bool))
        self.layers = nn.ModuleDict()

    def add_layer(self, name, width, heads):
        assert (name, width, heads) == ("0", 4, 2)
        self.layers[name] = _Branch()
        return self.layers[name]


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4, bias=False)
        nn.init.eye_(self.to_q.weight)
        self.to_out = nn.ModuleList([nn.Identity()])
        self.heads = 2
        self.scale = None

    def forward(self, value):
        return self.to_out[0](self.to_q(value))


def test_diffusers_hook_restores_flattened_native_layout_before_to_out():
    attention = _Attention()
    holder = nn.Module()
    holder.add_module("attention", attention)
    install_diffusers_kv(holder, _Control())
    value = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)

    # The branch returns the post-output-projection [B,S,H*D] residual.
    # The hook must add that exact native layout before to_out.
    assert torch.equal(attention(value), 2 * value)


def test_native_groot_processor_masks_temporal_padding_and_inactive_action_width():
    from lerobot.lerobot_types import TransitionKey
    from lerobot.policies.groot.processor_groot import GrootN17PackInputsStep

    step = GrootN17PackInputsStep(
        action_horizon=3, valid_action_horizon=3, max_action_dim=4,
        max_state_dim=4, normalize_min_max=False,
    )
    transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.ACTION: torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
        TransitionKey.COMPLEMENTARY_DATA: {"action_is_pad": torch.tensor([[False, True, False]])},
    }
    packed = step(transition)
    mask = packed[TransitionKey.COMPLEMENTARY_DATA]["action_mask"]

    assert torch.equal(mask[0, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.equal(mask[0, 1], torch.zeros(4))
    assert torch.equal(mask[0, 2], torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_active_peft_modules_to_save_copy_replaces_the_frozen_hook_target():
    from peft.utils.other import ModulesToSaveWrapper
    from oct_vla.policies.conditioning.hosts import _active_control
    from oct_vla.policies.object_conditioning import ObjectConditioning

    class Config:
        object_attention_heads = 2
        object_entity_normalizer = None

    original = ObjectConditioning(Config(), 4)
    original.add_layer("0", 4, 2)
    wrapped = ModulesToSaveWrapper(original, "default")
    owner = nn.Module()
    owner.add_module("object_conditioning", wrapped)

    active = _active_control(owner, original)
    assert active is wrapped.modules_to_save["default"]
    assert active is not original
    assert not next(original.layers["0"].parameters()).requires_grad
    assert next(active.layers["0"].parameters()).requires_grad

    wrapped.enable_adapters(False)
    assert _active_control(owner, original) is original
