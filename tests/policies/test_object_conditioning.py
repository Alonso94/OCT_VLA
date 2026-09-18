"""ControlVLA's guarantee is that a conditioned policy is *numerically* its
pretrained self at step 0, so any later difference is evidence the object tokens
are used. That property is silent when it breaks -- a missing token key returns
None rather than raising, and the loss falls either way -- so it is pinned here
along with the wiring that used to be hardcoded to `self.model`."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from oct_vla.policies.object_conditioning import (  # noqa: E402
    ObjectConditionedPolicyMixin,
    ObjectConditioning,
    ObjectConditioningError,
    resolve_module,
)


class Config:
    object_token_dim = 15
    object_token_mode = "full"
    object_queries = 4
    object_attention_heads = 8
    object_token_key = "observation.object_tokens"
    object_token_mask_key = "observation.object_token_mask"
    object_token_rank_key = "observation.object_token_rank"
    object_token_shuffle = False

    @property
    def effective_object_token_dim(self) -> int:
        from oct_vla.data.token_transforms import token_dim_for_mode

        return token_dim_for_mode(self.object_token_dim, self.object_token_mode)


def conditioning(width=64):
    return ObjectConditioning(Config(), width)


# ------------------------------------------------------------ the guarantee


def test_the_residual_is_identity_at_initialisation():
    """The whole comparability argument: at step 0 the conditioned arm and the
    unconditioned one are the same function."""
    oc = conditioning()
    emb = torch.randn(2, 10, 64)
    oc.set_inputs(torch.randn(2, 8, 15), torch.ones(2, 8))
    assert torch.equal(oc.residual(emb), emb)
    assert not oc.is_live


def test_the_residual_moves_once_the_injection_trains():
    oc = conditioning()
    emb = torch.randn(2, 10, 64)
    oc.set_inputs(torch.randn(2, 8, 15), torch.ones(2, 8))
    torch.optim.SGD(oc.parameters(), lr=1.0).zero_grad()
    oc.residual(emb).pow(2).mean().backward()
    with torch.no_grad():
        oc.injection.weight -= oc.injection.weight.grad
    assert oc.is_live
    assert not torch.equal(oc.residual(emb), emb)


def test_without_tokens_the_residual_is_a_no_op():
    """So one code path serves the conditioned and unconditioned arms, and a
    forward that never got tokens cannot silently inject stale ones."""
    oc = conditioning()
    emb = torch.randn(2, 10, 64)
    assert torch.equal(oc.residual(emb), emb)
    oc.set_inputs(torch.randn(2, 8, 15), torch.ones(2, 8))
    oc.clear()
    assert torch.equal(oc.residual(emb), emb)


def test_the_context_is_broadcast_over_the_whole_chunk():
    """The scene summary conditions the chunk, not one step of it."""
    oc = conditioning()
    with torch.no_grad():
        oc.injection.weight.fill_(0.01)
    oc.set_inputs(torch.randn(1, 8, 15), torch.ones(1, 8))
    delta = oc.residual(torch.zeros(1, 5, 64))
    assert torch.allclose(delta[0, 0], delta[0, 4])


# -------------------------------------------------------------- the wiring


class Host(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.object_conditioning = conditioning(width)


def policy_with(path: str, depth_builder):
    class Policy(ObjectConditionedPolicyMixin, nn.Module):
        object_module_path = path

        def __init__(self):
            super().__init__()
            self.config = Config()
            depth_builder(self)

    return Policy()


def test_the_module_is_found_wherever_a_backbone_mounts_it():
    """pi0.5 and SmolVLA hold it at `model`; X-VLA at `model.transformer`,
    VLA-JEPA at `model.action_model`. The old code assumed `model`."""
    def shallow(p):
        p.model = Host()

    def deep(p):
        mid = nn.Module()
        mid.action_model = Host()
        p.model = mid

    assert isinstance(policy_with("model", shallow).object_conditioning, ObjectConditioning)
    assert isinstance(
        policy_with("model.action_model", deep).object_conditioning, ObjectConditioning
    )


def test_peft_and_state_prefixes_follow_the_path():
    """`modules_to_save` naming the wrong path is the silent failure: the run
    trains, the loss falls, and the object path is never saved."""
    def deep(p):
        mid = nn.Module()
        mid.action_model = Host()
        p.model = mid

    policy = policy_with("model.action_model", deep)
    targets = policy._add_object_peft_targets({})
    assert targets["modules_to_save"] == ["model.action_model.object_conditioning"]
    defaults = policy._add_object_state_defaults({})
    assert defaults
    assert all(k.startswith("model.action_model.object_conditioning.") for k in defaults)


def test_a_wrong_path_says_which_attribute_is_missing():
    """Not an AttributeError: nn.Module.__getattr__ intercepts those and
    replaces the message with its own, discarding the one detail that matters."""
    def shallow(p):
        p.model = Host()

    policy = policy_with("model.transformer", shallow)
    with pytest.raises(ObjectConditioningError, match="has no attribute 'transformer'"):
        _ = policy.object_conditioning


def test_a_host_that_never_built_the_module_is_named():
    def bare(p):
        p.model = nn.Module()

    policy = policy_with("model", bare)
    with pytest.raises(ObjectConditioningError, match="not ObjectConditioning"):
        _ = policy.object_conditioning


def test_hooks_chain_only_when_the_backbone_defines_them():
    """The SmolVLA bug: it called super()._prepare_pretrained_state_dict, which
    only PI05Policy defines, so the object params were never defaulted in."""
    def shallow(p):
        p.model = Host()

    policy = policy_with("model", shallow)   # nn.Module has neither hook
    assert policy._prepare_pretrained_state_dict({})     # does not raise
    assert policy._get_default_peft_targets()["modules_to_save"]


def test_a_backbone_hook_is_still_chained_when_present():
    class WithHooks(nn.Module):
        def _prepare_pretrained_state_dict(self, state_dict):
            return {**state_dict, "backbone.seen": torch.zeros(1)}

        def _get_default_peft_targets(self):
            return {"target_modules": "q_proj"}

    class Policy(ObjectConditionedPolicyMixin, WithHooks):
        def __init__(self):
            super().__init__()
            self.config = Config()
            self.model = Host()

    policy = Policy()
    assert "backbone.seen" in policy._prepare_pretrained_state_dict({})
    targets = policy._get_default_peft_targets()
    assert targets["target_modules"] == "q_proj"
    assert targets["modules_to_save"] == ["model.object_conditioning"]


def test_resolve_module_walks_a_dotted_path():
    root = nn.Module()
    mid = nn.Module()
    leaf = nn.Linear(2, 2)
    mid.leaf = leaf
    root.mid = mid
    assert resolve_module(root, "mid.leaf") is leaf
    assert resolve_module(root, "mid") is mid
