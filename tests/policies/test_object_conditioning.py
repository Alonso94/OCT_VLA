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


def test_a_config_declared_lora_target_is_used_when_the_backbone_has_none():
    """GR00T and VLA-JEPA define no default targets. LeRobot refuses the run
    rather than attaching LoRA to nothing, so the config must supply them."""
    class WithTargets(Config):
        lora_target_modules = r"action_head\..*"

    class Policy(ObjectConditionedPolicyMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.config = WithTargets()
            self.model = Host()

    targets = Policy()._get_default_peft_targets()
    assert targets["target_modules"] == r"action_head\..*"
    assert targets["modules_to_save"] == ["model.object_conditioning"]


def test_a_backbone_default_wins_over_the_config_declaration():
    """pi0.5 and SmolVLA ship targets tuned to their own module names; a config
    field must not silently override them."""
    class WithTargets(Config):
        lora_target_modules = r"should_not_win"

    class Parent(nn.Module):
        def _get_default_peft_targets(self):
            return {"target_modules": "backbone_default"}

    class Policy(ObjectConditionedPolicyMixin, Parent):
        def __init__(self):
            super().__init__()
            self.config = WithTargets()
            self.model = Host()

    assert Policy()._get_default_peft_targets()["target_modules"] == "backbone_default"


# ------------------------------------------- the closed-loop path is wrapped


def test_select_action_conditions_even_when_it_skips_predict_action_chunk():
    """SmolVLA's `select_action` calls `_get_action_chunk` directly rather than
    `predict_action_chunk`, so wrapping only the latter left that arm training
    *with* conditioning and evaluating *without* it -- silently, because the
    training loss falls normally and the rollout is simply the unconditioned
    policy's. This pins the wrap at the mixin, where a backbone's internal
    routing cannot route around it."""
    seen = {}

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Host()

        def select_action(self, batch, **kwargs):
            # Stands in for the backbone reaching the network by its own route.
            seen["inputs"] = self.model.object_conditioning._inputs
            return torch.zeros(1, 1)

    class Policy(ObjectConditionedPolicyMixin, Backbone):
        object_module_path = "model"

        def __init__(self):
            super().__init__()
            self.config = Config()

    policy = Policy()
    tokens = torch.randn(2, 8, 15)
    mask = torch.ones(2, 8)
    policy.select_action({
        "observation.object_tokens": tokens,
        "observation.object_token_mask": mask,
    })
    assert seen["inputs"] is not None, "select_action reached the backbone unconditioned"
    assert seen["inputs"][0].shape == tokens.shape
    # Cleared afterwards, so one step's tokens cannot leak into the next.
    assert policy.object_conditioning._inputs is None


def test_select_action_clears_its_inputs_when_the_backbone_raises():
    """A raised exception must not leave one observation's tokens attached to
    the next, or a recovered episode silently conditions on a stale scene."""
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Host()

        def select_action(self, batch, **kwargs):
            raise RuntimeError("backbone failed")

    class Policy(ObjectConditionedPolicyMixin, Backbone):
        object_module_path = "model"

        def __init__(self):
            super().__init__()
            self.config = Config()

    policy = Policy()
    with pytest.raises(RuntimeError, match="backbone failed"):
        policy.select_action({"observation.object_tokens": torch.randn(2, 8, 15)})
    assert policy.object_conditioning._inputs is None


# ------------------------------------------------- the published ControlVLA


class ControlVLAConfig(Config):
    object_injection_mode = "controlvla"
    object_entity_positional = False
    object_max_entities = 16


def controlvla(width=64, **overrides):
    cfg = ControlVLAConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return ObjectConditioning(cfg, width)


def make_live(oc):
    """Move the zero-initialised KV projections off zero, as training would."""
    with torch.no_grad():
        oc.attention.to_k.weight.normal_(std=0.5)
        oc.attention.to_v.weight.normal_(std=0.5)
    return oc


def test_controlvla_zero_init_is_identity():
    """The paper's guarantee: zero-initialised K_z and V_z make the conditioned
    policy numerically its pretrained self at step 0."""
    oc = controlvla()
    emb = torch.randn(2, 10, 64)
    oc.set_inputs(torch.randn(2, 8, 15), torch.ones(2, 8))
    assert torch.equal(oc.residual(emb), emb)
    assert not oc.is_live


def test_controlvla_conditions_each_action_token_separately():
    """The property the pooled mode structurally cannot have. ControlVLA's added
    term is softmax(QK_z)V_z, so action token i computes its own weights over the
    objects; the pooled residual adds one identical bias to every token, which
    means it cannot attend to the grasp object and the place target at different
    points in the same chunk."""
    emb = torch.randn(2, 10, 64)
    tokens, mask = torch.randn(2, 8, 15), torch.ones(2, 8)

    oc = make_live(controlvla())
    oc.set_inputs(tokens, mask)
    delta = oc.residual(emb) - emb
    spread = (delta - delta.mean(dim=1, keepdim=True)).abs().max().item()
    assert spread > 1e-3, "controlvla added the same vector to every action token"

    pooled = conditioning()
    with torch.no_grad():
        pooled.injection.weight.normal_(std=0.5)
    pooled.set_inputs(tokens, mask)
    pooled_delta = pooled.residual(emb) - emb
    assert torch.allclose(
        pooled_delta, pooled_delta.mean(dim=1, keepdim=True), atol=1e-5
    ), "the pooled mode is supposed to be a broadcast bias; this test is stale"


def test_controlvla_masks_padding_and_survives_an_empty_scene():
    """Five of eight slots are empty in a three_object scene. Unmasked, the
    encoder attends to zero rows as though they were objects; all-masked, the
    softmax divides by zero and NaN poisons the batch."""
    oc = make_live(controlvla())
    emb = torch.randn(2, 6, 64)
    tokens = torch.randn(2, 8, 15)
    present = torch.zeros(2, 8)
    present[:, :3] = 1

    oc.set_inputs(tokens, present)
    masked = oc.residual(emb)
    # Padding rows must not affect the result: change them and nothing moves.
    scrambled = tokens.clone()
    scrambled[:, 3:] = torch.randn_like(scrambled[:, 3:]) * 10
    oc.set_inputs(scrambled, present)
    assert torch.allclose(masked, oc.residual(emb), atol=1e-5)

    oc.set_inputs(tokens, torch.zeros(2, 8))
    assert torch.isfinite(oc.residual(emb)).all(), "an empty scene produced NaN"


def test_entity_positional_is_what_makes_the_shuffle_control_valid():
    """Off by default the encoder is permutation-invariant, which is why the
    shuffled-token control has been a no-op. The flag is the fix, and it is
    opt-in so the production arms stay invariant."""
    emb = torch.randn(2, 10, 64)
    tokens, mask = torch.randn(2, 8, 15), torch.ones(2, 8)
    order = torch.randperm(8)

    invariant = make_live(controlvla())
    invariant.set_inputs(tokens, mask)
    before = invariant.residual(emb)
    invariant.set_inputs(tokens[:, order], mask[:, order])
    assert torch.allclose(before, invariant.residual(emb), atol=1e-5)

    ordered = make_live(controlvla(object_entity_positional=True))
    with torch.no_grad():
        ordered.embedding.entity_code.normal_(std=0.5)
    ordered.set_inputs(tokens, mask)
    before = ordered.residual(emb)
    ordered.set_inputs(tokens[:, order], mask[:, order])
    assert not torch.allclose(before, ordered.residual(emb), atol=1e-5)


def test_an_unknown_injection_mode_is_refused():
    with pytest.raises(ValueError, match="object_injection_mode"):
        controlvla(object_injection_mode="dual_attention")


def test_heads_must_divide_the_host_width():
    """A backbone whose width is not a multiple of the head count would
    otherwise fail inside a reshape, far from the cause."""
    with pytest.raises(ValueError, match="does not divide"):
        controlvla(width=100)
