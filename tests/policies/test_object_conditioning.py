"""The shared layer: where the module is found, what PEFT keeps, what a batch
must carry, and which pre-cleanup checkpoints still load. Every failure here is
silent in training -- the loss falls whether or not the object path runs."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
nn = torch.nn

from oct_vla.policies.object_conditioning import (  # noqa: E402
    LegacyCheckpointError,
    ObjectConditionedPolicyMixin,
    ObjectConditioning,
    ObjectConditioningError,
    resolve_module,
)

TOKENS, MASK = "observation.entity_tokens", "observation.entity_mask"


class Config:
    object_entity_normalizer = None
    object_attention_heads = 2


class Host(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.object_conditioning = ObjectConditioning(Config(), width)
        self.object_conditioning.add_layer("0", width, 2)


def policy_with(path: str, build):
    class Policy(ObjectConditionedPolicyMixin, nn.Module):
        object_module_path = path

        def __init__(self):
            super().__init__()
            self.config = Config()
            build(self)

    return Policy()


def shallow(p):
    p.model = Host()


def deep(p):
    mid = nn.Module()
    mid.action_model = Host()
    p.model = mid


# ------------------------------------------------------------ the container


def test_a_fresh_module_is_not_live_and_holds_no_inputs():
    oc = Host().object_conditioning
    assert not oc.is_live and oc._inputs is None
    oc.set_inputs(torch.zeros(1, 2, 17), torch.ones(1, 2, dtype=torch.bool))
    oc.clear()
    assert oc._inputs is None


def test_duplicate_layers_are_refused():
    with pytest.raises(ValueError, match="duplicate"):
        Host().object_conditioning.add_layer("0", 8, 2)


# -------------------------------------------------------------- the wiring


def test_the_module_is_found_wherever_a_backbone_mounts_it():
    assert isinstance(policy_with("model", shallow).object_conditioning, ObjectConditioning)
    assert isinstance(
        policy_with("model.action_model", deep).object_conditioning, ObjectConditioning
    )


def test_peft_and_state_prefixes_follow_the_path():
    """`modules_to_save` naming the wrong path is the silent failure: the run
    trains, the loss falls, and the object path is never saved."""
    policy = policy_with("model.action_model", deep)
    targets = policy._get_default_peft_targets()
    assert targets["modules_to_save"] == ["model.action_model.object_conditioning"]
    defaults = policy._prepare_pretrained_state_dict({})
    assert defaults and all(k.startswith("model.action_model.object_conditioning.") for k in defaults)
    assert policy._fresh_state_prefixes() == ("model.action_model.object_conditioning.",)


def test_a_wrong_path_says_which_attribute_is_missing():
    with pytest.raises(ObjectConditioningError, match="has no attribute 'transformer'"):
        _ = policy_with("model.transformer", shallow).object_conditioning


def test_a_host_that_never_built_the_module_is_named():
    def bare(p):
        p.model = nn.Module()

    with pytest.raises(ObjectConditioningError, match="not ObjectConditioning"):
        _ = policy_with("model", bare).object_conditioning


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
    assert targets == {"target_modules": "q_proj", "modules_to_save": ["model.object_conditioning"]}


def test_a_config_declared_lora_target_is_used_only_when_the_backbone_has_none():
    class WithTargets(Config):
        lora_target_modules = r"action_head\..*"

    class Policy(ObjectConditionedPolicyMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.config = WithTargets()
            self.model = Host()

    assert Policy()._get_default_peft_targets()["target_modules"] == r"action_head\..*"


def test_resolve_module_walks_a_dotted_path():
    root, mid, leaf = nn.Module(), nn.Module(), nn.Linear(2, 2)
    mid.leaf = leaf
    root.mid = mid
    assert resolve_module(root, "mid.leaf") is leaf


# ------------------------------------------- the closed-loop path is wrapped


def _recording_policy(fail=False):
    seen = {}

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Host()

        def select_action(self, batch, **kwargs):
            if fail:
                raise RuntimeError("backbone failed")
            seen["inputs"] = self.model.object_conditioning._inputs
            return torch.zeros(1, 1)

    class Policy(ObjectConditionedPolicyMixin, Backbone):
        def __init__(self):
            super().__init__()
            self.config = Config()

    return Policy(), seen


def test_select_action_conditions_even_when_it_skips_predict_action_chunk():
    """SmolVLA's `select_action` bypasses `predict_action_chunk`; wrapping only
    that left it evaluating unconditioned while training conditioned."""
    policy, seen = _recording_policy()
    policy.select_action({TOKENS: torch.randn(8, 17), MASK: torch.ones(8, dtype=torch.bool)})
    assert seen["inputs"][0].shape == (1, 8, 17), "unbatched rollout inputs get a batch dim"
    assert policy.object_conditioning._inputs is None


def test_select_action_clears_its_inputs_when_the_backbone_raises():
    policy, _ = _recording_policy(fail=True)
    with pytest.raises(RuntimeError, match="backbone failed"):
        policy.select_action({TOKENS: torch.randn(2, 8, 17), MASK: torch.ones(2, 8)})
    assert policy.object_conditioning._inputs is None


def test_a_batch_without_entities_is_refused_not_ignored():
    """`batch.get()` returning None would otherwise run the policy unconditioned."""
    policy, _ = _recording_policy()
    with pytest.raises(ValueError, match="needs observation.entity_tokens"):
        policy.select_action({})


# ------------------------------------------------ pre-cleanup checkpoints


def act_config(**fields):
    from oct_vla.policies.control_act import ControlACTConfig

    return ControlACTConfig(device="cpu", **fields)


def test_a_stage_a_checkpoint_config_loads_and_keeps_its_arm():
    """The exact legacy field values every Stage A control_act checkpoint carries."""
    stage_a = dict(
        object_injection_mode="layerwise", object_representation="entity_v2",
        object_token_mode="full", object_entity_positional=False, object_token_shuffle=False,
        object_token_dim=17, object_token_key=TOKENS, object_token_mask_key=MASK,
        object_token_rank_key="observation.object_token_rank", object_queries=4,
    )
    assert act_config(**stage_a).object_conditioning == "kv"
    assert act_config(**stage_a, object_adaln=True).object_conditioning == "kv_adaln"
    tokens = act_config(**stage_a, object_incontext=True, object_incontext_gate_init=-3.0)
    assert tokens.object_conditioning == "kv_tokens" and tokens.object_tokens_gate_init == -3.0
    assert tokens.object_injection_mode is None and tokens.object_incontext is None


@pytest.mark.parametrize("field,value", [
    ("object_injection_mode", "pooled"),
    ("object_injection_mode", "controlvla"),
    ("object_representation", "legacy"),
    ("object_token_mode", "role_stripped"),
    ("object_entity_positional", True),
])
def test_a_removed_mechanism_is_refused_with_the_tag_to_use(field, value):
    with pytest.raises(LegacyCheckpointError, match="stageA-2026-09-23"):
        act_config(**{field: value})


def test_an_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="object_conditioning must be one of"):
        act_config(object_conditioning="adaln")
