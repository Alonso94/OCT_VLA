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


# ------------------------------------------- information and composition controls


def test_a_derangement_never_leaves_a_sample_its_own_scene():
    from oct_vla.policies.object_conditioning import derangement

    torch.manual_seed(0)
    for size in range(2, 10):
        for _ in range(50):
            target = derangement(size)
            assert sorted(target.tolist()) == list(range(size))
            assert not (target == torch.arange(size)).any()
    with pytest.raises(ValueError, match="at least 2"):
        derangement(1)


def _shuffled_policy():
    policy, _ = _recording_policy()
    policy.config.object_conditioning = "kv_adaln_shuffled"
    return policy


def test_the_shuffled_arm_trains_on_other_scenes_and_evaluates_on_its_own():
    """The information control: in training every sample's entities come from a
    different sample; at evaluation the policy sees its own scene."""
    policy = _shuffled_policy()
    tokens = torch.arange(4, dtype=torch.float32)[:, None, None].expand(4, 3, 17).clone()
    mask = torch.ones(4, 3, dtype=torch.bool)
    policy.train()
    policy._set_object_inputs({TOKENS: tokens, MASK: mask})
    seen = policy.object_conditioning._inputs[0][:, 0, 0]
    assert sorted(seen.tolist()) == [0.0, 1.0, 2.0, 3.0]
    assert not (seen == torch.arange(4, dtype=torch.float32)).any()
    policy.eval()
    policy._set_object_inputs({TOKENS: tokens, MASK: mask})
    assert torch.equal(policy.object_conditioning._inputs[0], tokens)


def test_other_arms_are_never_shuffled():
    policy, _ = _recording_policy()
    policy.config.object_conditioning = "kv_adaln"
    tokens = torch.randn(4, 3, 17)
    policy.train()
    policy._set_object_inputs({TOKENS: tokens, MASK: torch.ones(4, 3, dtype=torch.bool)})
    assert torch.equal(policy.object_conditioning._inputs[0], tokens)


@pytest.mark.parametrize("arm", ["kv_adaln_shuffled", "scene_attn", "scene_mean"])
def test_a_host_without_the_control_arms_refuses_them(arm):
    """ACT (and pi0.5, SmolVLA, VLA-JEPA) implement only the nested arms; building
    a control there must fail, not silently train a different arm."""
    from oct_vla.policies.object_conditioning import CORE_ARMS, require_arm

    config = act_config(object_conditioning=arm)
    with pytest.raises(ValueError, match="does not implement"):
        require_arm(config, CORE_ARMS, "control_act")


def test_visual_columns_start_inert_and_are_read_once_trained():
    """object_visual_dim appends a frozen encoder's per-object feature to each
    entity; its projection starts at zero, so geometry + vision begins exactly
    as the geometry-only embedding does."""
    from oct_vla.policies.conditioning.entity import EntityEmbedding

    torch.manual_seed(0)
    geometry = EntityEmbedding(8)
    torch.manual_seed(0)
    vision = EntityEmbedding(8, visual_dim=4)
    tokens, visual = torch.randn(2, 3, 17), torch.randn(2, 3, 4)
    both = torch.cat([tokens, visual], dim=-1)
    assert torch.equal(vision(both), geometry(tokens))
    with torch.no_grad():
        vision.visual_projection.weight.normal_()
    assert not torch.allclose(vision(both), geometry(tokens))
    with pytest.raises(ValueError, match="17 geometry columns \\+ 4 visual"):
        vision(tokens)


def test_a_vision_arm_without_visual_inputs_is_refused():
    policy, _ = _recording_policy()
    policy.config.object_visual_dim = 4
    batch = {TOKENS: torch.randn(2, 3, 17), MASK: torch.ones(2, 3, dtype=torch.bool)}
    with pytest.raises(ValueError, match="needs observation.entity_visual"):
        policy._set_object_inputs(batch)
    batch["observation.entity_visual"] = torch.randn(2, 3, 5)
    with pytest.raises(ValueError, match="expected"):
        policy._set_object_inputs(batch)
    batch["observation.entity_visual"] = torch.randn(2, 3, 4)
    policy._set_object_inputs(batch)
    assert policy.object_conditioning._inputs[0].shape == (2, 3, 21)


def test_the_sigreg_term_uses_one_copy_of_the_movable_objects_only():
    """kv_adaln_sigreg regularises the movable objects' embeddings: GR00T tiles
    the entity batch, and duplicates or constant supports would bias the test."""
    from types import SimpleNamespace

    from oct_vla.perception.lejepa.sigreg import sigreg
    from oct_vla.policies.control_groot.modeling_control_groot import ControlGrootPolicy

    torch.manual_seed(0)
    unique, entities, width = 8, 5, 6
    tokens = torch.zeros(unique, entities, 17)
    tokens[:, :3, 13] = 1.0  # three movables
    tokens[:, 3:, 16] = 1.0  # two supports
    embedded = torch.randn(unique, entities, width)
    embedded[:, 3:] = 5.0  # constant support rows: would dominate if included
    control = SimpleNamespace(_inputs=(tokens, None))
    policy = SimpleNamespace(_control=lambda: control,
                             _embeddings=[(tokens.repeat(2, 1, 1), embedded.repeat(2, 1, 1))])
    torch.manual_seed(1)
    term = ControlGrootPolicy._object_sigreg(policy)
    torch.manual_seed(1)  # the same random directions
    expected = sigreg(embedded[:, :3].reshape(-1, width))
    torch.testing.assert_close(term, expected)
    one_movable = SimpleNamespace(_control=lambda: SimpleNamespace(_inputs=(tokens[:1], None)),
                                  _embeddings=[(tokens[:1, 2:], embedded[:1, 2:])])
    with pytest.raises(RuntimeError, match="two movable"):
        ControlGrootPolicy._object_sigreg(one_movable)


def test_a_sigreg_arm_with_no_embeddings_is_an_error_not_a_zero():
    from types import SimpleNamespace

    from oct_vla.policies.control_groot.modeling_control_groot import ControlGrootPolicy

    with pytest.raises(RuntimeError, match="ran no entity embedding"):
        ControlGrootPolicy._object_sigreg(SimpleNamespace(_control=lambda: None, _embeddings=[]))
