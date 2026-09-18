"""Each plugin asserts something about a LeRobot class it does not own: where
the action embedding lives, and what the width attribute is called. A LeRobot
upgrade that moves either would otherwise surface as a silently unconditioned
run -- the hook attaches to nothing, the loss falls, and the object path never
matters. These fail loudly instead."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402

import oct_vla.policies  # noqa: E402,F401  -- registers the plugin types


@pytest.mark.parametrize(
    "policy_type",
    ["control_pi05", "control_smolvla", "control_groot", "control_vla_jepa", "masked_pi05"],
)
def test_the_plugin_types_are_registered(policy_type):
    """An unregistered type fails only when a training job resolves it, which is
    after the queue wait."""
    assert policy_type in PreTrainedConfig._choice_registry


@pytest.mark.parametrize(
    "policy_type",
    ["control_pi05", "control_smolvla", "control_groot", "control_vla_jepa", "masked_pi05"],
)
def test_every_plugin_has_a_rename_map(policy_type):
    """`rename_map_for` raises on unknown types by design; evaluation calls it,
    so a missing entry strands a trained checkpoint."""
    from oct_vla.data.policy_inputs import rename_map_for

    assert isinstance(rename_map_for(policy_type), dict)


def test_every_object_plugin_ships_a_processor_factory():
    """A plugin without one trains fine and then fails at
    `make_pre_post_processors`, which is what evaluation calls -- after the GPU
    hours are spent."""
    import importlib

    for package, factory in (
        ("control_pi05", "make_control_pi05_pre_post_processors"),
        ("control_smolvla", "make_control_smolvla_pre_post_processors"),
        ("control_groot", "make_control_groot_pre_post_processors"),
        ("control_vla_jepa", "make_control_vla_jepa_pre_post_processors"),
    ):
        module = importlib.import_module(
            f"oct_vla.policies.{package}.processor_{package}"
        )
        assert callable(getattr(module, factory))


# ------------------------------------------- upstream structure these rely on


def test_groot_still_embeds_actions_where_the_plugin_expects():
    from lerobot.policies.groot.groot_n1_7 import GR00TN17ActionHead

    from oct_vla.policies.control_groot import ControlGrootPolicy

    assert ControlGrootPolicy.object_module_path == "_groot_model.action_head"
    assert "action_encoder" in GR00TN17ActionHead.__init__.__code__.co_names
    assert "input_embedding_dim" in GR00TN17ActionHead.__init__.__code__.co_names


def test_vla_jepa_still_embeds_actions_where_the_plugin_expects():
    from lerobot.policies.vla_jepa.action_head import VLAJEPAActionHead
    from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAModel

    from oct_vla.policies.control_vla_jepa import ControlVLAJEPAPolicy

    assert ControlVLAJEPAPolicy.object_module_path == "model.action_model"
    assert "action_model" in VLAJEPAModel.__init__.__code__.co_names
    assert "action_encoder" in VLAJEPAActionHead.__init__.__code__.co_names
    assert "input_embedding_dim" in VLAJEPAActionHead.__init__.__code__.co_names


def test_the_pi_family_still_exposes_embed_suffix():
    """pi0.5 and SmolVLA condition by overriding this rather than by a hook."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

    assert hasattr(PI05Pytorch, "embed_suffix")
    assert hasattr(VLAFlowMatching, "embed_suffix")


def test_paths_are_distinct_per_backbone():
    """The bug the refactor fixed: everything assumed `model`."""
    from oct_vla.policies.control_groot import ControlGrootPolicy
    from oct_vla.policies.control_pi05.modeling_control_pi05 import ControlPI05Policy
    from oct_vla.policies.control_vla_jepa import ControlVLAJEPAPolicy

    paths = {
        ControlPI05Policy.object_module_path,
        ControlGrootPolicy.object_module_path,
        ControlVLAJEPAPolicy.object_module_path,
    }
    assert len(paths) == 3
