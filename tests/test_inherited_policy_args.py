"""A conditioned arm must inherit every non-default field of its stage-1 config."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("lerobot")

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("inherited", ROOT / "scripts/inherited_policy_args.py")
inherited = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inherited)


def test_smolvla_hub_settings_carry_over_and_caller_fields_do_not():
    from oct_vla.policies.control_smolvla.configuration_control_smolvla import (
        ControlSmolVLAConfig,
    )

    source = {"type": "smolvla", "load_vlm_weights": True, "pad_language_to": "max_length",
              "n_action_steps": 50, "chunk_size": 50, "object_conditioning": "kv",
              "device": "cuda", "unknown_field": 1}
    flags = inherited.inherited(source, ControlSmolVLAConfig)
    assert "--policy.load_vlm_weights=true" in flags
    assert "--policy.pad_language_to=max_length" in flags
    # Equal to the default, owned by the caller, or not a field: none emitted.
    assert not [f for f in flags if f.split("=")[0].split(".")[1] in
                ("chunk_size", "n_action_steps", "object_conditioning", "device",
                 "unknown_field", "type")]


def test_flags_parse_back_into_the_same_values():
    import draccus
    from oct_vla.policies.control_smolvla.configuration_control_smolvla import (
        ControlSmolVLAConfig,
    )

    source = {"load_vlm_weights": True, "pad_language_to": "max_length", "prefix_length": 0}
    args = [f.removeprefix("--policy.") for f in inherited.inherited(source, ControlSmolVLAConfig)]
    config = draccus.parse(ControlSmolVLAConfig, args=[f"--{a}" for a in args])
    assert config.load_vlm_weights is True
    assert config.pad_language_to == "max_length"
    assert config.prefix_length == 0
