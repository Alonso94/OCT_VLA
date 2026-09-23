"""The live bridge and canonical exporter must agree on raw entity inputs."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from oct_vla.core.frames import Pose
from oct_vla.core.objects import ObjectScene, ObjectState, TaskContext
from oct_vla.core.state import ArmState, EEFState
from oct_vla.data.entity_tokens import build_entity_tokens, shelf_support_entities
from oct_vla.serve.codec import supports_from_json, supports_to_json
from oct_vla.tasks.shelf_restock.spec import DEFAULT_SPEC


def test_support_geometry_round_trip_preserves_entity_inputs():
    supports = shelf_support_entities(DEFAULT_SPEC)
    assert supports_from_json(supports_to_json(supports)) == supports


def test_online_entities_match_shared_builder_and_ignore_oracle_roles():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    spec = importlib.util.spec_from_file_location(
        "eval_entity_policy", Path(__file__).parents[1] / "scripts/eval_shelf_restock_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pose = Pose((0.1, 0.2, 0.9), (0.0, 0.0, 0.0, 1.0))
    scene = ObjectScene(0.0, (ObjectState("a", pose, (0.05, 0.04, 0.1), 1.0, 1.0),))
    eef = EEFState(ArmState(pose, 0.25), ArmState(pose, 0.75))
    supports = shelf_support_entities(DEFAULT_SPEC)
    frame = SimpleNamespace(data=bytes(12), height=2, width=2)
    obs = SimpleNamespace(scene=scene, eef=eef, supports=supports,
                          frame=lambda name: frame,
                          context=TaskContext("restock", "a"))
    actual = module.build_observation(obs, torch=torch, np=np,
                                      entity_max_entities=16)
    expected, mask = build_entity_tokens(scene, eef, supports, 16)
    torch.testing.assert_close(actual["observation.entity_tokens"], torch.tensor([expected]))
    assert torch.equal(actual["observation.entity_mask"], torch.tensor([mask]))
    assert "observation.object_tokens" not in actual
    obs.context = TaskContext("restock", "unobserved-oracle-id", phase="release")
    changed = module.build_observation(obs, torch=torch, np=np,
                                       entity_max_entities=16)
    assert torch.equal(actual["observation.entity_tokens"], changed["observation.entity_tokens"])
    obs.supports = ()
    with pytest.raises(ValueError, match="updated simulator server"):
        module.build_observation(obs, torch=torch, np=np,
                                 entity_max_entities=16)
