"""Training launchers consume the export contract without fitting new statistics."""

import importlib.util
import json
from pathlib import Path

import pytest

from oct_vla.data.entity_tokens import ENTITY_TOKEN_SCHEMA

_spec = importlib.util.spec_from_file_location(
    "entity_training_args", Path(__file__).parents[1] / "scripts/entity_training_args.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def _dataset(tmp_path, **overrides):
    stats = {
        "schema": ENTITY_TOKEN_SCHEMA,
        "position_mean": [1, 2, 3],
        "position_std": [2, 3, 4],
        "size_mean": [0, 0, 0],
        "size_std": [1, 1, 1],
    }
    metadata = dict(
        schema=ENTITY_TOKEN_SCHEMA, token_dim=17, capacity=8, raw_on_disk=True, normalization=stats
    )
    metadata.update(overrides)
    info = {
        "entity_tokens": metadata,
        "features": {
            "observation.entity_tokens": {"shape": [8, 17]},
            "observation.entity_mask": {"shape": [8]},
        },
    }
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    return tmp_path


def test_training_uses_exported_statistics(tmp_path):
    args = _module.entity_training_args(_dataset(tmp_path))
    normalizer = json.loads(next(x.split("=", 1)[1] for x in args if "normalizer=" in x))
    assert normalizer["position_mean"] == [1, 2, 3]
    assert "--policy.object_max_entities=8" in args


@pytest.mark.parametrize(
    "changes",
    [{"normalization": None}, {"raw_on_disk": False}, {"capacity": 9}, {"schema": "legacy"}],
)
def test_malformed_or_unfitted_export_is_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        _module.entity_training_args(_dataset(tmp_path, **changes))


def test_a_keyframe_view_switches_the_future_branch_on(tmp_path):
    dataset = _dataset(tmp_path)
    info = json.loads((dataset / "meta/info.json").read_text())
    info["features"]["observation.future_keyframe"] = {"shape": [3, 24]}
    info["features"]["observation.future_keyframe_mask"] = {"shape": [3]}
    (dataset / "meta/info.json").write_text(json.dumps(info))
    args = _module.entity_training_args(dataset)
    assert "--policy.object_future=true" in args and "--policy.object_future_rows=3" in args


def test_a_keyframe_view_with_the_wrong_width_is_refused(tmp_path):
    dataset = _dataset(tmp_path)
    info = json.loads((dataset / "meta/info.json").read_text())
    info["features"]["observation.future_keyframe"] = {"shape": [3, 17]}
    info["features"]["observation.future_keyframe_mask"] = {"shape": [3]}
    (dataset / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="future_keyframe"):
        _module.entity_training_args(dataset)


def test_views_without_a_keyframe_leave_the_branch_off(tmp_path):
    assert not any("object_future" in a for a in _module.entity_training_args(_dataset(tmp_path)))


def test_a_latent_view_selects_the_latent_kind_at_the_priors_width(tmp_path):
    dataset = _dataset(tmp_path)
    info = json.loads((dataset / "meta/info.json").read_text())
    info["features"]["observation.future_keyframe"] = {"shape": [10, 128]}
    info["features"]["observation.future_keyframe_mask"] = {"shape": [10]}
    info["future_keyframe"] = {"source": "pfn", "kind": "latent"}
    (dataset / "meta/info.json").write_text(json.dumps(info))
    args = _module.entity_training_args(dataset)
    assert "--policy.object_future_rows=10" in args
    assert "--policy.object_future_kind=latent" in args
    assert "--policy.object_future_dim=128" in args


def test_a_latent_shaped_view_that_says_keyframe_is_refused(tmp_path):
    dataset = _dataset(tmp_path)
    info = json.loads((dataset / "meta/info.json").read_text())
    info["features"]["observation.future_keyframe"] = {"shape": [10, 128]}
    info["features"]["observation.future_keyframe_mask"] = {"shape": [10]}
    (dataset / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="keyframe view"):
        _module.entity_training_args(dataset)
