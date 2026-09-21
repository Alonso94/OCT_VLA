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
