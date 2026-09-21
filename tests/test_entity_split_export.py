"""The production split builder fits entity statistics only on selected training runs."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene
from oct_vla.core.state import ArmState, EEFState
from oct_vla.data import lerobot_export, store

_spec = importlib.util.spec_from_file_location(
    "build_entity_splits", Path(__file__).parents[1] / "scripts/build_shelf_restock_splits.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_full_run_split_normalizes_selected_training_only(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical"
    for seed in (100, 101, 200):
        path = canonical / f"seed_{seed}" / "episode_0"
        path.mkdir(parents=True)
        (path / "episode.json").write_text(json.dumps({"metadata": {"episode_kind": "full_run"}}))
    read_seeds = []

    def read(path):
        seed = _module.seed_of(path)
        read_seeds.append(seed)
        pose = Pose((float(seed), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), WORKCELL_FRAME)
        sample = SimpleNamespace(
            scene=ObjectScene(0.0, ()),
            observation=SimpleNamespace(eef=EEFState(ArmState(pose, 0), ArmState(pose, 1))),
        )
        return SimpleNamespace(samples=[sample])

    captured = {}

    def export(sources, output, **kwargs):
        captured.update(kwargs)
        captured["seeds"] = [_module.seed_of(path) for path in sources]
        output.mkdir()
        return SimpleNamespace(exported=sources, skipped=[], output=output)

    monkeypatch.setattr(store, "read_episode", read)
    monkeypatch.setattr(lerobot_export, "export_episodes", export)
    output = tmp_path / "export"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build",
            "--canonical-root",
            str(canonical),
            "--output",
            str(output),
            "--repo-id",
            "local/test",
            "--entity-tokens",
            "--episode-kind",
            "full_run",
            "--max-runs",
            "1",
        ],
    )
    assert _module.main() == 0
    assert read_seeds == [100]
    assert captured["seeds"] == [100, 200]
    assert captured["entity_normalizer"] is not None
    manifest = json.loads((output / "split_manifest.json").read_text())
    assert manifest["train"]["seeds"] == [100]
    assert manifest["val"]["seeds"] == [200]
    assert len(manifest["entity_normalization_training_sources"]) == 1
