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


def _paired_corpus(root, model_ids):
    """A paired collection: each seed holds two clips and the full run."""
    for seed, model_id in model_ids.items():
        for name, kind in (("episode_0", "atomic_restock"), ("episode_1", "atomic_restock"),
                           ("episode_full", "full_run")):
            path = root / f"seed_{seed}" / name
            path.mkdir(parents=True)
            metadata = {"episode_kind": kind, "model_ids": str(model_id), "paired_run": "true"}
            (path / "episode.json").write_text(json.dumps({"metadata": metadata}))


def _build(monkeypatch, tmp_path, canonical, name, *extra):
    captured = {}

    def export(sources, output, **kwargs):
        captured["sources"] = list(sources)
        output.mkdir()
        return SimpleNamespace(exported=sources, skipped=[], output=output)

    monkeypatch.setattr(lerobot_export, "export_episodes", export)
    output = tmp_path / name
    monkeypatch.setattr("sys.argv", ["build", "--canonical-root", str(canonical),
                                     "--output", str(output), "--repo-id", "local/t", *extra])
    assert _module.main() == 0
    return captured["sources"], json.loads((output / "split_manifest.json").read_text())


def test_paired_views_hold_the_same_runs_and_a_declared_holdout(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical"
    _paired_corpus(canonical, {100: 1, 101: 6, 102: 3, 200: 2, 201: 0})
    atomic, a = _build(monkeypatch, tmp_path, canonical, "a", "--episode-kind", "atomic",
                       "--holdout-model-ids", "0,5,6")
    full, f = _build(monkeypatch, tmp_path, canonical, "f", "--episode-kind", "full_run",
                     "--holdout-model-ids", "0,5,6")
    assert {p.name for p in full} == {"episode_full"}
    assert {p.name for p in atomic} == {"episode_0", "episode_1"}
    for split in ("train", "val"):
        assert a[split]["seeds"] == f[split]["seeds"]
    assert a["train"]["seeds"] == [100, 102] and a["val"]["seeds"] == [200]
    assert a["identity_holdout"]["seeds"] == [101, 201]
    assert a["identity_holdout"]["model_ids"] == [0, 5, 6]


def test_a_paired_corpus_refuses_to_export_both_kinds_at_once(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical"
    _paired_corpus(canonical, {100: 1, 200: 2})
    monkeypatch.setattr("sys.argv", ["build", "--canonical-root", str(canonical),
                                     "--output", str(tmp_path / "x"), "--repo-id", "local/t"])
    import pytest

    with pytest.raises(SystemExit, match="mixes episode kinds"):
        _module.main()
