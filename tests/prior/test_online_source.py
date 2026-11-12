"""The learned future source the evaluation client runs (prior/online.py)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oct_vla.prior import shelf_adapter as A  # noqa: E402
from oct_vla.prior.generator import N_MAX  # noqa: E402
from oct_vla.prior.model import KeyframePFN  # noqa: E402
from oct_vla.prior.online import PriorFutureSource  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_shelf_adapter import SUPPORTS, _demo, _eef, _scene  # noqa: E402


def _view(tmp_path, kind="keyframe", demos=4):
    torch.manual_seed(0)
    model = KeyframePFN(width=16, heads=2, layers=1)
    run = tmp_path / "prior"
    run.mkdir()
    torch.save({"config": model.config, "state": model.state_dict()}, run / "model.pt")
    pool = [A.support_demo(_demo(), SUPPORTS)] * demos
    (tmp_path / "meta").mkdir()
    np.savez_compressed(tmp_path / "meta/pfn_support.npz", episode_index=np.arange(demos),
                        seed=np.arange(100, 100 + demos),
                        **{f: np.stack([getattr(d, f) for d in pool])
                           for f in ("feats", "types", "rows", "event", "gripper", "flag")})
    return {"source": "pfn", "kind": kind, "prior": str(run), "prior_config": model.config,
            "lang": None, "k": 3, "support_seed": 0, "support_pool": "meta/pfn_support.npz",
            "instruction": A.INSTRUCTION, "eef_to_grip": A.EEF_TO_GRIP}


def _observation(**positions):
    # Deliberately no `context`: the learned source must not need the task manager.
    return SimpleNamespace(scene=_scene(**positions), eef=_eef(), supports=SUPPORTS)


START = {"a": (-0.3, -0.25, 0.79), "b": (0.1, -0.25, 0.79)}


def test_keyframe_rows_from_an_observation_without_task_context(tmp_path):
    source = PriorFutureSource(_view(tmp_path), tmp_path)
    source.reset(800, _observation(**START))
    rows, mask, conf = source(_observation(**START))
    assert rows.shape == (3, 24) and mask.all() and np.isfinite(rows).all()
    assert 0.0 < conf <= 1.0


def test_latent_cells_have_the_priors_shape(tmp_path):
    source = PriorFutureSource(_view(tmp_path, "latent"), tmp_path)
    source.reset(800, _observation(**START))
    cells, mask, _ = source(_observation(**START))
    assert cells.shape == (N_MAX + 1, 16) and mask.sum() == 7  # 2 movables, 2 grippers, 2 decks, event


def test_supports_are_drawn_per_scene_and_reproducibly(tmp_path):
    view = _view(tmp_path, demos=8)
    a, b = PriorFutureSource(view, tmp_path), PriorFutureSource(view, tmp_path)
    assert a.reset(801, _observation(**START)) == b.reset(801, _observation(**START))
    draws = {tuple(a.reset(seed, _observation(**START))) for seed in range(800, 820)}
    assert len(draws) > 1 and all(len(d) == 3 for d in draws)


def test_a_different_prior_or_adapter_is_refused(tmp_path):
    view = _view(tmp_path)
    with pytest.raises(ValueError, match="not the prior"):
        PriorFutureSource({**view, "prior_config": {**view["prior_config"], "width": 32}},
                          tmp_path)
    with pytest.raises(ValueError, match="adapter"):
        PriorFutureSource({**view, "eef_to_grip": 0.1}, tmp_path)


def test_calling_before_reset_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="reset"):
        PriorFutureSource(_view(tmp_path), tmp_path)(_observation(**START))


def test_the_exporters_learned_source_writes_rows_and_confidence(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "add_future_keyframe", Path(__file__).parents[2] / "scripts/add_future_keyframe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from oct_vla.data.episode import Episode

    torch.manual_seed(0)
    demo_episode = _demo()
    episode = Episode(seed=9, instruction="restock", samples=demo_episode.samples,
                      success=True, metadata={"rule": "leftmost"})
    for kind, shape in (("keyframe", (3, 24)), ("latent", (N_MAX + 1, 16))):
        source = object.__new__(module._LearnedSource)
        source.A, source.k, source.hits = A, 2, []
        source.args = SimpleNamespace(seed=0, kind=kind, device="cpu")
        source.pool_index = [0, 1, 2]
        source.demos = {i: A.support_demo(demo_episode, SUPPORTS) for i in range(3)}
        source.model, source.lang = KeyframePFN(width=16, heads=2, layers=1), None
        frames = np.arange(len(episode.samples))
        values, mask, conf, note = source(1, episode, frames)
        assert values.shape == (len(frames), *shape) and mask.shape == (len(frames), shape[0])
        assert conf.shape == (len(frames),) and ((conf > 0) & (conf <= 1)).all()
        assert "supports [0, 2]" in note  # never the episode itself (index 1)
