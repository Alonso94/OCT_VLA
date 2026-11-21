"""Stage-3 pose tokens: support choice and per-frame entries."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oct_vla.bpfn.benchmark_tools.evaluator import predict  # noqa: E402
from oct_vla.bpfn.data import batches, splits  # noqa: E402
from oct_vla.bpfn.data.syn_data import generator  # noqa: E402
from oct_vla.bpfn.downstream.views import PoseTokenBuilder, choose_supports  # noqa: E402
from oct_vla.bpfn.models.pose_window_pfn import PoseWindowPFN  # noqa: E402
from oct_vla.bpfn.schema import query_table, support_table  # noqa: E402
from oct_vla.bpfn.downstream import tokens  # noqa: E402


@pytest.fixture(scope="module")
def episodes():
    program = next(p for p in splits.programs_for("full") if p.manipulation == "carry_left"
                   and p.eligibility == "all")
    rng = np.random.default_rng(0)
    return [generator.sample_episode(program, 2, 0, rng) for _ in range(2)]


def test_supports_exclude_the_episode_and_are_fixed_per_episode(tmp_path):
    pool = [tmp_path / f"run_{i}" for i in range(6)]
    for p in pool:
        p.mkdir()
    own = pool[2]
    a = choose_supports(pool, own, bank=4, seed=0, episode_index=7)
    assert own not in a and len(set(a)) == 4
    assert a == choose_supports(pool, own, bank=4, seed=0, episode_index=7)
    with pytest.raises(SystemExit, match="need 6"):
        choose_supports(pool, own, bank=6, seed=0, episode_index=7)


def test_pose_window_entries_are_the_pfn_windows(episodes):
    query, support = episodes
    torch.manual_seed(0)
    model = PoseWindowPFN(width=16, heads=2, layers=1).eval()
    torch.nn.init.normal_(model.head[1].weight, std=0.5)
    entries = PoseTokenBuilder("pose_window", model).frame_entries(query, [support])
    assert len(entries) == query.T and len(entries[0]) == 1
    s_tab = support_table(support, batches.support_rows(support, batches.S_ROWS))
    t = query.T // 2
    item = {"support": s_tab, "query": query_table(query, t, batches.HISTORY),
            "target": np.zeros((model.horizon, 2, 7), np.float32),
            "target_valid": np.zeros(model.horizon, bool), "meta": {}}
    window = predict(model, [item], batches.S_ROWS)[0]
    np.testing.assert_allclose(entries[t][0], tokens.window_tokens(window), atol=1e-6)
    assert entries[t][0].shape[0] == PoseTokenBuilder("pose_window", model).rows


def test_support_table_entries_do_not_depend_on_the_frame(episodes):
    query, support = episodes
    entries = PoseTokenBuilder("support_table").frame_entries(query, [support, query])
    assert all(np.array_equal(entries[0][0], entries[t][0]) for t in range(query.T))
    with pytest.raises(ValueError, match="need the PFN"):
        PoseTokenBuilder("pose_window")
