"""PoseWindowPFN: caching, invariances, gradients, the loss and the decision scorer."""

import collections
import copy

import numpy as np
import pytest
import torch

from oct_vla.bpfn import schema
from oct_vla.bpfn.benchmark_tools import metrics
from oct_vla.bpfn.data import batches, splits
from oct_vla.bpfn.data.syn_data import generator
from oct_vla.bpfn.models.pose_window_pfn import PoseWindowPFN, to_torch
from oct_vla.bpfn.training_tools.losses.window import WindowLoss
from oct_vla.bpfn.training_tools.trainer import conflict_items


@pytest.fixture(scope="module")
def items():
    return conflict_items(3, seed=0, s_rows=32)


def _model(**kw):
    torch.manual_seed(0)
    m = PoseWindowPFN(width=32, heads=2, layers=2, **kw)
    # Non-trivial outputs: the zero-initialised head would make every test pass trivially.
    torch.nn.init.normal_(m.head[1].weight, std=0.1)
    return m.eval()


def test_cached_support_equals_uncached(items):
    m = _model()
    b = to_torch(batches.pack(items, s_rows=32))
    with torch.no_grad():
        a = m(b)["window"]
        c = m.predict(m.encode_support(b), b)["window"]
    torch.testing.assert_close(a, c)


def test_predictions_are_invariant_to_column_order(items):
    m = _model()
    rng = np.random.default_rng(1)
    shuffled = []
    for it in items:
        it2 = dict(it)
        it2["support"], _ = schema.shuffle_columns(it["support"], rng)
        it2["query"], _ = schema.shuffle_columns(it["query"], rng)
        shuffled.append(it2)
    with torch.no_grad():
        a = m(to_torch(batches.pack(items, s_rows=32)))["window"]
        b = m(to_torch(batches.pack(shuffled, s_rows=32)))["window"]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_support_matters_and_scene_only_ignores_it(items):
    m, so = _model(), _model(use_support=False)
    b = to_torch(batches.pack(items, s_rows=32))
    other = dict(b)
    other.update({k: v.flip(0) for k, v in b.items() if k.startswith("s_")})   # swap supports
    with torch.no_grad():
        assert not torch.allclose(m(b)["window"], m(other)["window"])
        torch.testing.assert_close(so(b)["window"], so(other)["window"])


def test_every_parameter_learns(items):
    m = PoseWindowPFN(width=32, heads=2, layers=2)
    b = to_torch(batches.pack(items, s_rows=32))
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for _ in range(2):     # the zero head blocks upstream gradient on the first step only
        loss, _ = WindowLoss()(m(b)["window"], b["target"], b["target_valid"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    dead = [n for n, p in m.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not dead, dead


def test_outputs_are_unit_quaternions_anchored_at_the_current_wrist(items):
    m = PoseWindowPFN(width=32, heads=2, layers=2).eval()   # zero head: hold
    b = to_torch(batches.pack(items, s_rows=32))
    with torch.no_grad():
        w = m(b)["window"]
    torch.testing.assert_close(w[..., 3:].norm(dim=-1), torch.ones(w.shape[:-1]))
    held = np.stack([metrics.hold(it["query"], batches.HORIZON) for it in items])
    np.testing.assert_allclose(w.numpy(), held, atol=1e-5)


def test_loss_is_sign_invariant_and_masks_invalid_horizons():
    t = torch.zeros(1, 2, 2, 7)
    t[..., 6] = 1
    flipped = t.clone()
    flipped[..., 3:] *= -1
    l, _ = WindowLoss()(flipped, t, torch.ones(1, 2, dtype=torch.bool))
    assert float(l) == pytest.approx(0.0, abs=1e-7)
    far = t.clone()
    far[:, 1, :, 0] = 5.0
    l, _ = WindowLoss()(far, t, torch.tensor([[True, False]]))
    assert float(l) == pytest.approx(0.0, abs=1e-7)


def test_decision_rule_recovers_the_expert_and_rejects_hold(items):
    pairs = collections.defaultdict(list)
    for it in items:
        pairs[it["meta"]["pair"]].append((metrics.decision(it["target"], it["query"]),
                                          it["meta"]["first_pick"]))
    assert metrics.paired_both_correct(pairs)["both_correct"] == 1.0
    held = {k: [(metrics.decision(metrics.hold(it["query"], 8), it["query"]), it["meta"]["first_pick"])
                for it in items if it["meta"]["pair"] == k] for k in pairs}
    r = metrics.paired_both_correct(held)
    assert r["both_correct"] == 0.0 and r["unassignable"] == 1.0


def test_an_empty_support_is_defined_not_nan(items):
    m = _model()
    empty = [dict(it, support=schema.empty_support(it["support"])) for it in items]
    b = to_torch(batches.pack(empty, s_rows=32))
    assert not b["s_row_valid"].any()         # no row of the demonstration is valid
    with torch.no_grad():
        w = m(b)["window"]
    assert torch.isfinite(w).all()
    # ...and it is the same prediction whatever support was emptied.
    other = [dict(it, support=schema.empty_support(items[(k + 1) % len(items)]["support"]))
             for k, it in enumerate(items)]
    with torch.no_grad():
        w2 = m(to_torch(batches.pack(other, s_rows=32)))["window"]
    torch.testing.assert_close(w, w2)


def test_errors_ignore_invalid_horizons_past_the_episode_end():
    """Regression: every training job crashed at its first dev evaluation, because the
    angle was computed on zero-filled targets past an episode's end."""
    ep = __import__("oct_vla.bpfn.data.syn_data.branches", fromlist=["handover"]).handover()
    w, v = schema.target_window(ep, ep.T - 3, horizon=16)
    assert v.sum() == 2 and np.allclose(np.linalg.norm(w[..., 3:], axis=-1), 1.0)
    zeroed = w.copy()
    zeroed[~v] = 0.0                                   # what an old pack would hold
    r = metrics.errors(w[None], zeroed[None], v[None])
    assert r["trans_mm"] == pytest.approx(0.0) and r["rot_deg"] == pytest.approx(0.0, abs=1e-3)
    # Moving / idle is decided on the last valid horizon, not on a padded one.
    assert r["idle_drift_mm"] == pytest.approx(0.0)
