"""KeyframePFN wiring: context, language, equivariance, gradients; the baselines."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oct_vla.prior import baselines as BL  # noqa: E402
from oct_vla.prior import batches as B  # noqa: E402
from oct_vla.prior import interventions as I  # noqa: E402
from oct_vla.prior import metrics as MT  # noqa: E402
from oct_vla.prior import model as M  # noqa: E402
from oct_vla.prior import programs as P  # noqa: E402

LANG = 8


@pytest.fixture(scope="module")
def batch():
    return B.sample_batch(np.random.default_rng(11), B.ItemConfig(), 24)


@pytest.fixture(scope="module")
def table():
    g = torch.Generator().manual_seed(0)
    tokens = torch.randn(len(P.ALL_INSTRUCTIONS), 6, LANG, generator=g)
    mask = torch.ones(len(P.ALL_INSTRUCTIONS), 6, dtype=torch.bool)
    mask[:, 4:] = False
    return tokens, mask


def _model(**kw):
    torch.manual_seed(0)
    return M.KeyframePFN(width=32, heads=4, layers=2, lang_dim=LANG, **kw).eval()


def _pointer(model, b, table):
    with torch.no_grad():
        return model(M.to_torch(b, table, "cpu"), teacher=False)["pointer"]


def test_support_reaches_the_prediction_only_with_context(batch, table):
    b = I.language(batch, "none", np.random.default_rng(0))
    shuffled, _ = I.support(b, "shuffled", np.random.default_rng(1))
    with_ctx = _model()
    assert not torch.allclose(_pointer(with_ctx, b, table), _pointer(with_ctx, shuffled, table))
    blind = _model(use_context=False)
    torch.testing.assert_close(_pointer(blind, b, table), _pointer(blind, shuffled, table))


def test_instruction_is_read_only_when_present(batch, table):
    rng = np.random.default_rng(0)
    model = _model()
    explicit = I.language(batch, "explicit", rng)
    ambiguous = I.language(batch, "ambiguous", rng)
    assert not torch.allclose(_pointer(model, explicit, table), _pointer(model, ambiguous, table))
    none = I.language(batch, "none", rng)
    other_table = (table[0] + 5.0, table[1])
    torch.testing.assert_close(_pointer(model, none, table), _pointer(model, none, other_table))


def test_pointer_is_equivariant_to_query_column_order(batch, table):
    model = _model()
    perm = np.random.default_rng(3).permutation(batch["q_types"].shape[1])
    moved = dict(
        batch,
        q_feats=batch["q_feats"][:, :, perm],
        q_types=batch["q_types"][:, perm],
        t_obj=np.argsort(perm)[batch["t_obj"]],
    )
    a, b = _pointer(model, batch, table), _pointer(model, moved, table)
    torch.testing.assert_close(a[:, perm], b, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("head", M.HEADS)
def test_every_parameter_gets_a_gradient(batch, table, head):
    model = _model(head=head).train()
    out = model(M.to_torch(batch, table, "cpu"))
    total, terms = M.loss(model, out, M.to_torch(batch, table, "cpu"))
    assert np.isfinite(total.item())
    total.backward()
    missing = [
        n for n, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0
    ]
    assert not missing, missing


def test_to_torch_refuses_an_instruction_index_without_a_kind(batch, table):
    bad = dict(batch, lang_kind=np.zeros_like(batch["lang_kind"]))
    if (batch["lang_index"] >= 0).any():
        with pytest.raises(ValueError):
            M.to_torch(bad, table, "cpu")


def test_metric_units_round_trip():
    pose = torch.tensor([[0.1, -0.2, 0.9, 0.0, 1.0]])
    back = M.to_metric({"obj": M.target_pose(pose)})["obj"]
    np.testing.assert_allclose(back, pose.numpy(), atol=1e-6)


def test_baselines_run_and_retrieval_beats_copy_on_the_pointer():
    cfg = B.ItemConfig(conflict_fraction=0.0, language=(1.0, 0.0, 0.0), k_range=(4, 4))
    b = B.sample_batch(np.random.default_rng(5), cfg, 120)
    copy, ret = MT.summarise(BL.copy_current(b), b), MT.summarise(BL.retrieval(b), b)
    assert ret["pointer"]["acc"] > copy["pointer"]["acc"] + 0.1
    # A grasp leaves the object in place, so copy-current is exact there (given the pointer).
    exact = (b["t_event"] == 1) & (b["alpha"] == 0)
    assert exact.any()
    np.testing.assert_allclose(
        MT.correctness(BL.copy_current(b), b)["obj_err"][exact], 0.0, atol=1e-6
    )
