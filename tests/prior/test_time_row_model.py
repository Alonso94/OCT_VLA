"""TimeRowPFN (B1), the A1 control and the v1 bridge (prior/time_row_model.py, _v1_bridge.py)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oct_vla.prior import model as M  # noqa: E402
from oct_vla.prior.batches import ItemConfig  # noqa: E402
from oct_vla.prior.generator import sample_episode  # noqa: E402
from oct_vla.prior.programs import Program  # noqa: E402
from oct_vla.prior.programs_v2 import TRAIN_PROGRAMS_V2  # noqa: E402
from oct_vla.prior.time_row_batches import (  # noqa: E402
    TimeRowConfig,
    pack_time_row_item,
    sample_time_row_batch,
)
from oct_vla.prior.time_row_model import TimeRowPFN, paired_loss, to_torch_time_row  # noqa: E402
from oct_vla.prior.time_row_v1_bridge import KeyframePairedPFN, to_keyframe_batch  # noqa: E402

V1 = TimeRowConfig(item=ItemConfig(language=(1, 0, 0)))
V2 = TimeRowConfig(item=ItemConfig(programs=TRAIN_PROGRAMS_V2, language=(1, 0, 0)))


def _batch(cfg, size=6, seed=0):
    inputs, targets, metas = sample_time_row_batch(np.random.default_rng(seed), cfg, size)
    return inputs, targets, metas


def _model(grammar="v1", **kw):
    torch.manual_seed(0)
    return TimeRowPFN(width=32, heads=4, layers=2, grammar=grammar, **kw)


@pytest.mark.parametrize("grammar,cfg", [("v1", V1), ("v2", V2)])
def test_every_parameter_learns_on_both_grammars(grammar, cfg):
    inputs, targets, _ = _batch(cfg)
    model = _model(grammar).train()
    b = to_torch_time_row(inputs, targets, None, "cpu")
    total, terms = paired_loss(model(b), b, grammar == "v2")
    total.backward()
    assert torch.isfinite(total) and ("rot" in terms) == (grammar == "v2")
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_query_targets_are_not_needed_and_support_labels_are_read():
    inputs, targets, _ = _batch(V1)
    model = _model().eval()
    with torch.no_grad():
        base = model(to_torch_time_row(inputs, None, None, "cpu"), teacher=False)
        relabel = dict(inputs)
        relabel["s_g_event"] = np.where(inputs["s_row_valid"], 1, inputs["s_g_event"])
        moved = model(to_torch_time_row(relabel, None, None, "cpu"), teacher=False)
    assert not torch.allclose(base["group_event"], moved["group_event"])


def test_pointers_follow_a_permutation_of_the_query_columns():
    inputs, _, _ = _batch(V1, size=4, seed=3)
    model = _model().eval()
    n_valid = (inputs["q_types"] >= 0).sum(1)
    perm = np.arange(inputs["q_types"].shape[1])
    k = int(n_valid.min())
    perm[:k] = np.random.default_rng(0).permutation(k)
    moved = dict(inputs)
    moved["q_feats"], moved["q_types"] = inputs["q_feats"][:, :, perm], inputs["q_types"][:, perm]
    with torch.no_grad():
        a = model(to_torch_time_row(inputs, None, None, "cpu"), teacher=False)
        b = model(to_torch_time_row(moved, None, None, "cpu"), teacher=False)
    torch.testing.assert_close(b["group_pointer"], a["group_pointer"][:, perm], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(b["pos"], a["pos"], rtol=1e-4, atol=1e-4)


def test_a_tiny_batch_can_be_overfit():
    inputs, targets, _ = _batch(V1, size=8, seed=4)
    b = to_torch_time_row(inputs, targets, None, "cpu")
    model = _model().train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        total, _ = paired_loss(model(b), b, False)
        first = float(total) if first is None else first
        opt.zero_grad()
        total.backward()
        opt.step()
    assert float(total) < 0.5 * first


def test_reload_reproduces_predictions(tmp_path):
    inputs, _, _ = _batch(V2, size=3)
    model = _model("v2", query_initial=False).eval()
    torch.save({"config": model.config, "state": model.state_dict()}, tmp_path / "m.pt")
    blob = torch.load(tmp_path / "m.pt")
    again = TimeRowPFN(**{k: v for k, v in blob["config"].items() if k != "schema"})
    again.load_state_dict(blob["state"])
    again.eval()
    b = to_torch_time_row(inputs, None, None, "cpu")
    with torch.no_grad():
        for k, v in model(b, teacher=False).items():
            torch.testing.assert_close(again(b, teacher=False)[k], v)


def test_the_bridge_rebuilds_v1_keyframe_tables():
    rng = np.random.default_rng(6)
    program = Program("left_to_right", "higher", "row_left", "handover")
    demos = [sample_episode(rng, program) for _ in range(2)]
    q = sample_episode(rng, program)
    inputs, targets, _ = pack_time_row_item(rng, program, demos, q, 1, 0.0, 0, V1)
    kb = to_keyframe_batch({k: np.asarray(v)[None] for k, v in inputs.items()},
                           {k: np.asarray(v)[None] for k, v in targets.items()})
    for d, ep in enumerate(demos):
        n_kf = len(ep.keyframes)
        assert kb["ctx_rows"][0, d].sum() == n_kf
        assert kb["ctx_event"][0, d, :n_kf].tolist() == [kf.event for kf in ep.keyframes]
        assert kb["ctx_gripper"][0, d, 1:n_kf].tolist() == [kf.gripper for kf in ep.keyframes[1:]]
    a1 = KeyframePairedPFN(width=32, heads=4, layers=2, use_edges=False).eval()
    with torch.no_grad():
        out = a1.forward_paired(M.to_torch(kb, None, "cpu"), teacher=False)
    assert out["pos"].shape == (1, 2, 3) and out["group_pointer"].shape == (1, 9)


# --- the object-relative head (anchor="object") --------------------------------------

@pytest.mark.parametrize("grammar,cfg", [("v1", V1), ("v2", V2)])
def test_object_anchor_learns_on_both_grammars(grammar, cfg):
    inputs, targets, _ = _batch(cfg)
    model = _model(grammar, anchor="object").train()
    b = to_torch_time_row(inputs, targets, None, "cpu")
    total, _ = paired_loss(model(b), b, grammar == "v2")
    total.backward()
    assert torch.isfinite(total)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_object_anchor_starts_each_arm_where_its_event_says():
    """Same weights, two anchors: the positions differ by exactly anchor - current gripper."""
    from oct_vla.prior import time_row_schema as S
    from oct_vla.prior.time_row_model import _gripper_columns, anchor_kinds

    inputs, targets, _ = _batch(V1, size=8, seed=4)
    grip, obj = _model(), _model(anchor="object")
    obj.load_state_dict(grip.state_dict())
    b = to_torch_time_row(inputs, targets, None, "cpu")
    with torch.no_grad():
        a, o = grip.eval()(b, teacher=True), obj.eval()(b, teacher=True)
    cur = M.normalise(b["q_feats"][:, -1][..., :9])[..., :3]
    cols = _gripper_columns(b["q_types"].long())
    kinds = anchor_kinds(S.ARM_EVENTS)[b["t_y_event"].long()]
    for i in range(8):
        for arm in range(2):
            g = cur[i, cols[i, arm]]
            want = {0: g, 1: cur[i, b["t_y_object"][i, arm].clamp_min(0)],
                    2: o["goal"][i, :3]}[int(kinds[i, arm])]
            torch.testing.assert_close(o["pos"][i, arm] - a["pos"][i, arm], want - g)
    assert (kinds == 1).any() and (kinds == 0).any()


def test_a1_takes_the_anchor_and_old_configs_default_to_the_gripper():
    model = KeyframePairedPFN(width=32, heads=4, layers=2, use_edges=False, anchor="object")
    assert model.config["anchor"] == "object" and model.paired.anchor == "object"
    assert _model().config["anchor"] == "gripper"
    with pytest.raises(ValueError):
        _model(anchor="elsewhere")
