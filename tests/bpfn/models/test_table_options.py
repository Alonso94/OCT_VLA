"""TablePFN options: relative frame, several hypotheses, patched demonstrations, bf16 trunk,
rule-factor heads; and their losses."""

import json

import numpy as np
import pytest
import torch

from oct_vla.bpfn.data import batches, splits
from oct_vla.bpfn.models import build
from oct_vla.bpfn.models.pose_window_pfn import to_torch
from oct_vla.bpfn.training_tools.losses.window import HypothesisLoss, WindowLoss, factor_loss
from oct_vla.bpfn.training_tools.trainer import PFNTrainer, TrainConfig, conflict_items


@pytest.fixture(scope="module")
def items():
    return conflict_items(3, seed=0, s_rows=32)


def _model(**kw):
    torch.manual_seed(0)
    config = dict(arch="table", width=32, heads=2, support_layers=2, query_layers=2,
                  task_read="film", output="rows")
    config.update(kw)
    model = build(config)
    torch.nn.init.normal_(model.head[1].weight, std=0.1)
    for layer in model.film:
        torch.nn.init.normal_(layer.weight, std=0.1)
    return model.eval()


def _batch(items):
    return to_torch(batches.pack(items, s_rows=32))


def _shift(b, offset):
    """Every position of both tables moved by ``offset`` (a layout shifted as a whole)."""
    moved = dict(b)
    for key in ("s_pose", "s_goal", "q_pose"):
        value = b[key].clone()
        value[..., :3] += torch.tensor(offset)
        moved[key] = value
    return moved


def test_the_relative_frame_makes_a_shifted_layout_look_the_same(items):
    offset = (0.07, -0.05, 0.02)
    b = _batch(items)
    relative = _model(frame="relative")
    with torch.no_grad():
        a = relative(b)["window"]
        shifted = relative(_shift(b, offset))["window"]
    torch.testing.assert_close(shifted[..., :3], a[..., :3] + torch.tensor(offset), atol=1e-5, rtol=0)
    torch.testing.assert_close(shifted[..., 3:], a[..., 3:], atol=1e-5, rtol=0)
    workcell = _model(frame="workcell")                 # the control: absolute positions matter
    with torch.no_grad():
        moved = workcell(_shift(b, offset))["window"][..., :3] - torch.tensor(offset)
        assert not torch.allclose(moved, workcell(b)["window"][..., :3], atol=1e-4)


def test_hypotheses_return_the_most_confident_future(items):
    model = _model(hypotheses=3)
    with torch.no_grad():
        out = model(_batch(items))
    assert out["windows"].shape == (len(items), 3, batches.HORIZON, 2, 7)
    best = out["confidence"].argmax(1)
    torch.testing.assert_close(out["window"], out["windows"][torch.arange(len(items)), best])
    assert not torch.allclose(out["windows"][:, 0], out["windows"][:, 1])   # they start apart
    with pytest.raises(ValueError, match="rows output"):
        build(dict(arch="table", output="head", hypotheses=2))


def test_the_hypothesis_loss_trains_every_future_and_the_confidence(items):
    model = _model(hypotheses=3).train()
    b = _batch(items)
    out = model(b)
    loss, stats = HypothesisLoss(WindowLoss())(out["windows"], out["confidence"], b["target"],
                                               b["target_valid"])
    loss.backward()
    assert float(stats["oracle_mm"]) <= float(stats["trans_mm"]) + 1e-6
    for k in range(3):                       # the spread term reaches every hypothesis
        rows = slice(9 * k, 9 * k + 9)
        assert model.head[1].weight.grad[rows].abs().sum() > 0
    assert model.confidence[1].weight.grad.abs().sum() > 0


def test_patched_demonstrations_keep_caching_and_grouping(items):
    model = _model(support_patch=4)
    b = _batch(items)
    with torch.no_grad():
        direct = model(b)["window"]
        cached = model.predict(model.encode_support(b), b)["window"]
    torch.testing.assert_close(direct, cached)
    groups = [{"support": items[i]["support"],
               "items": [{k: it[k] for k in ("query", "target", "target_valid", "meta")}
                         for it in items[i:i + 2]]} for i in range(0, len(items), 2)]
    with torch.no_grad():
        grouped = model(to_torch(batches.pack_groups(groups, s_rows=32)))["window"]
        flat = model(to_torch(batches.pack(batches.ungroup(groups), s_rows=32)))["window"]
    torch.testing.assert_close(grouped, flat, atol=1e-5, rtol=1e-5)
    odd = dict(b, **{k: v[:, :30] for k, v in b.items() if k in ("s_pose", "s_goal", "s_goal_valid",
                                                                 "s_time", "s_row_valid",
                                                                 "s_observed", "s_age")})
    with pytest.raises(ValueError, match="patches"):
        model(odd)


def test_the_bf16_trunk_is_float32_off_cuda(items):
    b = _batch(items)
    with torch.no_grad():
        torch.testing.assert_close(_model(bf16_trunk=True)(b)["window"], _model()(b)["window"])


def test_rule_factors_are_packed_and_learned_without_unknown_labels(items):
    b = _batch(items)
    factors = b["s_factors"]
    assert factors.shape == (len(items), len(batches.FACTORS)) and (factors >= 0).all()
    program = items[0]["meta"]["program"].split("|")
    assert batches.FACTORS[0][1][int(factors[0, 0])] == program[0]
    no_meta = to_torch(batches.pack([dict(items[0], meta={})], s_rows=32))
    assert (no_meta["s_factors"] == -1).all()
    model = _model(aux_factors=True).train()
    out = model(b)
    assert [lg.shape for lg in out["factor_logits"]] == [(len(items), len(v)) for _, v in batches.FACTORS]
    loss = factor_loss(out["factor_logits"], factors.long())
    loss.backward()
    assert torch.isfinite(loss) and model.factor_heads[0].weight.grad.abs().sum() > 0
    unknown = torch.full_like(factors, -1).long()
    assert float(factor_loss(out["factor_logits"], unknown).detach()) == 0.0


def test_training_with_every_option_and_ema(tmp_path):
    cfg = TrainConfig(out=tmp_path / "run", steps=3, batch=4, width=16, layers=1, workers=0,
                      s_rows=32, device="cpu", arch="table", task_read="film", output="rows",
                      frame="relative", hypotheses=2, support_patch=4, aux_factors=0.1,
                      ema=0.9, log_every=1)
    trainer = PFNTrainer(cfg)
    trainer.train()
    saved = torch.load(tmp_path / "run" / "model.pt", weights_only=False)["state"]
    averaged = trainer.ema_model.module.state_dict()
    raw = trainer.model.state_dict()
    assert all(torch.equal(saved[k], averaged[k]) for k in saved)
    assert any(not torch.equal(averaged[k], raw[k]) for k in raw)         # the average differs
    log = [json.loads(line) for line in (tmp_path / "run" / "history.jsonl").read_text().splitlines()]
    assert "oracle_mm" in log[-1] and "factor_loss" in log[-1]


def test_the_model_never_reads_the_rule_labels(items):
    model = _model(aux_factors=True)
    b = _batch(items)
    scrambled = dict(b, s_factors=torch.randint(0, 2, b["s_factors"].shape))
    with torch.no_grad():
        a, c = model(b), model(scrambled)
    torch.testing.assert_close(a["window"], c["window"])
    for x, y in zip(a["factor_logits"], c["factor_logits"]):
        torch.testing.assert_close(x, y)
