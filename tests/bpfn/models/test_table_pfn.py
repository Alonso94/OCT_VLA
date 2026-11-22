"""TablePFN: cells of pose + header, a time column, row CLS cells, a pooled task; every way
of reading the task and both outputs."""

import copy
import itertools

import numpy as np
import pytest
import torch

from oct_vla.bpfn import schema
from oct_vla.bpfn.benchmark_tools import metrics
from oct_vla.bpfn.data import batches, splits
from oct_vla.bpfn.models import build
from oct_vla.bpfn.models.pose_window_pfn import to_torch
from oct_vla.bpfn.training_tools.losses.window import WindowLoss
from oct_vla.bpfn.training_tools.trainer import conflict_items

VARIANTS = list(itertools.product(("cell", "film", "cross"), ("head", "rows")))


@pytest.fixture(scope="module")
def items():
    return conflict_items(3, seed=0, s_rows=32)


@pytest.fixture(scope="module")
def history_items():
    """Ordinary items whose queries have several observed rows (conflict pairs often sit at
    t = 0, where a query has a single row)."""
    cfg = batches.ItemConfig(programs=splits.programs_for("full"), conflict_fraction=0.0, s_rows=32)
    rng = np.random.default_rng(1)
    chosen = []
    while len(chosen) < 6:
        for item in batches.sample_items(rng, cfg, 8):
            if len(item["query"].time) >= 8 and len(chosen) < 6:
                chosen.append(item)
    return chosen


def _model(task_read, output, randomise=True, **kw):
    torch.manual_seed(0)
    model = build(dict(arch="table", width=32, heads=2, support_layers=2, query_layers=2,
                       task_read=task_read, output=output, **kw))
    if randomise:
        # The zero-initialised head would make every comparison pass trivially.
        torch.nn.init.normal_(model.head[1].weight, std=0.1)
        if task_read == "film":
            for layer in model.film:
                torch.nn.init.normal_(layer.weight, std=0.1)
    return model.eval()


def _batch(items):
    return to_torch(batches.pack(items, s_rows=32))


@pytest.mark.parametrize("task_read,output", VARIANTS)
def test_outputs_are_unit_quaternions_and_a_zero_head_holds(items, task_read, output):
    model = _model(task_read, output, randomise=False)
    with torch.no_grad():
        window = model(_batch(items))["window"]
    assert window.shape == (len(items), batches.HORIZON, 2, 7)
    torch.testing.assert_close(window[..., 3:].norm(dim=-1), torch.ones(window.shape[:-1]))
    held = np.stack([metrics.hold(it["query"], batches.HORIZON) for it in items])
    np.testing.assert_allclose(window.numpy(), held, atol=1e-5)


@pytest.mark.parametrize("task_read,output", VARIANTS)
def test_cached_support_equals_uncached(items, task_read, output):
    model, b = _model(task_read, output), _batch(items)
    with torch.no_grad():
        direct = model(b)["window"]
        cached = model.predict(model.encode_support(b), b)["window"]
    torch.testing.assert_close(direct, cached)


@pytest.mark.parametrize("task_read,output", VARIANTS)
def test_predictions_do_not_depend_on_column_order(items, task_read, output):
    model, rng = _model(task_read, output), np.random.default_rng(0)
    shuffled = []
    for item in items:
        item = dict(item)
        item["support"], _ = schema.shuffle_columns(item["support"], rng)
        item["query"], _ = schema.shuffle_columns(item["query"], rng)
        shuffled.append(item)
    with torch.no_grad():
        a = model(_batch(items))["window"]
        b = model(_batch(shuffled))["window"]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("task_read,output", VARIANTS)
def test_the_support_matters_and_scene_only_ignores_it(items, task_read, output):
    model = _model(task_read, output)
    scene_only = _model(task_read, output, use_support=False)
    b = _batch(items)
    swapped = dict(b)
    swapped.update({k: v.flip(0) for k, v in b.items() if k.startswith("s_")})
    with torch.no_grad():
        assert not torch.allclose(model(b)["window"], model(swapped)["window"])
        torch.testing.assert_close(scene_only(b)["window"], scene_only(swapped)["window"])


@pytest.mark.parametrize("task_read,output", VARIANTS)
def test_only_pose_header_and_time_since_the_first_row_enter(history_items, task_read, output):
    model, b = _model(task_read, output), _batch(history_items)
    assert (b["q_row_valid"].sum(1) >= 8).all()
    changed = dict(b)
    for prefix in ("s_", "q_"):
        changed[prefix + "observed"] = torch.zeros_like(b[prefix + "observed"])
        changed[prefix + "age"] = b[prefix + "age"] + 3.0
        changed[prefix + "time"] = b[prefix + "time"] + 7.5       # another time origin
    with torch.no_grad():
        torch.testing.assert_close(model(b)["window"], model(changed)["window"])
        # ... while the spacing of the rows does matter
        stretched = dict(b, q_time=b["q_time"] * 2)
        assert not torch.allclose(model(b)["window"], model(stretched)["window"])


@pytest.mark.parametrize("task_read,output", VARIANTS)
def test_an_empty_support_is_defined(items, task_read, output):
    model = _model(task_read, output)
    empty = [dict(it, support=schema.empty_support(it["support"])) for it in items]
    with torch.no_grad():
        window = model(_batch(empty))["window"]
    assert torch.isfinite(window).all()


@pytest.mark.parametrize("task_read,output", VARIANTS)
def test_every_parameter_learns(items, task_read, output):
    torch.manual_seed(0)
    model = build(dict(arch="table", width=32, heads=2, support_layers=2, query_layers=2,
                       task_read=task_read, output=output)).train()
    b = _batch(items)
    loss_fn = WindowLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(3):      # zero-initialised head and FiLM block gradients on the first steps
        loss, _ = loss_fn(model(b)["window"], b["target"], b["target_valid"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    dead = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not dead, dead


@pytest.mark.parametrize("task_read,output", VARIANTS)
def test_grouped_batches_equal_one_support_per_item(items, task_read, output):
    model = _model(task_read, output)
    groups = []
    for start in range(0, len(items), 2):
        groups.append({"support": items[start]["support"],
                       "items": [{k: it[k] for k in ("query", "target", "target_valid", "meta")}
                                 for it in items[start:start + 2]]})
    grouped = to_torch(batches.pack_groups(groups, s_rows=32))
    flat = to_torch(batches.pack(batches.ungroup(groups), s_rows=32))
    with torch.no_grad():
        torch.testing.assert_close(model(grouped)["window"], model(flat)["window"], atol=1e-5, rtol=1e-5)
