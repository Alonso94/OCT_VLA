"""PFNTrainer: seeding, the learning-rate schedule and the compiled forward."""

import pytest
import torch

from oct_vla.bpfn.training_tools.trainer import PFNTrainer, TrainConfig


def _initial_state(tmp_path, name, seed):
    cfg = TrainConfig(out=tmp_path / name, seed=seed, width=16, layers=1, workers=0, s_rows=32,
                      device="cpu")
    return PFNTrainer(cfg).model.state_dict()


def test_the_seed_fixes_the_initial_weights(tmp_path):
    torch.manual_seed(1)
    a = _initial_state(tmp_path, "a", seed=0)
    torch.manual_seed(2)                 # whatever ran before must not matter
    b = _initial_state(tmp_path, "b", seed=0)
    c = _initial_state(tmp_path, "c", seed=1)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not all(torch.equal(a[k], c[k]) for k in a)


def test_the_lr_schedule_decays_to_its_floor(tmp_path):
    cfg = TrainConfig(out=tmp_path / "run", steps=1000, warmup=10, lr_floor=0.1, width=16,
                      layers=1, workers=0, s_rows=32, device="cpu")
    trainer = PFNTrainer(cfg)
    assert trainer._lr_factor(0) == 0.1                   # warmup
    assert abs(trainer._lr_factor(10) - 1.0) < 1e-3
    assert abs(trainer._lr_factor(1000) - 0.1) < 1e-9     # the floor, not zero


def test_the_compiled_forward_matches_the_eager_one():
    import shutil

    if shutil.which("g++") is None:
        pytest.skip("torch.compile on CPU needs a C++ compiler")
    from oct_vla.bpfn.data import batches
    from oct_vla.bpfn.training_tools.trainer import conflict_items
    from oct_vla.bpfn.models.pose_window_pfn import PoseWindowPFN, to_torch

    torch.manual_seed(0)
    model = PoseWindowPFN(width=16, heads=2, layers=1).train()
    torch.nn.init.normal_(model.head[1].weight, std=0.5)
    b = to_torch(batches.pack(conflict_items(2, seed=0, s_rows=32), s_rows=32))
    compiled = torch.compile(model)
    out_c = compiled(b)["window"]
    out_e = model(b)["window"]
    torch.testing.assert_close(out_c, out_e, atol=1e-4, rtol=1e-4)
    out_c.sum().backward()
    grads_c = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    out_e.sum().backward()
    for gc, ge in zip(grads_c, [p.grad for p in model.parameters()]):
        torch.testing.assert_close(gc, ge, atol=1e-3, rtol=1e-3)
