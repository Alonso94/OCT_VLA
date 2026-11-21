"""PFNTrainer: a seed fixes the initial weights."""

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
