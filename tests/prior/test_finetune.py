"""Stage B fine-tuning items (prior/finetune.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oct_vla.data.episode import Episode  # noqa: E402
from oct_vla.prior import finetune as FT  # noqa: E402
from oct_vla.prior import model as M  # noqa: E402
from oct_vla.prior import shelf_adapter as A  # noqa: E402
from oct_vla.prior.batches import ItemConfig, sample_batch  # noqa: E402
from oct_vla.prior.schema import GRASP, PLACE  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_shelf_adapter import SUPPORTS, _demo  # noqa: E402


def _episode(seed):
    d = _demo()
    return Episode(seed=seed, instruction="restock", samples=d.samples, success=True,
                   metadata={})


def test_the_protocol_split_and_its_reserved_blocks():
    assert FT.split_of(100) == FT.split_of(399) == "train"
    assert FT.split_of(200) == "val"
    for reserved in (250, 300, 460, 700):
        with pytest.raises(ValueError, match="reserved"):
            FT.split_of(reserved)


def test_items_target_the_next_recorded_event():
    items = FT.episode_items(_episode(101), SUPPORTS, "leftmost", 0)
    # Frames 0..3 precede an event; frame 4 is the last place, so it has none.
    assert len(items) == 4
    first = items[0].target
    assert (first["t_obj"], first["t_grip"], first["t_event"]) == (0, 0, GRASP)  # a, left
    # The gripper point is the EEF at the grasp (frame 1), shifted to the prior's convention.
    assert first["t_grip_pos"][2] == pytest.approx(0.94 - A.EEF_TO_GRIP)
    np.testing.assert_allclose(first["t_final"][:3], (-0.1, -0.06, 0.975), atol=1e-6)
    assert items[1].target["t_event"] == PLACE
    assert items[2].target["t_obj"] == 1  # then b


def test_build_needs_two_rules_and_training_seeds_only():
    eps = {"leftmost": [_episode(101)], "rightmost": [_episode(102)]}
    sup = {"leftmost": SUPPORTS, "rightmost": SUPPORTS}
    with pytest.raises(ValueError, match="one rule"):
        FT.build({"leftmost": eps["leftmost"]}, sup)
    with pytest.raises(ValueError, match="not a training seed"):
        FT.build({**eps, "rightmost": [_episode(210)]}, sup)
    data = FT.build(eps, sup)
    # One frame-0 item per rule, each paired with the other rule's target.
    assert len(data.pairs) == 2
    for own, partner in data.pairs:
        assert np.array_equal(own.query.feats, partner.query.feats)
        want = {"leftmost": 0, "rightmost": 1}[partner.rule]  # a is left of b
        assert partner.target["t_obj"] == want
        assert partner.target["m_final"] == 0.0 and partner.target["m_aperture"] == 0.0


def test_a_mixed_batch_trains():
    eps = {"leftmost": [_episode(101), _episode(102)],
           "rightmost": [_episode(103), _episode(104)]}
    data = FT.build(eps, {r: SUPPORTS for r in eps})
    rng = np.random.default_rng(0)
    real = FT.sample(data, rng, 8, pair_fraction=0.5)
    assert len(real["t_obj"]) >= 8 and set(real) == set(FT.KEYS)
    batch = FT.concat(real, FT.with_masks(sample_batch(rng, ItemConfig(language=(1, 0, 0)), 4)))
    torch.manual_seed(0)
    model = M.KeyframePFN(width=16, heads=2, layers=1).train()
    b = M.to_torch(batch, None, "cpu")
    total, terms = M.loss(model, model(b), b)
    total.backward()
    assert torch.isfinite(total) and all(np.isfinite(v) for v in terms.values())


def test_a_low_data_subset_is_deterministic_and_training_only():
    eps = [_episode(s) for s in (101, 105, 110, 150, 360)]
    a = FT.subset(eps, 2, seed=0, rule="leftmost")
    assert a == FT.subset(list(reversed(eps)), 2, seed=0, rule="leftmost")  # listing order
    assert len(a) == 2 and [e.seed for e in a] == sorted(e.seed for e in a)
    # Another seed or rule draws independently; None or n >= len keeps everything.
    draws = {tuple(e.seed for e in FT.subset(eps, 2, seed=s, rule="leftmost")) for s in range(10)}
    assert len(draws) > 1
    assert [e.seed for e in FT.subset(eps, None, 0, "rightmost")] == [101, 105, 110, 150, 360]
    with pytest.raises(ValueError, match="not a training seed"):
        FT.subset(eps + [_episode(210)], 2, 0, "leftmost")
    with pytest.raises(ValueError, match="at least one"):
        FT.subset(eps, 0, 0, "leftmost")


def test_two_demos_per_rule_still_train():
    eps = {r: FT.subset([_episode(s) for s in (101, 102, 103)], 2, 0, r)
           for r in ("leftmost", "rightmost")}
    data = FT.build(eps, {r: SUPPORTS for r in eps})
    batch = FT.sample(data, np.random.default_rng(0), 8, pair_fraction=0.5)
    assert len(batch["t_obj"]) >= 8
