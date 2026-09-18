"""Padded action targets are repeats LeRobot inserts where a chunk runs past the
end of an episode. ACT masks them out of its loss; pi0.5's `losses.mean()` does
not. On this corpus that is 17.4% of all targets and 40.2% on the shortest
clips, so the difference is not marginal."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from oct_vla.policies.masked_loss import masked_loss, padded_fraction  # noqa: E402


def batch(pad):
    return {"action_is_pad": torch.tensor(pad, dtype=torch.bool)}


def test_padded_timesteps_do_not_contribute():
    """The second timestep is padding and carries a huge error; masking it must
    leave the loss equal to the first timestep's alone."""
    losses = torch.tensor([[[1.0, 1.0], [99.0, 99.0]]])
    assert masked_loss(losses, batch([[False, True]])) == pytest.approx(1.0)


def test_the_loss_keeps_its_scale_whatever_fraction_is_padded():
    """Normalising by the full element count instead of the valid one would make
    batches near an episode's end look artificially good, and quietly reweight
    them against batches from the middle."""
    losses = torch.full((1, 4, 2), 3.0)
    for pad in ([[False] * 4], [[False, False, True, True]], [[False, True, True, True]]):
        assert masked_loss(losses, batch(pad)) == pytest.approx(3.0)


def test_an_unmasked_batch_is_reduced_normally():
    """A dataloader built without delta_timestamps carries no action_is_pad.
    Returning a masked zero there would be worse than the padding."""
    losses = torch.tensor([[[2.0, 4.0]]])
    assert masked_loss(losses, {}) == pytest.approx(3.0)


def test_a_fully_padded_sample_does_not_divide_by_zero():
    losses = torch.full((1, 2, 2), 5.0)
    assert torch.isfinite(masked_loss(losses, batch([[True, True]])))


def test_per_sample_reduction_normalises_each_sample_separately():
    """A sample that is mostly padding must not be scored as mostly correct just
    because its few valid steps are outnumbered."""
    losses = torch.tensor([
        [[2.0], [2.0]],      # no padding -> 2.0
        [[6.0], [100.0]],    # second step padded -> 6.0
    ])
    out = masked_loss(losses, batch([[False, False], [False, True]]), reduction="none")
    assert out.shape == (2,)
    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(6.0)


def test_padded_fraction_reports_what_was_excluded():
    assert padded_fraction(batch([[False, False, True, True]])) == pytest.approx(0.5)
    assert padded_fraction({}) == 0.0
