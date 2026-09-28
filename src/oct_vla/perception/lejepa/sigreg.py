"""SIGReg: Sketched Isotropic Gaussian Regularisation (LeJEPA, Balestriero & LeCun 2025).

A joint-embedding objective with no teacher, no EMA and no stop-gradient has
one trivial minimiser: every input maps to the same point. LeJEPA's answer is
to require the embedding distribution to be an isotropic Gaussian, which it
shows is the distribution minimising downstream prediction risk, and to test
that requirement cheaply: by Cramer-Wold a distribution is N(0, I) iff every
1-D projection is N(0, 1), so it projects the batch onto M random unit
directions and runs a univariate goodness-of-fit test on each.

The test is Epps-Pulley: the squared distance between the empirical
characteristic function of the projected samples and the standard normal's,
exp(-t^2 / 2), weighted by that same Gaussian and integrated over t by
trapezoidal quadrature, times the sample count N. Characteristic functions are
bounded, so the statistic has bounded gradients even for outlying samples --
the reason the paper prefers it over moment matching. Scaled by N it is O(1)
under the null and grows linearly in N for any fixed departure, so collapse
(all samples equal) or a low-rank embedding (most directions near zero
variance) is expensive.

Embeddings are not normalised onto a sphere: the target is N(0, I) itself, so
scale is part of what is regularised.
"""

from __future__ import annotations

import torch

DEFAULT_SLICES = 256
#: The quadrature grid of the reference implementation: 17 knots on [-5, 5].
#: The Gaussian weight is < 4e-6 beyond |t| = 5, so the tails are negligible.
DEFAULT_T_MAX = 5.0
DEFAULT_KNOTS = 17


def random_directions(
    dim: int, slices: int, *, step: int | None = None, device=None, dtype=torch.float32
) -> torch.Tensor:
    """(dim, slices) unit columns. Seeded by `step` when given, so every
    process of a multi-GPU run (and a rerun) tests the same directions."""
    generator = None
    if step is not None:
        generator = torch.Generator(device=device or "cpu")
        generator.manual_seed(int(step))
    directions = torch.randn(dim, slices, generator=generator, device=device, dtype=dtype)
    return directions / directions.norm(dim=0, keepdim=True)


def epps_pulley(
    samples: torch.Tensor, *, t_max: float = DEFAULT_T_MAX, knots: int = DEFAULT_KNOTS
) -> torch.Tensor:
    """Epps-Pulley statistic of each column of `samples` (N, K) against N(0, 1).

    Returns (K,). Written with cos/sin means rather than a complex exponential
    so it runs under autocast and on backends without complex support.
    """
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError(f"expected (N >= 2, K) samples, got {tuple(samples.shape)}")
    samples = samples.float()
    t = torch.linspace(-t_max, t_max, knots, device=samples.device, dtype=samples.dtype)
    target = torch.exp(-0.5 * t.square())
    phase = samples.unsqueeze(-1) * t  # (N, K, T)
    real = torch.cos(phase).mean(dim=0)
    imag = torch.sin(phase).mean(dim=0)
    error = ((real - target).square() + imag.square()) * target
    return torch.trapezoid(error, t, dim=-1) * samples.shape[0]


def sigreg(
    embeddings: torch.Tensor,
    *,
    slices: int = DEFAULT_SLICES,
    step: int | None = None,
    directions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean Epps-Pulley statistic of `embeddings` (N, D) over random 1-D slices."""
    if embeddings.ndim != 2:
        raise ValueError(f"expected (N, D) embeddings, got {tuple(embeddings.shape)}")
    if directions is None:
        directions = random_directions(
            embeddings.shape[1], slices, step=step, device=embeddings.device
        )
    return epps_pulley(embeddings.float() @ directions.float()).mean()


def effective_rank(embeddings: torch.Tensor) -> float:
    """exp(entropy) of the normalised singular values of the covariance.

    1 for a collapsed embedding, D for an isotropic one. The number to watch:
    a falling loss with a falling effective rank is collapse in progress.
    """
    with torch.no_grad():
        centred = embeddings.float() - embeddings.float().mean(dim=0, keepdim=True)
        covariance = centred.T @ centred / max(embeddings.shape[0] - 1, 1)
        values = torch.linalg.svdvals(covariance)
        total = values.sum()
        if not torch.isfinite(total) or total <= 0:
            return 1.0
        p = values / total
        p = p[p > 0]
        return float(torch.exp(-(p * p.log()).sum()))
