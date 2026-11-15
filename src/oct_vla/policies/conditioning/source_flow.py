"""Conditional-source flow for GR00T N1.7: start the action flow from a proposal.

GR00T's flow-matching head (LeRobot ``groot_n1_7.GR00TN17ActionHead``) draws its source
inline: ``noise = torch.randn(...)`` in ``forward`` (training; x_t = (1-t) noise + t a,
target a - noise) and ``actions = torch.randn(...)`` in ``get_action_with_features``
(4 Euler steps from it). With ``action_source="via"`` both draws become

    x0 = mu + sigma * eps   on the 16 eeabs columns, with probability rho per sample,
    x0 = eps                otherwise, and on the padding columns,

where mu is the normalised kinematic proposal (``action_source.via_point_proposal``)
from the per-arm via-points, and eps is GR00T's own Gaussian draw. The per-sample
choice draws nothing from torch's generator: it is u < rho with u = Phi(eps) on a
padding column (``DECISION``), which is Uniform(0, 1) and independent of the 16 action
columns. So rho = 0 is GR00T's Gaussian path bit for bit, and a rollout seeded per
scene stays reproducible whatever ran before it. Training and
inference use the same mixture (plan rev. 8), and a sample whose source is invalid
(no via-point) always gets eps.

Installation does not copy LeRobot's methods. It wraps them and, for the duration of
the call only, gives the head's module a ``torch`` whose ``randn`` passes GR00T's draw
through the source. The wrapper asserts exactly one draw per call, of the action
shape: if LeRobot ever adds a draw or moves this one, every call is refused rather
than silently sampling from the wrong source.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import torch
from torch import Tensor

ACTION_COLUMNS = 16
#: The padding entry of GR00T's draw whose value decides, per sample, which source
#: is used (step 0, the last action column: masked out of the loss, never executed).
DECISION = (0, -1)


@dataclass
class Source:
    """One batch's source: normalised proposal mu [B, H, 16], which samples have one
    (valid [B]), the mixture weight rho and the spread sigma. `use` is filled when
    GR00T draws."""

    mu: Tensor
    valid: Tensor
    rho: float
    sigma: float
    use: Tensor | None = None

    def apply(self, eps: Tensor) -> Tensor:
        if eps.ndim != 3 or eps.shape[0] != self.mu.shape[0] or eps.shape[1] != self.mu.shape[1]:
            raise RuntimeError(f"source {tuple(self.mu.shape)} does not fit GR00T's draw "
                               f"{tuple(eps.shape)}")
        if eps.shape[-1] <= ACTION_COLUMNS:
            raise RuntimeError("GR00T's draw has no padding column to decide the source with")
        u = 0.5 * (1 + torch.erf(eps[:, DECISION[0], DECISION[1]].float() / 2 ** 0.5))
        self.use = self.valid.to(eps.device) & (u < self.rho)
        if not bool(self.use.any()):
            return eps
        out = eps.clone()
        sel = self.use
        mu = self.mu.to(device=eps.device, dtype=eps.dtype)
        out[sel, :, :ACTION_COLUMNS] = mu[sel] + self.sigma * eps[sel, :, :ACTION_COLUMNS]
        return out


class _TorchWithSource(types.ModuleType):
    """`torch`, except `randn`, which routes GR00T's draw through the source."""

    def __init__(self, source: Source, draws: list):
        super().__init__("torch")
        self._source, self._draws = source, draws

    def __getattr__(self, name):
        return getattr(torch, name)

    def randn(self, *args, **kwargs):
        eps = torch.randn(*args, **kwargs)
        self._draws.append(tuple(eps.shape))
        return self._source.apply(eps)


def install_source_flow(head) -> None:
    """Wrap `head.forward` and `head.get_action_with_features`; idempotent per head.

    `head.action_source` (a `Source` or None) is set per batch by the policy. None means
    the policy did not set one, which is refused: a source arm that silently fell back
    to Gaussian would train and score as plain kv."""
    if getattr(head, "_source_flow_installed", False):
        return
    module = __import__(type(head).__module__, fromlist=["torch"])
    if getattr(module, "torch", None) is not torch:
        raise RuntimeError(f"{module.__name__} has no module-level torch to route; the "
                           "source flow cannot install on this LeRobot version")
    head.action_source = None
    head.source_counts = {"samples": 0, "used": 0}

    def wrap(method, name):
        def run(*args, **kwargs):
            source = head.action_source
            if source is None:
                raise RuntimeError(f"{name}: the source flow is installed but no source was "
                                   "set for this batch")
            draws: list = []
            module.torch = _TorchWithSource(source, draws)
            try:
                out = method(*args, **kwargs)
            finally:
                module.torch = torch
            if len(draws) != 1 or len(draws[0]) != 3:
                raise RuntimeError(f"{name} drew {draws} from torch.randn; the source flow "
                                   "expects exactly one draw of the action chunk")
            head.source_counts["samples"] += int(source.use.numel())
            head.source_counts["used"] += int(source.use.sum())
            return out
        return run

    head.forward = wrap(head.forward, "forward")
    head.get_action_with_features = wrap(head.get_action_with_features, "get_action_with_features")
    head._source_flow_installed = True


def make_source(tokens: Tensor, entity_mask: Tensor, targets: Tensor, target_mask: Tensor, *,
                lo: Tensor, hi: Tensor, rho: float, sigma: float, speeds, dwell: int,
                horizon: int) -> Source:
    """The batch's source from raw entity tokens and per-arm via-points."""
    from oct_vla.policies.conditioning.action_source import normalize_actions, via_point_proposal

    with torch.no_grad():
        raw, valid = via_point_proposal(tokens.float(), entity_mask.bool(), targets.float(),
                                        target_mask.bool(), speed=speeds, dwell=dwell,
                                        horizon=horizon)
        mu = normalize_actions(raw, lo.to(raw), hi.to(raw))
    return Source(mu=mu, valid=valid, rho=rho, sigma=sigma)
