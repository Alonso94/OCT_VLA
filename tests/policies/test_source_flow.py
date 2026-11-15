"""The conditional-source flow (conditioning/source_flow.py)."""

from __future__ import annotations

import inspect
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from oct_vla.policies.conditioning.source_flow import (  # noqa: E402
    Source,
    install_source_flow,
    make_source,
)

H, D = 40, 132


def _fake_head(draws_per_call=1):
    """A stand-in head whose module draws its source with `torch.randn`, as GR00T's does."""
    module = types.ModuleType("fake_groot_head")
    module.torch = torch
    code = f'''
class Head:
    def forward(self, actions):
        noise = None
        for _ in range({draws_per_call}):
            noise = torch.randn(actions.shape, dtype=actions.dtype)
        return noise
    def get_action_with_features(self, batch):
        return torch.randn(size=(batch, {H}, {D}))
'''
    exec(code, module.__dict__)
    module.Head.__module__ = module.__name__
    sys.modules[module.__name__] = module
    return module.Head()


def test_lerobot_draws_its_source_exactly_once_where_we_route_it():
    """Drift guard: the wrapped methods each contain one `torch.randn(` (the source)."""
    groot = pytest.importorskip("lerobot.policies.groot.groot_n1_7")
    head = groot.GR00TN17ActionHead
    for name in ("forward", "get_action_with_features"):
        assert inspect.getsource(getattr(head, name)).count("torch.randn(") == 1, name
    assert groot.torch is torch


def test_rho_zero_is_gaussian_bit_for_bit_and_rho_one_starts_at_the_proposal():
    head = _fake_head()
    install_source_flow(head)
    mu = torch.rand(2, H, 16) * 2 - 1
    torch.manual_seed(0)
    plain = torch.randn(2, H, D)
    head.action_source = Source(mu=mu, valid=torch.tensor([True, True]), rho=0.0, sigma=0.5)
    torch.manual_seed(0)
    assert torch.equal(head.forward(torch.zeros(2, H, D)), plain)  # rho 0: Gaussian exactly
    head.action_source = Source(mu=mu, valid=torch.tensor([True, False]), rho=1.0, sigma=0.5)
    torch.manual_seed(0)
    x0 = head.forward(torch.zeros(2, H, D))
    torch.testing.assert_close(x0[0, :, :16], mu[0] + 0.5 * plain[0, :, :16])
    assert torch.equal(x0[0, :, 16:], plain[0, :, 16:])  # padding stays Gaussian
    assert torch.equal(x0[1], plain[1])  # an unused sample stays Gaussian
    assert head.source_counts == {"samples": 4, "used": 1}
    # The RNG stream is untouched by the choice: the next draw is the same either way.
    assert torch.equal(torch.randn(3), (torch.manual_seed(0), torch.randn(2, H, D),
                                        torch.randn(3))[2])


def test_the_choice_follows_rho_and_is_a_function_of_the_draw():
    head = _fake_head()
    install_source_flow(head)
    n = 4000
    head.action_source = Source(mu=torch.zeros(n, H, 16), valid=torch.ones(n, dtype=torch.bool),
                                rho=0.3, sigma=0.5)
    torch.manual_seed(3)
    head.forward(torch.zeros(n, H, D))
    first = head.action_source.use.clone()
    assert abs(first.float().mean().item() - 0.3) < 0.03
    torch.manual_seed(3)
    head.forward(torch.zeros(n, H, D))
    assert torch.equal(head.action_source.use, first)


def test_inference_routes_too_and_a_missing_source_is_refused():
    head = _fake_head()
    install_source_flow(head)
    head.action_source = Source(mu=torch.zeros(1, H, 16), valid=torch.tensor([True]), rho=1.0,
                                sigma=1e-6)
    out = head.get_action_with_features(1)
    assert out[:, :, :16].abs().max() < 1e-4
    head.action_source = None
    with pytest.raises(RuntimeError, match="no source"):
        head.get_action_with_features(1)


def test_an_extra_draw_is_refused_not_silently_routed():
    head = _fake_head(draws_per_call=2)
    install_source_flow(head)
    head.action_source = Source(mu=torch.zeros(1, H, 16), valid=torch.tensor([False]), rho=1.0,
                                sigma=0.5)
    with pytest.raises(RuntimeError, match="exactly one draw"):
        head.forward(torch.zeros(1, H, D))
    assert sys.modules["fake_groot_head"].torch is torch  # restored after the failure


def test_make_source_uses_only_valid_samples_and_normalises():
    from oct_vla.core.frames import WORKCELL_FRAME, Pose
    from oct_vla.data.entity_tokens import _token

    down = (-0.5, 0.5, 0.5, 0.5)
    tokens = torch.tensor([[_token(Pose((-0.3, -0.1, 1.2), down, WORKCELL_FRAME), (0, 0, 0),
                                   0.83, "left_gripper"),
                            _token(Pose((0.4, -0.2, 1.2), down, WORKCELL_FRAME), (0, 0, 0),
                                   0.83, "right_gripper")]])
    emask = torch.ones(1, 2, dtype=torch.bool)
    t = torch.zeros(1, 2, 6, 8)
    t[0, 0, 0] = torch.tensor([0.0, 0.1, 1.2, *down, 0.0])
    m = torch.zeros(1, 2, 6, dtype=torch.bool)
    m[0, 0, 0] = True
    t2 = torch.cat([t, t]).float()
    m2 = torch.cat([m, torch.zeros_like(m)])
    tok2, em2 = torch.cat([tokens, tokens]).float(), torch.cat([emask, emask])
    lo, hi = torch.full((16,), -2.0), torch.full((16,), 2.0)
    src = make_source(tok2, em2, t2, m2, lo=lo, hi=hi, rho=1.0, sigma=0.3, speeds=(0.3, 0.2),
                      dwell=4, horizon=H)
    assert src.valid.tolist() == [True, False]
    assert src.mu.shape == (2, H, 16) and (src.mu.abs() <= 1).all()
    # The last step is the via-point, normalised: (0 - (-2)) / 4 * 2 - 1 = 0 for x.
    assert src.mu[0, -1, 0] == pytest.approx(0.0, abs=1e-6)


def test_config_requires_the_action_stats_for_a_source():
    from oct_vla.policies.control_groot import ControlGrootConfig

    with pytest.raises(ValueError, match="action_source_min"):
        ControlGrootConfig(object_conditioning="kv", action_source="via")
    ControlGrootConfig(object_conditioning="kv", action_source="via",
                       action_source_min=[-1.0] * 16, action_source_max=[1.0] * 16)
    with pytest.raises(ValueError, match="action_source must be"):
        ControlGrootConfig(object_conditioning="kv", action_source="pfn")
