"""The ablation arms are only meaningful if these transforms really remove
what they claim to remove, so each test asserts the *absence* of a leak rather
than merely the presence of the right shape."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from oct_vla.data.token_transforms import (  # noqa: E402
    ROLE_DIMS,
    apply_token_mode,
    shuffle_tokens,
    strip_roles,
    token_dim_for_mode,
)

TARGET, PREVIOUS, OTHER = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def _token(position, role, *, fill=0.5):
    """A stored 15-d token: position (3), orientation (4), size (3),
    visibility/confidence (2), role (3)."""
    return [*position, *([fill] * 4), *([fill] * 3), 1.0, 1.0, *role]


def _scene(rows, slots=4):
    tokens = [_token(*row) for row in rows]
    mask = [1.0] * len(tokens) + [0.0] * (slots - len(tokens))
    tokens += [[0.0] * 15] * (slots - len(tokens))
    return torch.tensor([tokens]), torch.tensor([mask])


def test_token_dim_for_mode_drops_exactly_the_role_dims():
    assert token_dim_for_mode(15, "full") == 15
    assert token_dim_for_mode(15, "role_stripped") == 15 - ROLE_DIMS


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown object token mode"):
        token_dim_for_mode(15, "target_only")


def test_role_stripping_removes_the_role_one_hot():
    tokens, mask = _scene([((0.3, 0.0, 0.0), TARGET), ((0.1, 0.0, 0.0), OTHER)])
    stripped, _ = strip_roles(tokens, mask)
    assert stripped.shape[-1] == 15 - ROLE_DIMS
    # Every surviving dimension comes from the geometric prefix.
    assert torch.equal(stripped, tokens[..., : 15 - ROLE_DIMS].gather(
        1, torch.tensor([[1, 0, 2, 3]]).unsqueeze(-1).expand(-1, -1, 12)
    ))


def test_role_stripping_also_removes_the_target_first_ordering():
    """The ordering leaks the role even once the one-hot is gone: whichever
    scene is passed, the target must not land in a fixed slot."""
    target_first, mask = _scene([((0.9, 0.0, 0.0), TARGET), ((0.1, 0.0, 0.0), OTHER)])
    stripped, _ = strip_roles(target_first, mask)
    # Position 0.1 sorts before 0.9, so the non-target now leads.
    assert stripped[0, 0, 0].item() == pytest.approx(0.1)
    assert stripped[0, 1, 0].item() == pytest.approx(0.9)


def test_role_stripped_order_is_independent_of_the_input_order():
    """Two scenes identical except for which object was labelled target must
    become byte-identical -- that is exactly what 'no leakage' means."""
    a, mask = _scene([((0.9, 0.0, 0.0), TARGET), ((0.1, 0.0, 0.0), PREVIOUS)])
    b, _ = _scene([((0.1, 0.0, 0.0), TARGET), ((0.9, 0.0, 0.0), PREVIOUS)])
    assert not torch.equal(a, b)
    assert torch.equal(strip_roles(a, mask)[0], strip_roles(b, mask)[0])


def test_role_stripping_keeps_padding_at_the_tail():
    tokens, mask = _scene([((0.9, 0.0, 0.0), TARGET), ((0.1, 0.0, 0.0), OTHER)], slots=4)
    _, out_mask = strip_roles(tokens, mask)
    assert out_mask.tolist() == [[1.0, 1.0, 0.0, 0.0]]


def test_shuffle_preserves_the_multiset_and_the_valid_count():
    tokens, mask = _scene(
        [((0.1, 0.0, 0.0), TARGET), ((0.2, 0.0, 0.0), PREVIOUS), ((0.3, 0.0, 0.0), OTHER)],
        slots=4,
    )
    generator = torch.Generator().manual_seed(0)
    shuffled, out_mask = shuffle_tokens(tokens, mask, generator=generator)
    assert out_mask.tolist() == mask.tolist(), "padding must stay at the tail"
    assert sorted(shuffled[0, :3, 0].tolist()) == pytest.approx([0.1, 0.2, 0.3])


def test_shuffle_actually_permutes_some_scene():
    tokens, mask = _scene(
        [((float(i) / 10, 0.0, 0.0), OTHER) for i in range(8)], slots=8
    )
    batch = tokens.expand(64, -1, -1).contiguous(), mask.expand(64, -1).contiguous()
    generator = torch.Generator().manual_seed(0)
    shuffled, _ = shuffle_tokens(*batch, generator=generator)
    assert not torch.equal(shuffled, batch[0]), "shuffle left every scene untouched"


def test_apply_token_mode_collapses_a_history_axis():
    tokens, mask = _scene([((0.9, 0.0, 0.0), TARGET), ((0.1, 0.0, 0.0), OTHER)])
    out, out_mask = apply_token_mode(tokens.unsqueeze(1), mask.unsqueeze(1), mode="full")
    assert out.shape == tokens.shape
    assert out_mask.shape == mask.shape


def test_apply_token_mode_is_a_no_op_without_tokens():
    assert apply_token_mode(None, None, mode="role_stripped") == (None, None)


def test_shuffle_runs_after_role_stripping():
    """A role-stripped model's shuffled control must scramble the 12-d tokens
    that model was trained on, not the stored 15-d ones."""
    tokens, mask = _scene([((0.9, 0.0, 0.0), TARGET), ((0.1, 0.0, 0.0), OTHER)])
    out, _ = apply_token_mode(tokens, mask, mode="role_stripped", shuffle=True)
    assert out.shape[-1] == 15 - ROLE_DIMS
    assert sorted(out[0, :2, 0].tolist()) == pytest.approx([0.1, 0.9])
