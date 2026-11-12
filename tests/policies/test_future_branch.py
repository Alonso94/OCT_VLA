"""The future-keyframe branch (conditioning/future.py) on GR00T's real DiT.

Acceptance tests from the PFN-VLA plan: with the branch present but untrained
the actions are bit-identical to plain kv, at init and after a checkpoint
reload; gradients reach the branch; a trained branch responds to the keyframe;
and the policy boundary refuses what would otherwise run silently as kv.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from torch import nn  # noqa: E402

from oct_vla.policies.conditioning.entity import TOKEN_DIM  # noqa: E402
from oct_vla.policies.conditioning.future import (  # noqa: E402
    FUTURE_DIM,
    FUTURE_KEYFRAME,
    FUTURE_MASK,
    check_future_inputs,
)
from oct_vla.policies.conditioning.hosts import install_diffusers_kv  # noqa: E402
from oct_vla.policies.object_conditioning import (  # noqa: E402
    ObjectConditionedPolicyMixin,
    ObjectConditioning,
    ObjectConditioningError,
)

ROWS = 3


LATENT_ROWS, LATENT_DIM = 10, 16


def _config(future: bool, kind: str = "keyframe"):
    latent = kind == "latent"
    return SimpleNamespace(object_entity_normalizer=None, object_width=None,
                           object_visual_dim=None, object_future=future,
                           object_future_rows=LATENT_ROWS if latent else ROWS,
                           object_future_kind=kind,
                           object_future_dim=LATENT_DIM if latent else FUTURE_DIM,
                           object_conditioning="kv")


def _model(future: bool, seed: int = 0, kind: str = "keyframe"):
    from lerobot.policies.groot.action_head.cross_attention_dit import DiT

    torch.manual_seed(seed)
    dit = DiT(num_attention_heads=2, attention_head_dim=4, output_dim=3, num_layers=2,
              dropout=0.0, cross_attention_dim=6, interleave_self_attention=True,
              positional_embeddings=None, final_dropout=False)
    owner = nn.Module()
    owner.model = dit
    owner.object_conditioning = ObjectConditioning(_config(future, kind), dit.inner_dim)
    install_diffusers_kv(dit, owner.object_conditioning, conditioning_owner=owner)
    torch.manual_seed(seed + 1)
    for branch in owner.object_conditioning.layers.values():  # a trained kv model
        for layer in (branch.to_k, branch.to_v):
            nn.init.normal_(layer.weight, std=0.3)
    return owner.eval()


def _scene(seed=0, batch=2):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(batch, 4, TOKEN_DIM, generator=g)
    mask = torch.tensor([[True, True, True, False]] * batch)
    keyframe = torch.randn(batch, ROWS, FUTURE_DIM, generator=g)
    future_mask = torch.ones(batch, ROWS, dtype=torch.bool)
    hidden = torch.randn(batch, 5, 8, generator=g)
    context = torch.randn(batch, 7, 6, generator=g)
    return tokens, mask, keyframe, future_mask, hidden, context


def _run(owner, scene, *, future=True):
    tokens, mask, keyframe, future_mask, hidden, context = scene
    control = owner.object_conditioning
    control.set_inputs(tokens, mask)
    if future and control.has_future:
        control.set_future_inputs(keyframe, future_mask)
    try:
        return owner.model(hidden, context, timestep=torch.tensor([3, 7]))
    finally:
        control.clear()


def _copy_kv(source, target):
    keys = {k: v for k, v in source.state_dict().items()}
    missing, unexpected = target.load_state_dict(keys, strict=False)
    assert not unexpected and all(".future." in k for k in missing)


def test_untrained_branch_is_bit_identical_to_kv():
    kv, both = _model(False), _model(True)
    _copy_kv(kv, both)
    scene = _scene()
    assert torch.equal(_run(kv, scene), _run(both, scene))
    assert not any(b.is_live for b in both.object_conditioning.future.values())


def test_kv_checkpoints_carry_no_future_keys():
    assert not any("future" in k for k in _model(False).state_dict())
    assert any(".future." in k for k in _model(True).state_dict())


def test_gradient_reaches_the_branch():
    owner = _model(True).train()
    out = _run(owner, _scene())
    out.square().sum().backward()
    future = owner.object_conditioning.future
    assert all(b.to_v.weight.grad is not None and b.to_v.weight.grad.abs().sum() > 0
               for b in future.values())


def _train_future(owner, seed=5):
    torch.manual_seed(seed)
    for branch in owner.object_conditioning.future.values():
        for layer in (branch.to_k, branch.to_v):
            nn.init.normal_(layer.weight, std=0.3)


def test_a_trained_branch_responds_to_the_keyframe_and_only_through_it():
    kv, both = _model(False), _model(True)
    _copy_kv(kv, both)
    _train_future(both)
    scene = _scene()
    base = _run(both, scene)
    moved = list(scene)
    moved[2] = scene[2] + 1.0
    assert not torch.allclose(base, _run(both, tuple(moved)))
    # An all-padding keyframe is an exact zero residual: kv again, bit for bit.
    empty = list(scene)
    empty[3] = torch.zeros_like(scene[3])
    assert torch.equal(_run(both, tuple(empty)), _run(kv, scene))


def test_reload_reproduces_actions(tmp_path):
    both = _model(True)
    _train_future(both)
    path = tmp_path / "head.pt"
    torch.save(both.state_dict(), path)
    fresh = _model(True, seed=9)
    fresh.load_state_dict(torch.load(path))
    scene = _scene(seed=3)
    assert torch.equal(_run(both, scene), _run(fresh, scene))


@pytest.mark.parametrize("shape", [(2, ROWS, FUTURE_DIM - 1), (2, 1, ROWS, FUTURE_DIM),
                                   (2, ROWS + 1, FUTURE_DIM)])
def test_shape_contract_is_strict(shape):
    with pytest.raises(ValueError):
        check_future_inputs(torch.zeros(shape), torch.ones(2, ROWS, dtype=torch.bool), rows=ROWS)


class _Policy(ObjectConditionedPolicyMixin, nn.Module):
    object_module_path = "head"

    def __init__(self, owner, future=True):
        nn.Module.__init__(self)
        self.head = owner
        self.config = _config(future)


def _batch(scene, **drop):
    tokens, mask, keyframe, future_mask, *_ = scene
    batch = {"observation.entity_tokens": tokens, "observation.entity_mask": mask,
             FUTURE_KEYFRAME: keyframe, FUTURE_MASK: future_mask}
    for key in drop.get("keys", ()):
        batch.pop(key)
    return batch


def test_policy_feeds_the_keyframe_and_refuses_without_it():
    policy = _Policy(_model(True))
    policy._set_object_inputs(_batch(_scene()))
    assert policy.object_conditioning._future_inputs is not None
    policy._clear_object_inputs()
    assert policy.object_conditioning._future_inputs is None
    with pytest.raises(ValueError, match="future_keyframe"):
        policy._set_object_inputs(_batch(_scene(), keys=[FUTURE_KEYFRAME]))


def test_policy_refuses_a_host_that_built_no_future_layers():
    owner = nn.Module()
    owner.object_conditioning = ObjectConditioning(_config(True), 8)  # nothing installed
    with pytest.raises(ObjectConditioningError, match="no future layers"):
        _Policy(owner)._set_object_inputs(_batch(_scene()))


def test_unbatched_inference_inputs_get_a_batch_dimension():
    policy = _Policy(_model(True))
    tokens, mask, keyframe, future_mask, *_ = _scene(batch=1)
    policy._set_object_inputs({"observation.entity_tokens": tokens[0],
                               "observation.entity_mask": mask[0],
                               FUTURE_KEYFRAME: keyframe[0], FUTURE_MASK: future_mask[0]})
    assert policy.object_conditioning._future_inputs[0].shape == (1, ROWS, FUTURE_DIM)


def test_config_refuses_future_without_a_kv_branch():
    from oct_vla.policies.control_groot import ControlGrootConfig

    with pytest.raises(ValueError, match="no KV branch"):
        ControlGrootConfig(object_conditioning="scene_attn", object_future=True)


def _latent_scene(seed=0, batch=2):
    tokens, mask, _, _, hidden, context = _scene(seed, batch)
    g = torch.Generator().manual_seed(seed + 100)
    cells = torch.randn(batch, LATENT_ROWS, LATENT_DIM, generator=g)
    cell_mask = torch.ones(batch, LATENT_ROWS, dtype=torch.bool)
    cell_mask[:, 7:9] = False  # padded entity columns of the prior
    return tokens, mask, cells, cell_mask, hidden, context


def test_the_latent_kind_builds_its_own_projection_and_starts_as_kv():
    from oct_vla.policies.conditioning.future import FutureLatentAttention

    kv, latent = _model(False), _model(True, kind="latent")
    assert all(isinstance(b, FutureLatentAttention)
               for b in latent.object_conditioning.future.values())
    _copy_kv(kv, latent)
    scene = _latent_scene()
    assert torch.equal(_run(kv, scene), _run(latent, scene))


def test_the_latent_kind_learns_and_responds_to_the_cells():
    owner = _model(True, kind="latent").train()
    _run(owner, _latent_scene()).square().sum().backward()
    assert all(b.to_v.weight.grad.abs().sum() > 0
               for b in owner.object_conditioning.future.values())
    owner = _model(True, kind="latent")
    _train_future(owner)
    scene = _latent_scene()
    moved = list(scene)
    moved[2] = scene[2] * 2.0 + 1.0
    assert not torch.allclose(_run(owner, scene), _run(owner, tuple(moved)))


def test_the_shape_contract_follows_the_kind():
    cells = torch.zeros(2, LATENT_ROWS, LATENT_DIM)
    mask = torch.ones(2, LATENT_ROWS, dtype=torch.bool)
    check_future_inputs(cells, mask, rows=LATENT_ROWS, dim=LATENT_DIM)
    with pytest.raises(ValueError):
        check_future_inputs(cells, mask, rows=LATENT_ROWS)  # keyframe width expected


@pytest.mark.parametrize("kind, dim", [("latent", 0), ("keyframe", 128), ("cells", 24)])
def test_config_refuses_an_inconsistent_kind(kind, dim):
    from oct_vla.policies.control_groot import ControlGrootConfig

    with pytest.raises(ValueError, match="object_future"):
        ControlGrootConfig(object_conditioning="kv", object_future=True,
                           object_future_kind=kind, object_future_dim=dim)
