# ruff: noqa: E402
"""Does the gradient actually reach the entity encoder, and is it the right one?

`validate_control_act.py` showed the loss falling and the output responding to
changed entity inputs. Neither proves what it looks like it proves: the encoder
could sit at its random initialisation forever while only the zero-initialised
K/V projections adapt, and the output would still move when the entities did.
So these check the encoder's own parameters -- that they receive gradient, that
the gradient is numerically the derivative of the loss, and that the optimiser
moves them.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
from lerobot.configs.types import FeatureType, PolicyFeature

from oct_vla.policies.control_act import ControlACTConfig, ControlACTPolicy
from oct_vla.policies.conditioning.kv import KVAttention

class _EntityConfig:
    """The fields LayerwiseObjectAttention reads off a policy config."""

    object_representation = "entity_v2"
    object_attention_heads = 2
    object_entity_normalizer = None
    object_token_dim = 17
    effective_object_token_dim = 17


ENCODER_PREFIXES = ("embedding.numeric_projection", "embedding.type_projection", "embedding.output_norm")


def policy(dropout=0.0, seed=7):
    torch.manual_seed(seed)
    config = ControlACTConfig(
        device="cpu",
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=2,
        chunk_size=4,
        n_action_steps=2,
        use_vae=False,
        dropout=dropout,
        pretrained_backbone_weights=None,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
            "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            "observation.entity_tokens": PolicyFeature(type=FeatureType.STATE, shape=(8, 17)),
            "observation.entity_mask": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
    )
    return ControlACTPolicy(config)


def data(entities=3):
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :entities] = True
    return {
        "observation.state": torch.randn(2, 16),
        "observation.images.head": torch.rand(2, 3, 32, 32),
        "observation.entity_tokens": torch.randn(2, 8, 17),
        "observation.entity_mask": mask,
        "action": torch.randn(2, 4, 16),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }


def object_grads(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if "object_conditioning" in name
    }


def test_only_the_value_projection_moves_on_the_first_step():
    """The ControlNet property, from the mathematics rather than by assertion.

    With V_z = 0 the branch output is zero, so d(out)/d(V_z) is the softmax
    weights -- non-zero -- while d(out)/d(K_z) and everything upstream of it is
    proportional to V_z and vanishes. An encoder receiving gradient at step 0
    would mean the zero initialisation is not what it claims to be.
    """
    model = policy()
    loss, _ = model(data())
    loss.backward()
    grads = object_grads(model)
    assert grads, "no object parameters found"
    moved = {n for n, p in grads.items() if p.grad is not None and p.grad.abs().sum() > 0}
    assert moved, "nothing in the object path received gradient"
    assert all(n.endswith("to_v.weight") or n.endswith("to_v.bias") for n in moved), sorted(moved)


def test_the_encoder_receives_gradient_once_the_value_projection_has_moved():
    """The question `entity_sensitivity` cannot answer.

    Output sensitivity to entity inputs is satisfied by K/V alone; the encoder
    could stay frozen at its random init and the test would still pass. This
    asserts the encoder's *own* parameters get gradient and are moved by the
    optimiser.
    """
    model = policy()
    batch = data()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    before = {
        name: parameter.detach().clone()
        for name, parameter in object_grads(model).items()
        if any(prefix in name for prefix in ENCODER_PREFIXES)
    }
    assert before, f"no encoder parameters matched {ENCODER_PREFIXES}"

    for _ in range(4):
        optimizer.zero_grad()
        loss, _ = model(batch)
        loss.backward()
        optimizer.step()

    grads = object_grads(model)
    for name in before:
        grad = grads[name].grad
        assert grad is not None, f"{name} has no .grad"
        assert torch.isfinite(grad).all(), f"{name} gradient is not finite"
        assert grad.abs().sum() > 0, f"{name} still receives zero gradient after 4 steps"
        assert not torch.equal(before[name], grads[name].detach()), f"{name} never moved"

    # And nothing anywhere in the object path is NaN or infinite.
    for name, parameter in grads.items():
        assert torch.isfinite(parameter).all(), f"{name} became non-finite"


def test_the_encoder_gradient_is_the_derivative_of_the_loss():
    """The backward pass is *valid*, not merely non-zero.

    A wired-up-but-wrong branch -- a transposed reshape, a mismatched mask --
    still produces a finite non-zero gradient and a falling loss. Central
    differences in float64 against the analytic gradient is what distinguishes
    the two.
    """
    torch.manual_seed(3)
    layer = KVAttention(_EntityConfig(), 8, heads=2).double()
    tokens = torch.randn(2, 5, 17, dtype=torch.float64)
    mask = torch.ones(2, 5, dtype=torch.bool)
    query = torch.randn(2, 2, 3, 4, dtype=torch.float64)
    weight = torch.eye(8, dtype=torch.float64)
    with torch.no_grad():  # leave the zero init, or every derivative is 0
        layer.to_k.weight.normal_(std=0.3)
        layer.to_v.weight.normal_(std=0.3)

    target = layer.embedding.numeric_projection.weight

    def loss_of(flat):
        with torch.no_grad():
            target.copy_(flat.view_as(target))
        return layer(query, tokens, mask, output_weight=weight).pow(2).sum()

    flat = target.detach().flatten().clone()
    analytic = torch.autograd.grad(loss_of(flat), target)[0].flatten().clone()

    # float64 end to end: the branch promotes its score accumulation to at
    # least float32, so a double-precision input keeps double precision and a
    # central difference is meaningful.
    eps = 1e-6
    for index in (0, 7, 23):
        plus, minus = flat.clone(), flat.clone()
        plus[index] += eps
        minus[index] -= eps
        numeric = (loss_of(plus).item() - loss_of(minus).item()) / (2 * eps)
        assert numeric == pytest.approx(analytic[index].item(), rel=1e-4, abs=1e-6), index


def test_padded_entities_contribute_no_gradient():
    """A zero row that still reaches the encoder would train it on padding."""
    torch.manual_seed(5)
    layer = KVAttention(_EntityConfig(), 8, heads=2)
    with torch.no_grad():
        layer.to_k.weight.normal_(std=0.3)
        layer.to_v.weight.normal_(std=0.3)
    query = torch.randn(2, 2, 3, 4)
    weight = torch.eye(8)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, :2] = True

    tokens = torch.randn(2, 6, 17, requires_grad=True)
    layer(query, tokens, mask, output_weight=weight).pow(2).sum().backward()
    assert tokens.grad[:, 2:].abs().sum() == 0, "gradient leaked into padded entity slots"
    assert tokens.grad[:, :2].abs().sum() > 0, "the real entities received no gradient"


def test_every_object_parameter_is_reachable_by_an_optimiser():
    """`requires_grad` alone is not enough -- a parameter left out of the
    optimiser trains nothing while the loss falls normally around it."""
    model = policy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    in_optimizer = {id(p) for group in optimizer.param_groups for p in group["params"]}
    for name, parameter in object_grads(model).items():
        assert parameter.requires_grad, f"{name} does not require grad"
        assert id(parameter) in in_optimizer, f"{name} is not in the optimiser"
