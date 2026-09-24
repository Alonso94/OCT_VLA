"""aug_act: the augmentations act in training, only where configured, and never at eval."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402

from oct_vla.policies.aug_act import AugACTConfig, AugACTPolicy  # noqa: E402
from oct_vla.policies.aug_act.modeling_aug_act import random_shift  # noqa: E402

HEAD, WRIST = "observation.images.head", "observation.images.left_wrist"


def _policy(**overrides):
    torch.manual_seed(0)
    config = AugACTConfig(
        device="cpu",
        input_features={HEAD: PolicyFeature(FeatureType.VISUAL, (3, 24, 32)),
                        WRIST: PolicyFeature(FeatureType.VISUAL, (3, 24, 32)),
                        "observation.state": PolicyFeature(FeatureType.STATE, (4,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (4,))},
        chunk_size=4, n_action_steps=2, dim_model=16, n_heads=2, dim_feedforward=32,
        n_encoder_layers=1, n_decoder_layers=1, latent_dim=4, n_vae_encoder_layers=1,
        pretrained_backbone_weights=None, **overrides,
    )
    return AugACTPolicy(config)


def _batch(n=256):
    return {HEAD: torch.rand(n, 3, 24, 32) + 1, WRIST: torch.rand(n, 3, 24, 32) + 1,
            "observation.state": torch.randn(n, 4), "action": torch.randn(n, 4, 4),
            "action_is_pad": torch.zeros(n, 4, dtype=torch.bool)}


def test_head_dropout_blanks_only_the_head_at_about_the_rate():
    policy = _policy(head_dropout=0.5)
    batch = _batch()
    out = policy._augment(batch)
    dropped = (out[HEAD].flatten(1) == 0).all(dim=1)
    assert 0.35 < dropped.float().mean() < 0.65
    assert torch.equal(out[HEAD][~dropped], batch[HEAD][~dropped])
    assert torch.equal(out[WRIST], batch[WRIST])


def test_random_shift_moves_content_but_keeps_shape():
    images = torch.arange(2 * 3 * 24 * 32, dtype=torch.float32).view(2, 3, 24, 32)
    torch.manual_seed(1)
    shifted = random_shift(images, 4)
    assert shifted.shape == images.shape
    assert not torch.equal(shifted, images)


def test_nothing_is_augmented_outside_training():
    policy = _policy(head_dropout=0.9, shift_pad=4)
    batch = _batch(8)
    seen = {}
    original = policy.model.forward

    def spy(b):
        seen["head"] = b[HEAD] if HEAD in b else b["observation.images"][0]
        return original(b)

    policy.model.forward = spy
    policy.eval()
    policy.forward(batch)
    assert torch.equal(seen["head"], batch[HEAD])
    policy.train()
    policy.forward(batch)
    assert not torch.equal(seen["head"], batch[HEAD])


def test_a_misnamed_dropout_camera_is_refused():
    policy = _policy(head_dropout=0.5, dropout_cameras=["observation.images.top"])
    with pytest.raises(KeyError, match="top"):
        policy._augment(_batch(4))
