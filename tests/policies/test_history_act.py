"""history_act: the earlier frame must reach the output, at training and at rollout."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402

from oct_vla.policies.history_act import HistoryACTConfig, HistoryACTPolicy  # noqa: E402

CAMERAS = ("observation.images.a", "observation.images.b")


def _policy(**overrides):
    torch.manual_seed(0)
    config = HistoryACTConfig(
        device="cpu",
        input_features={
            **{k: PolicyFeature(FeatureType.VISUAL, (3, 32, 32)) for k in CAMERAS},
            "observation.state": PolicyFeature(FeatureType.STATE, (4,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (4,))},
        chunk_size=6, n_action_steps=2, dim_model=32, n_heads=2, dim_feedforward=64,
        n_encoder_layers=1, n_decoder_layers=1, latent_dim=4, n_vae_encoder_layers=1,
        pretrained_backbone_weights=None, history_stride=2, **overrides,
    )
    return HistoryACTPolicy(config)


def _history(batch=2, steps=2):
    return {
        **{k: torch.rand(batch, steps, 3, 32, 32) for k in CAMERAS},
        "observation.state": torch.randn(batch, steps, 4),
        "action": torch.randn(batch, 6, 4),
        "action_is_pad": torch.zeros(batch, 6, dtype=torch.bool),
    }


def test_the_dataset_is_asked_for_the_history_oldest_first():
    assert _policy().config.observation_delta_indices == [-2, 0]
    assert _policy(history_steps=3).config.observation_delta_indices == [-4, -2, 0]


def test_training_uses_every_frame_and_tells_them_apart():
    policy = _policy(use_vae=False).eval()
    batch = _history()
    base = policy.predict_action_chunk(batch)
    for key in ("observation.state", CAMERAS[0]):
        changed = dict(batch)
        changed[key] = batch[key].clone()
        changed[key][:, 0] += 1.0  # only the *earlier* frame
        assert not torch.allclose(policy.predict_action_chunk(changed), base), key
    # Same content in both frames, swapped: different output, so frame order is seen.
    swapped = dict(batch)
    swapped[CAMERAS[0]] = batch[CAMERAS[0]].flip(1)
    swapped[CAMERAS[1]] = batch[CAMERAS[1]].flip(1)
    swapped["observation.state"] = batch["observation.state"].flip(1)
    assert not torch.allclose(policy.predict_action_chunk(swapped), base)


def test_loss_backpropagates_into_the_frame_embedding():
    policy = _policy().train()
    loss, info = policy.forward(_history())
    loss.backward()
    assert policy.model.history_frame_embed.grad.abs().sum() > 0
    assert "kld_loss" in info


def test_a_single_frame_batch_is_refused_in_training():
    batch = _history()
    batch["observation.state"] = batch["observation.state"][:, -1]
    with pytest.raises(ValueError, match="observation.state"):
        _policy().forward(batch)


def test_rollout_history_matches_the_training_layout():
    """select_action, fed one frame per step, must see what the dataset would
    have given for the same trajectory: frames `stride` apart, and the first
    frame repeated before the episode had any history."""
    policy = _policy(use_vae=False).eval()
    frames = [
        {**{k: torch.rand(1, 3, 32, 32) for k in CAMERAS}, "observation.state": torch.randn(1, 4)}
        for _ in range(5)
    ]
    policy.reset()
    seen = []
    original = policy.predict_action_chunk

    def spy(batch):
        seen.append({k: v.clone() for k, v in batch.items() if k in frames[0]})
        return original(batch)

    policy.predict_action_chunk = spy
    for frame in frames:
        policy.select_action(frame)
    # n_action_steps=2: chunks are predicted at steps 0, 2 and 4.
    assert len(seen) == 3
    for call, step in zip(seen, (0, 2, 4), strict=True):
        earlier = max(step - 2, 0)
        for key in frames[0]:
            assert torch.equal(call[key][:, 0], frames[earlier][key]), (step, key)
            assert torch.equal(call[key][:, 1], frames[step][key]), (step, key)
    policy.reset()
    assert len(policy._history) == 0
