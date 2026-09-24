"""ACT with the training-time augmentations of `AugACTConfig`."""

import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from torch import Tensor
from torch.nn import functional as F

from .configuration_aug_act import AugACTConfig


def random_shift(images: Tensor, pad: int) -> Tensor:
    """Shift each image in the batch by up to `pad` pixels (replicate-padded),
    independently per sample, as DrQ does."""
    batch, _, height, width = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    dx = torch.randint(0, 2 * pad + 1, (batch,), device=images.device)
    dy = torch.randint(0, 2 * pad + 1, (batch,), device=images.device)
    rows = torch.arange(height, device=images.device)[None, :] + dy[:, None]
    cols = torch.arange(width, device=images.device)[None, :] + dx[:, None]
    index = torch.arange(batch, device=images.device)[:, None, None]
    return padded.permute(0, 2, 3, 1)[index, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)


class AugACTPolicy(ACTPolicy):
    config_class = AugACTConfig
    name = "aug_act"

    def _augment(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        config = self.config
        batch = dict(batch)
        if config.head_dropout > 0:
            for key in config.dropout_cameras:
                if key not in batch:
                    raise KeyError(f"dropout camera {key!r} is not an input; have "
                                   f"{sorted(k for k in batch if k.startswith('observation.images'))}")
                images = batch[key]
                drop = torch.rand(images.shape[0], device=images.device) < config.head_dropout
                batch[key] = images.masked_fill(drop[:, None, None, None], 0.0)
        if config.shift_pad > 0:
            for key in config.image_features:
                batch[key] = random_shift(batch[key], config.shift_pad)
        return batch

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.training:
            batch = self._augment(batch)
        return super().forward(batch)
