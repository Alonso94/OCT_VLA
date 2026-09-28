"""The object-crop encoder: a small ViT, its predictor, and the exported 32-d head.

Sized for 64x64 crops of a 30-100 px object -- 8x8 patches give an 8x8 token
grid, and 192-d / 6 blocks / 3 heads is ViT-Tiny's width at half its depth,
about 2.7M parameters. The embedding is the mean over patch tokens rather
than a CLS token: LeJEPA regularises the embedding distribution directly, and
a mean pool has no learned token that can drift independently of the patches.

Downstream consumers get `visual_dim` = 32, not the 192-d embedding. That
projection is PCA fitted on frozen training embeddings after training (see
`PCAProjection`), not a learned layer: a learned head trained against nothing
would be arbitrary, while PCA keeps the directions the encoder actually uses
and, whitened, hands the policy a unit-variance vector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

#: Per-channel normalisation of [0, 1] RGB. The scene is bright and low
#: saturation; 0.5/0.25 centres it without fitting statistics to the corpus.
PIXEL_MEAN = 0.5
PIXEL_STD = 0.25


@dataclass(frozen=True)
class EncoderConfig:
    image_size: int = 64
    patch_size: int = 8
    dim: int = 192
    depth: int = 6
    heads: int = 3
    mlp_ratio: float = 4.0
    predictor_hidden: int = 768
    visual_dim: int = 32

    def to_dict(self) -> dict:
        return asdict(self)


def images_to_float(images: torch.Tensor) -> torch.Tensor:
    """(B, H, W, 3) or (B, 3, H, W) uint8 -> (B, 3, H, W) float in [0, 1].

    The one sanctioned entry from stored crops. Requiring uint8 here is the
    guard against the bug this codebase has had before: uint8-range floats
    reaching a model that expected [0, 1], 255x out of range and silent.
    """
    if images.dtype != torch.uint8:
        raise TypeError(f"expected uint8 crops, got {images.dtype}")
    if images.ndim != 4:
        raise ValueError(f"expected a 4-D batch, got {tuple(images.shape)}")
    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    return images.float().div_(255.0)


class ViTEncoder(nn.Module):
    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        config = config or EncoderConfig()
        if config.image_size % config.patch_size:
            raise ValueError("image_size must be a multiple of patch_size")
        self.config = config
        grid = config.image_size // config.patch_size
        self.patch = nn.Conv2d(3, config.dim, config.patch_size, stride=config.patch_size)
        self.position = nn.Parameter(torch.zeros(1, grid * grid, config.dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            config.dim,
            config.heads,
            int(config.dim * config.mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, config.depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) float in [0, 1] -> (B, dim) embedding."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(images.shape)}")
        size = self.config.image_size
        if images.shape[-2:] != (size, size):
            raise ValueError(f"expected {size}x{size} crops, got {tuple(images.shape[-2:])}")
        # One host sync per call, worth it: 0-255 floats would train "fine".
        if float(images.detach().amax()) > 1.0 + 1e-3:
            raise ValueError("encoder input exceeds 1.0: pass uint8 crops through images_to_float")
        x = (images - PIXEL_MEAN) / PIXEL_STD
        tokens = self.patch(x).flatten(2).transpose(1, 2) + self.position
        return self.norm(self.blocks(tokens)).mean(dim=1)


class Predictor(nn.Module):
    """Context embedding -> predicted target embedding, same width."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        config = config or EncoderConfig()
        self.net = nn.Sequential(
            nn.Linear(config.dim, config.predictor_hidden),
            nn.GELU(),
            nn.Linear(config.predictor_hidden, config.dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LeJEPA(nn.Module):
    """Encoder plus predictor; the loss lives in train.py."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        config = config or EncoderConfig()
        self.config = config
        self.encoder = ViTEncoder(config)
        self.predictor = Predictor(config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)


@dataclass(frozen=True)
class PCAProjection:
    """The exported `visual_dim` head: z -> (z - mean) @ components.T [/ sqrt(var)]."""

    mean: np.ndarray  # (D,)
    components: np.ndarray  # (k, D), rows orthonormal
    explained_variance: np.ndarray  # (k,)
    whiten: bool = True

    @classmethod
    def fit(cls, embeddings: np.ndarray, k: int, whiten: bool = True) -> PCAProjection:
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.shape[0] <= k:
            raise ValueError(f"need more than {k} embeddings to fit a {k}-d PCA")
        mean = embeddings.mean(axis=0)
        _, singular, vt = np.linalg.svd(embeddings - mean, full_matrices=False)
        variance = singular[:k] ** 2 / (embeddings.shape[0] - 1)
        if not np.all(variance > 0):
            raise ValueError("PCA found a zero-variance component: the embedding has collapsed")
        return cls(mean, vt[:k], variance, whiten)

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        projected = (np.asarray(embeddings, dtype=np.float64) - self.mean) @ self.components.T
        if self.whiten:
            projected = projected / np.sqrt(self.explained_variance)
        return projected.astype(np.float32)

    def save(self, path) -> None:
        np.savez(
            path,
            mean=self.mean,
            components=self.components,
            explained_variance=self.explained_variance,
            whiten=np.array(self.whiten),
        )

    @classmethod
    def load(cls, path) -> PCAProjection:
        data = np.load(path)
        return cls(
            data["mean"], data["components"], data["explained_variance"], bool(data["whiten"])
        )
