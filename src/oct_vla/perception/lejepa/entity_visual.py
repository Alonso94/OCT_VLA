"""Per-entity visual features from a frozen LeJEPA encoder.

Plan Phase E, steps 3-5: every movable entity row gets a `visual_dim`-wide
embedding of its object's crop in the head camera, next to its geometry.

The crop box is built from the **entity token itself** -- position, the two
rot6D columns, size -- rather than from the simulator's object list. So the
training export (dataset tokens + the canonical raw head frame) and live
evaluation (served tokens + the live head frame) go through the same function
on the same inputs, and cannot drift apart. Frames must be raw simulator
frames in both: the dataset's AV1 video is not what evaluation sees.

Rows that are not movable, are padding, or do not project into the image get
zeros -- the same value a zero-initialised projection starts from, so a missing
crop is "no visual evidence", not a made-up one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .crops import (
    CROP_SIZE,
    PinholeCamera,
    default_head_camera,
    projected_bbox,
    resize_square,
    square_crop_box,
)

#: The dataset/batch key the visual features travel under.
ENTITY_VISUAL = "observation.entity_visual"
#: Columns of the 17-wide v2 entity token (data/entity_tokens.py).
_POSITION, _COLUMN_0, _COLUMN_1, _SIZE, _MOVABLE = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), 13


def token_corners(token) -> np.ndarray:
    """The eight world-frame corners (8, 3) of the box an entity token describes."""
    token = np.asarray(token, dtype=np.float64)
    c0, c1 = token[_COLUMN_0], token[_COLUMN_1]
    rotation = np.stack([c0, c1, np.cross(c0, c1)], axis=1)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return (signs * (token[_SIZE] / 2.0)) @ rotation.T + token[_POSITION]


def token_crop(image: np.ndarray, camera: PinholeCamera, token) -> np.ndarray | None:
    """The (64, 64, 3) uint8 crop of a movable entity, or None if it does not project."""
    bbox = projected_bbox(camera, token_corners(token))
    if bbox is None:
        return None
    square = square_crop_box(bbox, camera.width, camera.height)
    if square is None:
        return None
    left, top, side = square
    return resize_square(image[top : top + side, left : left + side], CROP_SIZE)


class EntityVisualEncoder:
    """The frozen encoder and PCA projection of one LeJEPA run."""

    def __init__(self, run_dir: str | Path, device: str = "cpu",
                 camera: PinholeCamera | None = None) -> None:
        from .model import PCAProjection
        from .train import load_encoder

        run_dir = Path(run_dir)
        self.run_dir = run_dir
        self.device = device
        self.model = load_encoder(run_dir / "lejepa.pt", device=device)
        self.pca = PCAProjection.load(run_dir / "pca32.npz")
        self.visual_dim = int(self.pca.components.shape[0])
        self.camera = camera or default_head_camera()

    def __call__(self, images, tokens, mask) -> np.ndarray:
        """``[F, N, visual_dim]`` features for F frames of ``[N, 17]`` entities.

        ``images`` are F raw head frames, (H, W, 3) uint8.
        """
        from .train import embed

        tokens = np.asarray(tokens, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or len(images) != tokens.shape[0]:
            raise ValueError(f"expected F images, [F, N, 17] tokens and [F, N] mask; got "
                             f"{len(images)}, {tokens.shape}, {mask.shape}")
        out = np.zeros((*tokens.shape[:2], self.visual_dim), dtype=np.float32)
        crops, where = [], []
        for frame, image in enumerate(images):
            image = np.asarray(image)
            if image.dtype != np.uint8 or image.shape != (self.camera.height, self.camera.width, 3):
                raise ValueError(f"head frame must be ({self.camera.height}, {self.camera.width}, 3)"
                                 f" uint8, got {image.shape} {image.dtype}")
            for row in np.flatnonzero(mask[frame] & (tokens[frame, :, _MOVABLE] > 0.5)):
                crop = token_crop(image, self.camera, tokens[frame, row])
                if crop is not None:
                    crops.append(crop)
                    where.append((frame, row))
        if crops:
            embedded = self.pca(embed(self.model, np.stack(crops), self.device))
            for (frame, row), vector in zip(where, embedded, strict=True):
                out[frame, row] = vector
        return out
