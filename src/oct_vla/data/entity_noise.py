"""Perception error on the object entities, for robustness evaluation.

The simulator hands the policy exact object poses; a real robot measures them
(ArUco markers, a pose estimator) with error and sometimes not at all. This
corrupts a built entity set the way such a measurement would, so the same
checkpoint can be scored against increasing error:

* position: isotropic Gaussian noise, in metres;
* rotation: a random rotation of small angle applied to both rot6D columns, so
  the corrupted rotation is still a rotation (independent noise on the six
  numbers would not be);
* dropout: an object goes unobserved -- its row is masked out and zeroed,
  exactly as padding is.

Only *movable* rows are touched. Gripper rows come from proprioception, which a
real robot measures accurately and which the policy also reads as its state;
support rows are fixed scene geometry. Rows are corrupted after the entity set
is built, so the movables keep their order and noise is not confounded with a
reordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Columns of the 17-wide v2 entity token (data/entity_tokens.py).
POSITION = slice(0, 3)
ROTATION_COLUMN_0 = slice(3, 6)
ROTATION_COLUMN_1 = slice(6, 9)
MOVABLE_TYPE = 13


@dataclass(frozen=True)
class EntityNoise:
    position_mm: float = 0.0
    rotation_deg: float = 0.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.position_mm < 0 or self.rotation_deg < 0:
            raise ValueError("noise magnitudes must be non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def is_clean(self) -> bool:
        return self.position_mm == 0 and self.rotation_deg == 0 and self.dropout == 0

    @property
    def tag(self) -> str:
        """A result-directory name for this level, e.g. ``pos5_rot2_drop10``."""
        return (f"pos{self.position_mm:g}_rot{self.rotation_deg:g}"
                f"_drop{round(self.dropout * 100):d}")


def _small_rotation(angle_std: float, rng: np.random.Generator) -> np.ndarray:
    """A rotation about a uniformly random axis by a N(0, angle_std) angle."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.normal(scale=angle_std)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def corrupt(tokens, mask, noise: EntityNoise, rng: np.random.Generator):
    """A corrupted copy of an ``[N, 17]`` entity set and its ``[N]`` mask.

    Inputs are not modified. At zero noise the output equals the input exactly.
    """
    tokens = np.array(tokens, dtype=np.float64, copy=True)
    mask = np.array(mask, dtype=bool, copy=True)
    if noise.is_clean:
        return tokens, mask
    movable = mask & (tokens[:, MOVABLE_TYPE] > 0.5)
    for row in np.flatnonzero(movable):
        if noise.dropout and rng.random() < noise.dropout:
            mask[row] = False
            tokens[row] = 0.0
            continue
        if noise.position_mm:
            tokens[row, POSITION] += rng.normal(scale=noise.position_mm / 1000.0, size=3)
        if noise.rotation_deg:
            rotation = _small_rotation(math.radians(noise.rotation_deg), rng)
            tokens[row, ROTATION_COLUMN_0] = rotation @ tokens[row, ROTATION_COLUMN_0]
            tokens[row, ROTATION_COLUMN_1] = rotation @ tokens[row, ROTATION_COLUMN_1]
    return tokens, mask
