"""Shared vocabulary of the interaction prior: entity features, events, grippers.

The prior reads a scene as a table of entities. Each entity is a feature vector
(position, yaw, size, aperture) and a type, the same four types as entity
schema v2 (`oct_vla.data.entity_tokens.ENTITY_TYPES`), so a real OCT-VLA scene
converts row for row (`from_entity_tokens`).

Grippers carry position and aperture only. Their orientation is set to zero,
because the synthetic generator cannot know the real end effector's quaternion
convention, and a feature it cannot get right is better left out than silently
mismatched.
"""

from __future__ import annotations

import numpy as np

from oct_vla.data.entity_tokens import ENTITY_TOKEN_DIM, ENTITY_TYPES

#: Continuous columns of one entity: x, y, z, sin(yaw), cos(yaw), sx, sy, sz, aperture.
FEATURES = ("x", "y", "z", "sin_yaw", "cos_yaw", "sx", "sy", "sz", "aperture")
N_FEATURES = len(FEATURES)
POS = slice(0, 3)
YAW = slice(3, 5)
SIZE = slice(5, 8)
APERTURE = 8

TYPES = ENTITY_TYPES  # ("movable", "left_gripper", "right_gripper", "support")
MOVABLE, LEFT_GRIPPER, RIGHT_GRIPPER, SUPPORT = range(4)
N_TYPES = len(TYPES)
#: Gripper index (0 = left, 1 = right) -> entity type.
GRIPPER_TYPES = (LEFT_GRIPPER, RIGHT_GRIPPER)

#: Interaction events. Row 0 of every demonstration is "initial".
EVENTS = ("initial", "grasp", "place", "regrasp", "release")
INITIAL, GRASP, PLACE, REGRASP, RELEASE = range(len(EVENTS))
N_EVENTS = len(EVENTS)
#: Events a query can be asked to predict (everything but "initial").
TARGET_EVENTS = (GRASP, PLACE, REGRASP, RELEASE)

#: Normalised aperture of an open gripper (`EEFState.gripper` is in [0, 1]).
OPEN = 1.0
#: Panda's maximum finger opening in metres; a closed aperture is width / this.
MAX_OPENING = 0.08


def yaw_features(yaw: np.ndarray) -> np.ndarray:
    return np.stack([np.sin(yaw), np.cos(yaw)], -1)


def from_entity_tokens(tokens: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Schema-v2 entity tokens [..., N, 17] -> (features [..., N, 9], types [..., N]).

    Invalid rows get type -1. An object's yaw is read from the first rotation
    column (R[:, 0]) projected onto the ground plane. That is exact for an
    upright object and is what the generator produces. Grippers get zero yaw
    (see the module docstring).
    """
    tokens = np.asarray(tokens, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if tokens.shape[-1] != ENTITY_TOKEN_DIM:
        raise ValueError(
            f"expected schema-v2 tokens of width {ENTITY_TOKEN_DIM}, got {tokens.shape[-1]}"
        )
    one_hot = tokens[..., 13:17]
    if not np.allclose(one_hot[mask].sum(-1), 1.0):
        raise ValueError("a valid entity token has no single type")
    types = np.where(mask, one_hot.argmax(-1), -1)
    yaw = np.arctan2(tokens[..., 4], tokens[..., 3])
    feats = np.zeros(tokens.shape[:-1] + (N_FEATURES,))
    feats[..., POS] = tokens[..., 0:3]
    feats[..., YAW] = yaw_features(yaw)
    feats[..., SIZE] = tokens[..., 9:12]
    feats[..., APERTURE] = tokens[..., 12]
    gripper = np.isin(types, GRIPPER_TYPES)
    feats[gripper, 3] = 0.0
    feats[gripper, 4] = 1.0
    feats[~mask] = 0.0
    return feats, types
