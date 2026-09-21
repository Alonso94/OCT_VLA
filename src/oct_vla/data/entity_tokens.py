"""Ground-truth entity tokens for object-centric policies.

This module deliberately contains no task context, object IDs, or array-slot
features.  It is the single schema used by export and later online inference.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt

from oct_vla.core.frames import WORKCELL_FRAME, Pose
from oct_vla.core.objects import ObjectScene
from oct_vla.core.state import EEFState

ENTITY_TOKEN_SCHEMA = "oct-vla-entity-tokens-v2"
ENTITY_TOKEN_DIM = 17
ENTITY_TYPES = ("movable", "left_gripper", "right_gripper", "support")


@dataclass(frozen=True)
class EntitySupport:
    """Static support geometry expressed in the workcell frame."""

    pose: Pose
    size_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.pose.frame != WORKCELL_FRAME:
            raise ValueError("EntitySupport pose must be expressed in the workcell frame")
        size = tuple(float(value) for value in self.size_xyz)
        if len(size) != 3 or not all(isfinite(value) and value > 0.0 for value in size):
            raise ValueError("EntitySupport size_xyz must contain three positive values")
        object.__setattr__(self, "size_xyz", size)


def shelf_support_entities(spec) -> tuple[EntitySupport, EntitySupport]:
    """Convert shelf deck geometry into schema-v2 support entities.

    Kept here so collection/export and online inference use exactly the same
    support geometry and pose convention.
    """
    return tuple(
        EntitySupport(
            Pose(region.center_xyz, (0.0, 0.0, 0.0, 1.0), WORKCELL_FRAME),
            tuple(2.0 * value for value in region.half_extent_xyz),
        )
        for region in (spec.lower_shelf, spec.upper_shelf)
    )  # type: ignore[return-value]


def _rotation6(pose: Pose) -> tuple[float, float, float, float, float, float]:
    """First two columns of the pose's quaternion rotation matrix (xyzw)."""
    x, y, z, w = pose.orientation
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    # Column-major preserves the common 6D rotation convention [R[:, 0], R[:, 1]].
    return (
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy + wz),
        2.0 * (xz - wy),
        2.0 * (xy - wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz + wx),
    )


def _token(pose: Pose, size_xyz: Sequence[float], aperture: float, kind: str) -> tuple[float, ...]:
    try:
        type_index = ENTITY_TYPES.index(kind)
    except ValueError as error:
        raise ValueError(f"unknown entity type {kind!r}") from error
    size = tuple(float(value) for value in size_xyz)
    aperture = float(aperture)
    if len(size) != 3 or not all(isfinite(value) for value in size):
        raise ValueError("entity size_xyz must contain three finite values")
    if not isfinite(aperture):
        raise ValueError("entity aperture must be finite")
    return (
        *pose.position,
        *_rotation6(pose),
        *size,
        aperture,
        *(1.0 if index == type_index else 0.0 for index in range(len(ENTITY_TYPES))),
    )


def _geometry_key(token: tuple[float, ...]) -> tuple[float, ...]:
    """A deterministic geometric order, independent of IDs and input order."""
    return token[:13]


def build_entity_tokens(
    scene: ObjectScene,
    eef: EEFState,
    supports: Sequence[EntitySupport] = (),
    max_entities: int = 16,
) -> tuple[tuple[tuple[float, ...], ...], tuple[bool, ...]]:
    """Build fixed-capacity schema-v2 tokens and a separate validity mask.

    Movable objects and supports are canonically ordered only by their current
    geometry.  No object identity, task role, or input-array position enters a
    token.  Left/right grippers have distinct fixed *type* labels because the
    embodiment makes that distinction observable and actionable.
    """
    if max_entities <= 0:
        raise ValueError("max_entities must be positive")
    movable = sorted(
        (_token(item.pose, item.size_xyz, 0.0, "movable") for item in scene.objects),
        key=_geometry_key,
    )
    grippers = (
        _token(eef.left.pose, (0.0, 0.0, 0.0), eef.left.gripper, "left_gripper"),
        _token(eef.right.pose, (0.0, 0.0, 0.0), eef.right.gripper, "right_gripper"),
    )
    support_tokens = sorted(
        (_token(item.pose, item.size_xyz, 0.0, "support") for item in supports), key=_geometry_key
    )
    tokens = tuple((*movable, *grippers, *support_tokens))
    if len(tokens) > max_entities:
        raise ValueError(
            f"scene has {len(tokens)} entities but entity token capacity is {max_entities}"
        )
    padding = ((0.0,) * ENTITY_TOKEN_DIM,) * (max_entities - len(tokens))
    return tokens + padding, (True,) * len(tokens) + (False,) * len(padding)


@dataclass(frozen=True)
class EntityTokenNormalizer:
    """Train-split statistics for schema v2; only position and size are scaled."""

    position_mean: tuple[float, float, float]
    position_std: tuple[float, float, float]
    size_mean: tuple[float, float, float]
    size_std: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in ("position_mean", "position_std", "size_mean", "size_std"):
            value = tuple(float(item) for item in getattr(self, name))
            if len(value) != 3 or not all(isfinite(item) for item in value):
                raise ValueError(f"{name} must contain three finite values")
            object.__setattr__(self, name, value)
        if not all(value > 0.0 for value in (*self.position_std, *self.size_std)):
            raise ValueError("entity normalization standard deviations must be finite and positive")

    @classmethod
    def fit(
        cls, batches: Iterable[tuple[Sequence[Sequence[float]], Sequence[bool]]]
    ) -> EntityTokenNormalizer:
        values = [[], [], [], [], [], []]
        for tokens, mask in batches:
            if len(tokens) != len(mask):
                raise ValueError("tokens and entity mask must have the same length")
            for token, valid in zip(tokens, mask, strict=True):
                if valid:
                    if len(token) != ENTITY_TOKEN_DIM:
                        raise ValueError(f"expected {ENTITY_TOKEN_DIM}-d entity token")
                    if not all(isfinite(float(value)) for value in token):
                        raise ValueError("valid entity tokens must contain only finite values")
                    for index, field in enumerate((0, 1, 2, 9, 10, 11)):
                        values[index].append(float(token[field]))
        if not values[0]:
            raise ValueError("cannot fit entity normalization without valid training entities")

        def stats(column: list[float]) -> tuple[float, float]:
            mean = sum(column) / len(column)
            variance = sum((value - mean) ** 2 for value in column) / len(column)
            return mean, max(sqrt(variance), 1e-8)

        computed = tuple(stats(column) for column in values)
        return cls(
            tuple(item[0] for item in computed[:3]),
            tuple(item[1] for item in computed[:3]),
            tuple(item[0] for item in computed[3:]),
            tuple(item[1] for item in computed[3:]),
        )

    def normalize(
        self, tokens: Sequence[Sequence[float]], mask: Sequence[bool]
    ) -> tuple[tuple[float, ...], ...]:
        if len(tokens) != len(mask):
            raise ValueError("tokens and entity mask must have the same length")
        normalized = []
        for token, valid in zip(tokens, mask, strict=True):
            if len(token) != ENTITY_TOKEN_DIM:
                raise ValueError(f"expected {ENTITY_TOKEN_DIM}-d entity token")
            if not valid:
                normalized.append((0.0,) * ENTITY_TOKEN_DIM)
                continue
            row = list(map(float, token))
            if not all(isfinite(value) for value in row):
                raise ValueError("valid entity tokens must contain only finite values")
            for offset in range(3):
                row[offset] = (row[offset] - self.position_mean[offset]) / self.position_std[offset]
                row[9 + offset] = (row[9 + offset] - self.size_mean[offset]) / self.size_std[offset]
            normalized.append(tuple(row))
        return tuple(normalized)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ENTITY_TOKEN_SCHEMA,
            "normalized_fields": ["position", "size"],
            "position_mean": list(self.position_mean),
            "position_std": list(self.position_std),
            "size_mean": list(self.size_mean),
            "size_std": list(self.size_std),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> EntityTokenNormalizer:
        if payload.get("schema") != ENTITY_TOKEN_SCHEMA:
            raise ValueError("entity normalization schema does not match v2")

        def vector(name: str) -> tuple[float, float, float]:
            value = tuple(float(item) for item in payload[name])  # type: ignore[index]
            if len(value) != 3:
                raise ValueError(f"{name} must contain three values")
            return value  # type: ignore[return-value]

        result = cls(
            vector("position_mean"), vector("position_std"), vector("size_mean"), vector("size_std")
        )
        if not all(
            isfinite(value) and value > 0.0 for value in (*result.position_std, *result.size_std)
        ):
            raise ValueError("entity normalization standard deviations must be finite and positive")
        if not all(isfinite(value) for value in (*result.position_mean, *result.size_mean)):
            raise ValueError("entity normalization means must be finite")
        return result


def relative_entity_geometry(
    tokens: Sequence[Sequence[float]], mask: Sequence[bool], reference_index: int
) -> tuple[tuple[float, float, float], ...]:
    """Current-frame position of each valid entity relative to one valid entity."""
    if (
        len(tokens) != len(mask)
        or not 0 <= reference_index < len(tokens)
        or not mask[reference_index]
    ):
        raise ValueError("reference_index must name a valid entity")
    reference = tokens[reference_index][:3]
    return tuple(
        tuple(float(token[axis]) - float(reference[axis]) for axis in range(3))
        if valid
        else (0.0, 0.0, 0.0)
        for token, valid in zip(tokens, mask, strict=True)
    )
