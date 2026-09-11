"""Deterministic object tokens for offline policy conditioning.

The token is derived only from the canonical ``ObjectScene`` and its
``TaskContext``.  It deliberately carries neither an asset identifier nor a
simulator handle, so the exported dataset can be used without RoboTwin.
"""

from __future__ import annotations

from dataclasses import dataclass

from oct_vla.core.objects import ObjectRole, ObjectScene, TaskContext, role_of


@dataclass(frozen=True)
class ObjectTokenSpec:
    """Fixed, documented layout used by the object-conditioned π0.5 arm."""

    max_objects: int = 8

    @property
    def token_dim(self) -> int:
        # workcell position (3), xyzw orientation (4), own-frame size (3),
        # visibility/confidence (2), and derived target/previous/other role (3).
        return 15


DEFAULT_OBJECT_TOKEN_SPEC = ObjectTokenSpec()


def object_tokens(
    scene: ObjectScene,
    context: TaskContext,
    *,
    spec: ObjectTokenSpec = DEFAULT_OBJECT_TOKEN_SPEC,
) -> tuple[tuple[tuple[float, ...], ...], tuple[bool, ...]]:
    """Return fixed-size scene tokens and a validity mask.

    Target and previous-neighbour objects occupy stable leading positions;
    remaining objects are ordered by their episode-local IDs only to make an
    unordered scene deterministic.  More than ``max_objects`` is a data
    contract error, since silent truncation could discard the target.
    """
    if spec.max_objects <= 0:
        raise ValueError("max_objects must be positive")
    order = {ObjectRole.TARGET: 0, ObjectRole.PREVIOUS_NEIGHBOR: 1, ObjectRole.OTHER: 2}
    ordered = sorted(
        scene.objects,
        key=lambda obj: (order[role_of(context, obj.track_id)], obj.track_id),
    )
    if len(ordered) > spec.max_objects:
        raise ValueError(
            f"scene has {len(ordered)} objects but object token capacity is {spec.max_objects}"
        )
    role_vectors = {
        ObjectRole.TARGET: (1.0, 0.0, 0.0),
        ObjectRole.PREVIOUS_NEIGHBOR: (0.0, 1.0, 0.0),
        ObjectRole.OTHER: (0.0, 0.0, 1.0),
    }
    encoded = tuple(
        (
            *obj.pose.position,
            *obj.pose.orientation,
            *obj.size_xyz,
            obj.visibility,
            obj.confidence,
            *role_vectors[role_of(context, obj.track_id)],
        )
        for obj in ordered
    )
    padding = ((0.0,) * spec.token_dim,) * (spec.max_objects - len(encoded))
    return encoded + padding, (True,) * len(encoded) + (False,) * (spec.max_objects - len(encoded))
