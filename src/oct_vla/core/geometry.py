"""Active rotations with xyzw unit quaternions and rotation vectors in radians."""

from collections.abc import Iterable
from math import atan2, cos, hypot, isfinite, sin

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


def finite_values(values: Iterable[float], length: int) -> tuple[float, ...]:
    """Copy a flat numeric sequence, rejecting wrong shapes and nonfinite values."""
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Expected a flat numeric sequence") from error
    if len(result) != length or not all(isfinite(value) for value in result):
        raise ValueError(f"Expected {length} finite values")
    return result


def vector3(values: Iterable[float]) -> Vector3:
    x, y, z = finite_values(values, 3)
    return x, y, z


def unit_quaternion(values: Iterable[float]) -> Quaternion:
    """Accept unit quaternions within 1e-6 and remove floating-point norm drift.

    Reject malformed input rather than silently repairing non-unit recordings.
    Choose a deterministic sign, including at exactly pi, so q and -q agree.
    """
    x, y, z, w = finite_values(values, 4)
    norm = hypot(x, y, z, w)
    if abs(norm - 1.0) > 1e-6:
        raise ValueError("Quaternion must have unit norm (tolerance 1e-6)")
    sign = next((value for value in (w, x, y, z) if value != 0.0), 1.0)
    scale = (1.0 if sign > 0 else -1.0) / norm
    return x * scale, y * scale, z * scale, w * scale


def multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    """Compose rotations: apply right, then left."""
    x, y, z, w = unit_quaternion(left)
    a, b, c, d = unit_quaternion(right)
    return unit_quaternion(
        (
            w * a + x * d + y * c - z * b,
            w * b - x * c + y * d + z * a,
            w * c + x * b - y * a + z * d,
            w * d - x * a - y * b - z * c,
        )
    )


def inverse(rotation: Quaternion) -> Quaternion:
    x, y, z, w = unit_quaternion(rotation)
    return unit_quaternion((-x, -y, -z, w))


def exp(rotation_vector: Vector3) -> Quaternion:
    """SO(3) exponential; preserves tiny nonzero rotation increments."""
    x, y, z = vector3(rotation_vector)
    angle = hypot(x, y, z)
    if not isfinite(angle):
        raise ValueError("Rotation magnitude overflow")
    if angle < 1e-8:
        scale = 0.5 - angle * angle / 48.0
        return unit_quaternion((x * scale, y * scale, z * scale, cos(angle / 2)))
    sine = sin(angle / 2)
    return unit_quaternion((x / angle * sine, y / angle * sine, z / angle * sine, cos(angle / 2)))


def log(rotation: Quaternion) -> Vector3:
    """Principal SO(3) logarithm, with magnitude in [0, pi]."""
    x, y, z, w = unit_quaternion(rotation)
    norm = hypot(x, y, z)
    scale = 2.0 if norm < 1e-12 else 2 * atan2(norm, w) / norm
    return x * scale, y * scale, z * scale


def rotate(rotation: Quaternion, vector: Vector3) -> Vector3:
    """Rotate a vector; translation is deliberately excluded."""
    x, y, z, w = unit_quaternion(rotation)
    a, b, c = vector3(vector)
    tx, ty, tz = 2 * (y * c - z * b), 2 * (z * a - x * c), 2 * (x * b - y * a)
    return vector3(
        (a + w * tx + y * tz - z * ty, b + w * ty + z * tx - x * tz, c + w * tz + x * ty - y * tx)
    )
