"""Vectorised xyzw quaternions, the convention of ``core/geometry.py`` (active rotations).

``core.geometry`` works on single tuples; the generator needs whole tracks at once.
The tests check these against ``core.geometry`` element by element.
"""

from __future__ import annotations

import numpy as np

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-9):
        raise ValueError("cannot normalise a zero quaternion")
    return q / n


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a * b: apply b, then a."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1, y1, z1, w1 = np.moveaxis(a, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(b, -1, 0)
    return np.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], axis=-1)


def inverse(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    return np.concatenate([-q[..., :3], q[..., 3:]], axis=-1)


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q, v = np.asarray(q, float), np.asarray(v, float)
    u, w = q[..., :3], q[..., 3:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def from_axis_angle(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    half = np.asarray(angle, float)[..., None] / 2
    return np.concatenate([axis * np.sin(half), np.cos(half)], axis=-1)


def to_matrix(q: np.ndarray) -> np.ndarray:
    q = normalize(q)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (Shepperd's method, w >= 0)."""
    m = np.asarray(m, float)
    flat = m.reshape(-1, 3, 3)
    out = np.empty((len(flat), 4))
    for i, r in enumerate(flat):
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            out[i] = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            out[i] = [s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            out[i] = [(r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            out[i] = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4, (r[1, 0] - r[0, 1]) / s]
    out = normalize(out)
    out = np.where(out[:, 3:] < 0, -out, out)
    return out.reshape(m.shape[:-2] + (4,))


def from_6d(r6: np.ndarray) -> np.ndarray:
    """The prior's 6D rotation (first two matrix columns, Gram-Schmidt) -> quaternion."""
    r6 = np.asarray(r6, float)
    a, b = r6[..., :3], r6[..., 3:6]
    c1 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (c1 * b).sum(-1, keepdims=True) * c1
    c2 = b / np.linalg.norm(b, axis=-1, keepdims=True)
    c3 = np.cross(c1, c2)
    return from_matrix(np.stack([c1, c2, c3], axis=-1))


def slerp(q0: np.ndarray, q1: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Shortest-arc interpolation; ``s`` broadcasts over a leading axis."""
    q0, q1 = normalize(q0), normalize(q1)
    s = np.asarray(s, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        return normalize(q0 + s[:, None] * (q1 - q0))
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    return (np.sin((1 - s) * theta)[:, None] * q0 + np.sin(s * theta)[:, None] * q1) / np.sin(theta)


def continuous(track: np.ndarray) -> np.ndarray:
    """Flip signs along axis 0 so consecutive quaternions have a non-negative dot product."""
    out = np.array(track, float)
    for t in range(1, len(out)):
        flip = (out[t] * out[t - 1]).sum(-1) < 0
        out[t][flip] *= -1
    return out


def angle(q_hat: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Sign-invariant angle in radians: 2 acos(clamp(|<q_hat, q_ref>|, 0, 1))."""
    d = np.abs((normalize(q_hat) * normalize(q_ref)).sum(-1))
    return 2 * np.arccos(np.clip(d, 0.0, 1.0))
