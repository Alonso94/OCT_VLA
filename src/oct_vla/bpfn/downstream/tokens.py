"""Pose tokens for stage-3 executors: the PFN's window, or the support's own table.

One token format for both, so the comparison is about *what* the executor reads::

    [ 0: 3]  position (workcell, m)          [13]     time in [0, 1] (h / (H-1), or row)
    [ 3: 7]  unit quaternion xyzw            [14:19]  colour one-hot (unknown, red, blue,
    [ 7:13]  role one-hot: left wrist, right          green, yellow)
             wrist, movable, fixture,
             window left, window right

The executor side (embedding, bank selection, residual actions) lives with the policies
(``oct_vla/policies/conditioning/pose_tokens.py``).
"""

from __future__ import annotations

import numpy as np

from oct_vla.bpfn.schema import FIXTURE

POSE_TOKEN_DIM = 19
ROLES = ("left_wrist", "right_wrist", "movable", "fixture", "window_left", "window_right")
COLOUR_COLS = 5
POSE_KINDS = ("pose_window", "support_table")
POS_CENTRE = (0.0, -0.15, 1.0)
POS_SCALE = 0.3


# --- building tokens (numpy) ---------------------------------------------------------

def _token(pose7, role: str, time: float, colour: int = 0) -> np.ndarray:
    t = np.zeros(POSE_TOKEN_DIM, np.float32)
    t[:7] = pose7
    t[7 + ROLES.index(role)] = 1.0
    t[13] = time
    t[14 + int(colour)] = 1.0
    return t


def window_tokens(window: np.ndarray) -> np.ndarray:
    """``[H, 2, 7]`` -> ``[2H, POSE_TOKEN_DIM]``, ordered ``h * 2 + arm``."""
    H = len(window)
    return np.stack([_token(window[h, a], ("window_left", "window_right")[a],
                            h / max(1, H - 1)) for h in range(H) for a in range(2)])


def support_tokens(episode, rows: int = 32) -> np.ndarray:
    """The direct-demo baseline's set: both wrists at ``rows`` evenly spaced rows, and every
    object's (and fixture's) first and last pose. ``episode`` is a pose-window Episode."""
    T = episode.T
    idx = np.unique(np.linspace(0, T - 1, rows).round().astype(int))
    out = []
    for r in idx:
        frac = r / max(1, T - 1)
        out.append(_token(episode.wrist[r, 0], "left_wrist", frac))
        out.append(_token(episode.wrist[r, 1], "right_wrist", frac))
    for j in range(len(episode.track_ids)):
        role = "fixture" if episode.kind[j] == FIXTURE else "movable"
        for r, frac in ((0, 0.0), (T - 1, 1.0)):
            out.append(_token(episode.objects[r, j], role, frac, episode.colour[j]))
    return np.stack(out)


def bank(entries: list[np.ndarray], rows: int) -> tuple[np.ndarray, np.ndarray]:
    """K token sets, each padded to ``rows``: ``[K * rows, D]`` and mask ``[K * rows]``."""
    toks = np.zeros((len(entries) * rows, POSE_TOKEN_DIM), np.float32)
    toks[:, 6] = 1.0                                   # padding holds a unit quaternion
    mask = np.zeros(len(entries) * rows, np.float32)
    for k, e in enumerate(entries):
        if len(e) > rows:
            raise ValueError(f"a token set of {len(e)} exceeds the {rows}-row bank slot")
        toks[k * rows:k * rows + len(e)] = e
        mask[k * rows:k * rows + len(e)] = 1.0
    return toks, mask
