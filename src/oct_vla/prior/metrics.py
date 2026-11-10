"""Scoring next-keyframe predictions, with Wilson intervals on every accuracy.

Predictions are dicts as returned by `KeyframePFN.forward` (after `to_metric`) or by
`baselines`, as numpy arrays. Positions are in metres.

- `keyframe` counts as correct when the pointer, gripper and event are all right *and* the
  object lands within `KEYFRAME_TOLERANCE` of its target.
- `contrast` is GEMIL's "both rows" accuracy over conflict pairs: the pair counts only if
  *both* items are right. A context-blind model can get at most one of the two, because
  the pair shares its query state.
"""

from __future__ import annotations

import math

import numpy as np

KEYFRAME_TOLERANCE = 0.03


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def correctness(pred: dict, b: dict) -> dict[str, np.ndarray]:
    """Per-item booleans and errors."""
    out = {
        "pointer": pred["pointer"].argmax(-1) == b["t_obj"],
        "final_err": np.linalg.norm(pred["final"][:, :3] - b["t_final"][:, :3], axis=-1),
    }
    if "gripper" in pred:
        out["gripper"] = pred["gripper"].argmax(-1) == b["t_grip"]
        out["event"] = pred["event"].argmax(-1) + 1 == b["t_event"]
        out["obj_err"] = np.linalg.norm(pred["obj"][:, :3] - b["t_obj_pose"][:, :3], axis=-1)
        out["grip_err"] = np.linalg.norm(pred["grip_pos"] - b["t_grip_pos"], axis=-1)
        out["keyframe"] = (
            out["pointer"] & out["gripper"] & out["event"] & (out["obj_err"] < KEYFRAME_TOLERANCE)
        )
    return out


def summarise(pred: dict, b: dict, mask: np.ndarray | None = None) -> dict:
    """Accuracies with Wilson CIs, median errors (cm), and contrast accuracy over the pairs
    that lie fully inside `mask`."""
    c = correctness(pred, b)
    mask = np.ones(len(b["t_obj"]), bool) if mask is None else mask
    res: dict = {"n": int(mask.sum())}
    for key in ("pointer", "gripper", "event", "keyframe"):
        if key in c:
            k, n = int(c[key][mask].sum()), int(mask.sum())
            res[key] = {"acc": k / n if n else float("nan"), "ci": wilson(k, n)}
    for key in ("obj_err", "grip_err", "final_err"):
        if key in c and mask.any():
            res[key + "_cm"] = float(np.median(c[key][mask]) * 100)
    pair = b["pair"]
    first = np.flatnonzero((pair > np.arange(len(pair))) & mask)
    first = first[mask[pair[first]]]
    # Pointer contrast only over pairs whose target object differs: a pair that differs
    # in destination or arm shares its pointer, and a context-blind model gets it free.
    split = {"pointer": first[b["t_obj"][first] != b["t_obj"][pair[first]]], "keyframe": first}
    for key in ("pointer", "keyframe"):
        sel = split[key]
        if key in c and len(sel):
            both = c[key][sel] & c[key][pair[sel]]
            res[f"contrast_{key}"] = {
                "acc": float(both.mean()),
                "ci": wilson(int(both.sum()), len(sel)),
                "pairs": len(sel),
            }
    return res
