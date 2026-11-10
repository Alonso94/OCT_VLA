"""Evaluation-time edits of a batch's support and instruction, never applied in training.

Support conditions:
    correct      the item's own demonstrations
    none         no demonstrations
    shuffled     another item's demonstrations, from a different program
    conflicting  the conflict partner's demonstrations. The partner shares this query
                 state, so its target is valid here; `partner_targets` returns it.

Language conditions:
    none, ambiguous (a generic instruction), explicit (names the item's own program)

Read the results by regime:
- With an ambiguous or no instruction, only the support identifies the rule, so a
  context-using model must fall under `shuffled` and must follow the partner under
  `conflicting`.
- With an explicit instruction, following the language over a contradicting support is
  the correct behaviour, not a failure. Report "follows language" and "follows support"
  separately.
"""

from __future__ import annotations

import numpy as np

from . import programs as P
from .batches import LANG_AMBIGUOUS, LANG_EXPLICIT, LANG_NONE

CTX_KEYS = (
    "ctx_feats",
    "ctx_types",
    "ctx_rows",
    "ctx_event",
    "ctx_gripper",
    "ctx_flag",
    "demo_valid",
)
TARGET_KEYS = ("t_obj", "t_grip", "t_event", "t_obj_pose", "t_grip_pos", "t_aperture", "t_final")
SUPPORTS = ("correct", "none", "shuffled", "conflicting")
LANGUAGES = ("as_generated", "none", "ambiguous", "explicit")


def _with_support_from(b: dict, src: np.ndarray) -> dict:
    out = dict(b)
    for k in CTX_KEYS:
        out[k] = b[k][src]
    return out


def support(b: dict, condition: str, rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    """(edited batch, mask of the items the condition applies to)."""
    n = len(b["t_obj"])
    if condition == "correct":
        return b, np.ones(n, bool)
    if condition == "none":
        out = dict(b)
        for k in CTX_KEYS:
            out[k] = np.zeros_like(b[k]) if k != "ctx_types" else np.full_like(b[k], -1)
        return out, np.ones(n, bool)
    if condition == "shuffled":
        src = np.arange(n)
        for i in range(n):
            other = np.flatnonzero(b["program"] != b["program"][i])
            if len(other):
                src[i] = other[rng.integers(len(other))]
        return _with_support_from(b, src), src != np.arange(n)
    if condition == "conflicting":
        has = b["pair"] >= 0
        src = np.where(has, b["pair"], np.arange(n))
        return _with_support_from(b, src), has
    raise ValueError(f"unknown support condition {condition!r}")


def partner_targets(b: dict) -> dict:
    """The batch with every conflict item's targets replaced by its partner's (same query).

    Each item's query columns are permuted independently. The partner's target column
    is therefore mapped into this item's column order, by matching the two (identical)
    query states row for row. Copying the raw index would point at another entity."""
    has = b["pair"] >= 0
    src = np.where(has, b["pair"], np.arange(len(has)))
    out = dict(b)
    for k in TARGET_KEYS:
        out[k] = b[k][src]
    for i in np.flatnonzero(has):
        j = src[i]
        row = b["q_feats"][j, :, b["t_obj"][j]]  # partner's target, both rows
        dist = np.abs(b["q_feats"][i] - row[:, None, :]).sum((0, 2))
        dist[b["q_types"][i] != b["q_types"][j][b["t_obj"][j]]] = np.inf
        col = int(np.argmin(dist))
        if dist[col] > 1e-5:
            raise ValueError(f"conflict partners {i} and {j} do not share a query state")
        out["t_obj"][i] = col
    return out


def language(b: dict, condition: str, rng: np.random.Generator) -> dict:
    if condition == "as_generated":
        return b
    out = dict(b)
    n = len(b["t_obj"])
    if condition == "none":
        out["lang_kind"] = np.full(n, LANG_NONE)
        out["lang_index"] = np.full(n, -1)
    elif condition == "ambiguous":
        out["lang_kind"] = np.full(n, LANG_AMBIGUOUS)
        amb = [P.ALL_INSTRUCTIONS.index(s) for s in P.AMBIGUOUS_INSTRUCTIONS]
        out["lang_index"] = np.asarray(amb)[rng.integers(len(amb), size=n)]
    elif condition == "explicit":
        out["lang_kind"] = np.full(n, LANG_EXPLICIT)
        out["lang_index"] = np.array(
            [P.ALL_INSTRUCTIONS.index(P.ALL_PROGRAMS[p].instructions()[0]) for p in b["program"]]
        )
    else:
        raise ValueError(f"unknown language condition {condition!r}")
    return out
