"""The two controls of the Phase 2 screen, on the same time-row items as TimeRowPFN.

- **A0:** the existing `KeyframePFN` checkpoints (P1, P2), unchanged.
- **A1:** the `KeyframePFN` encoder with the paired head of `time_row_model.PairedHead`.
  It separates the target change from the encoder change.

Both read v1's keyframe-table format, so `to_keyframe_batch` converts a v1-grammar
time-row item. The support's keyframe rows (integer row time) become the context tables.
A keyframe row's own event, acting gripper and acted object are the labels of the
*previous* keyframe row, which is exact because the previous row's next keyframe is this
one. The query is identical. Interpolated rows have no counterpart in v1, so the controls
never see them, and that difference is part of what the screen measures.
"""

from __future__ import annotations

import numpy as np
import torch

from . import time_row_schema as S
from .generator import K_MAX, N_MAX, S_MAX
from .model import KeyframePFN
from .schema import GRIPPER_TYPES, INITIAL, N_FEATURES
from .time_row_model import LABEL_CELLS, PairedHead, _gripper_columns, _take2


def to_keyframe_batch(inputs: dict, targets: dict) -> dict:
    """A batch of v1-grammar time-row items -> KeyframePFN's numpy batch format."""
    B = len(inputs["q_feats"])
    out = {
        "ctx_feats": np.zeros((B, K_MAX, S_MAX, N_MAX, N_FEATURES), np.float32),
        "ctx_types": inputs["s_types"].astype(np.int8),
        "ctx_rows": np.zeros((B, K_MAX, S_MAX), bool),
        "ctx_event": np.zeros((B, K_MAX, S_MAX), np.int8),
        "ctx_gripper": np.full((B, K_MAX, S_MAX), -1, np.int8),
        "ctx_flag": np.zeros((B, K_MAX, S_MAX, N_MAX), bool),
        "demo_valid": inputs["demo_valid"].copy(),
        "q_feats": inputs["q_feats"][..., :N_FEATURES].astype(np.float32),
        "q_types": inputs["q_types"].astype(np.int8),
        "lang_kind": inputs["lang_kind"], "lang_index": inputs["lang_index"],
        # v1's own target fields, for `model.predict` (inference only)
        "t_obj": np.asarray(targets["g_object"]).clip(0),
        "t_grip": np.asarray(targets["y_participation"]).argmax(-1),
        "t_event": np.asarray(targets["g_event"]),
    }
    if inputs["s_feats"].shape[-1] != N_FEATURES:
        raise ValueError("the v1 bridge takes v1-grammar items only")
    time, valid = inputs["s_row_time"], inputs["s_row_valid"]
    for b in range(B):
        for d in range(K_MAX):
            if not inputs["demo_valid"][b, d]:
                continue
            rows = [r for r in np.flatnonzero(valid[b, d])
                    if abs(time[b, d, r] - round(float(time[b, d, r]))) < 1e-6]
            types = inputs["s_types"][b, d]
            for k, r in enumerate(rows):
                out["ctx_feats"][b, d, k] = inputs["s_feats"][b, d, r]
                out["ctx_rows"][b, d, k] = True
                if k == 0:
                    out["ctx_event"][b, d, k] = INITIAL
                    continue
                prev = rows[k - 1]
                arm = int(inputs["s_y_participation"][b, d, prev].argmax())
                out["ctx_event"][b, d, k] = inputs["s_g_event"][b, d, prev]
                out["ctx_gripper"][b, d, k] = arm
                obj = inputs["s_g_object"][b, d, prev]
                if obj >= 0:
                    out["ctx_flag"][b, d, k, obj] = True
                out["ctx_flag"][b, d, k, int(np.flatnonzero(types == GRIPPER_TYPES[arm])[0])] = True
    return out


class KeyframePairedPFN(KeyframePFN):
    """A1: KeyframePFN's encoder, the paired head on its query cells.

    The label cells are the query's current-row left and right gripper cells and its
    event cell."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.config["paired"] = True
        self.paired = PairedHead(self.kind.shape[-1], len(S.ARM_EVENTS), len(S.GROUP_EVENTS),
                                 rot=False)

    def forward_paired(self, b: dict, teacher: bool | None = None) -> dict:
        teacher = self.training if teacher is None else teacher
        keys, key_ok = self.encode_context(b)
        x, _ = self.query_cells(b, keys, key_ok)
        cur = x[:, 1]
        n = b["q_types"].shape[1]
        ents, evc = cur[:, :n], cur[:, n]
        cols = _gripper_columns(b["q_types"].long())
        labels = torch.cat([_take2(ents, cols), evc[:, None]], 1)
        assert labels.shape[1] == LABEL_CELLS
        tgt = b.get("t_g_object") if teacher else None
        return self.paired(labels, ents, b["q_types"].long(), b["q_feats"][:, 1], tgt)


def a0_paired_predictions(pred: dict) -> dict:
    """`model.predict` output (metric) -> the paired fields TimeRowPFN reports.

    A0 names one acting arm, so the other arm's event is HOLD and its position is not
    predicted (NaN)."""
    B = len(pred["pointer"])
    if "gripper" not in pred:  # a goal-only prior: object and goal, nothing per arm
        return {"group_pointer": pred["pointer"].argmax(-1), "goal": pred["final"],
                "goal_only": np.ones(B, bool)}
    arm = pred["gripper"].argmax(-1)
    event = pred["event"].argmax(-1) + 1
    pos = np.full((B, 2, 3), np.nan, np.float32)
    pos[np.arange(B), arm] = pred["grip_pos"]
    arm_event = np.zeros((B, 2), np.int64)
    arm_event[np.arange(B), arm] = event
    participation = np.zeros((B, 2), np.float32)
    participation[np.arange(B), arm] = 1
    return {"group_pointer": pred["pointer"].argmax(-1), "group_event": event,
            "participation": participation, "arm_event": arm_event, "pos": pos,
            "goal": pred["final"]}
