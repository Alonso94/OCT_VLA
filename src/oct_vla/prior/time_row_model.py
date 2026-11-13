"""TimeRowPFN: TabPFN-style in-context prediction of paired next-event gripper targets.

Phase 2 of `paper/5_time_row_pfn_design.md` (§9). `KeyframePFN` (`model.py`) is untouched.

**Rows and cells.** A row (a support time step, or a query row) has:
- N **entity cells**: the v1/v2 entity features, typed;
- three **label cells:** left target, right target, group (event, object, goal). They
  hold the row's labels on support rows, and a learned mask token on query rows;
- one **time cell:** within-demo time on support rows, a learned token on query rows.

**Each block** (pre-norm attention, v1's `Attention`):
1. **Within-row attention** over all cells of a row.
2. **Down-column attention within a demo:** each column attends down its own track,
   over the demo's rows. The query's rows (initial, current) are a demo of their own.
3. **Across-row attention for the aligned columns:** left gripper, right gripper, the
   three label cells and time. Each row attends to every **support** row of every demo
   (TabPFN: training rows are the keys). Object columns are not aligned across demos,
   so they reach other demos only through the cells of their own row.
4. A feed-forward layer.

**Readout** (`PairedHead`, shared with the A1 control in `time_row_v1_bridge.py`), from
the query's current row:
- per arm: participation, event, object pointer, gripper position (a residual from the
  arm's current position), aperture, and for v2 a 6D rotation (a residual from the
  current rotation, trained without normalisation, so no antipodal saddle);
- group: event, object pointer, goal pose (`model._pose_rotated`, teacher-forced on the
  true object in training, argmax at inference).

Positions are in v1's normalised units (`model.normalise`); `to_metric` converts back.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import time_row_schema as S
from .model import NEG, POS_CENTRE, POS_SCALE, Attention, _mlp, _pose_rotated, _take, normalise
from .schema import LEFT_GRIPPER, MOVABLE, N_FEATURES, N_TYPES, RIGHT_GRIPPER

LABEL_CELLS = 3  # left, right, group
GRIPPER_TYPES = (LEFT_GRIPPER, RIGHT_GRIPPER)


def _time_features(t: torch.Tensor, width: int) -> torch.Tensor:
    """Sinusoidal features of within-demo time (segment index + alpha)."""
    half = width // 2
    freq = torch.exp(torch.arange(half, device=t.device) * (-math.log(100.0) / half))
    ang = t[..., None] * freq
    return torch.cat([ang.sin(), ang.cos()], -1)


def _gripper_columns(types: torch.Tensor) -> torch.Tensor:
    """types [..., N] -> [..., 2] column of the left and right gripper (each must exist)."""
    cols = torch.stack([(types == t).float().argmax(-1) for t in GRIPPER_TYPES], -1)
    present = torch.stack([(types == t).any(-1) for t in GRIPPER_TYPES], -1)
    return torch.where(present, cols, torch.zeros_like(cols))


class LabelEncoder(nn.Module):
    """Support-row labels -> the three label cells, each also seeing its pointed object."""

    def __init__(self, width: int, n_arm_events: int, n_group_events: int, rot: bool):
        super().__init__()
        self.rot = rot
        arm_in = 3 + 1 + 1 + 1 + (S.ROT_DIM if rot else 0)
        self.arm = _mlp(arm_in, width, width)
        self.arm_event = nn.Embedding(n_arm_events, width)
        self.side = nn.Embedding(2, width)
        self.group_event = nn.Embedding(n_group_events, width)
        self.goal = _mlp(S.POSE_DIM, width, width)
        self.pointed = nn.Linear(width, width)

    def forward(self, lab: dict, entities: torch.Tensor) -> torch.Tensor:
        """lab: support label tensors [..., (2,) ...]; entities [..., N, W] -> [..., 3, W]."""
        pos = (lab["y_pos"] - lab["y_pos"].new_tensor(POS_CENTRE)) / POS_SCALE
        parts = [pos, lab["y_aperture"][..., None], lab["y_participation"][..., None],
                 lab["y_engaged"][..., None]]
        if self.rot:
            parts.append(lab["y_rot"])
        arm = self.arm(torch.cat(parts, -1)) + self.arm_event(lab["y_event"].long())
        arm = arm + self.side.weight
        ptr = lab["y_object"].long()
        arm = arm + self.pointed(_gather_cells(entities, ptr))
        goal = lab["g_goal"].clone()
        goal[..., :3] = (goal[..., :3] - goal.new_tensor(POS_CENTRE)) / POS_SCALE
        group = self.goal(goal) + self.group_event(lab["g_event"].long())
        group = group + self.pointed(_gather_cells(entities, lab["g_object"].long()))
        return torch.cat([arm, group[..., None, :]], -2)


def _gather_cells(cells: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """cells [..., N, W]; index [...] (one pointer per row) or [..., k] (k pointers per row),
    where `...` is cells' leading shape; −1 = none -> zeros. Returns [..., W] or [..., k, W]."""
    single = index.dim() == cells.dim() - 2
    idx = index[..., None] if single else index
    out = cells.gather(-2, idx.clamp_min(0)[..., None].expand(*idx.shape, cells.shape[-1]))
    out = out * (idx >= 0)[..., None]
    return out.squeeze(-2) if single else out


class PairedHead(nn.Module):
    """Paired next-event targets from label cells [B, 3, W] and query entity cells."""

    def __init__(self, width: int, n_arm_events: int, n_group_events: int, rot: bool):
        super().__init__()
        self.rot = rot
        self.side = nn.Embedding(2, width)
        out = 1 + n_arm_events + 3 + 1 + (S.ROT_DIM if rot else 0)
        self.n_arm_events = n_arm_events
        self.arm = _mlp(2 * width, width, out)
        self.ptr_q, self.ptr_k = nn.Linear(width, width), nn.Linear(width, width)
        self.group_event = nn.Linear(width, n_group_events)
        self.goal = _mlp(2 * width, width, 5)

    def _pointer(self, query, entities, movable):
        logits = torch.einsum("...w,...nw->...n", self.ptr_q(query), self.ptr_k(entities))
        return logits / math.sqrt(entities.shape[-1])

    def forward(self, labels, entities, types, q_feats, teacher_object=None) -> dict:
        """labels [B, 3, W]; entities [B, N, W]; types [B, N]; q_feats [B, N, F] (raw)."""
        movable = types == MOVABLE
        cur = normalise(q_feats[..., :N_FEATURES])
        cols = _gripper_columns(types)  # [B, 2]
        gcells = _take2(entities, cols)  # [B, 2, W]
        arms = labels[:, :2] + self.side.weight
        raw = self.arm(torch.cat([arms, gcells], -1))  # [B, 2, out]
        e = self.n_arm_events
        grip_cur = _take2(cur, cols)  # [B, 2, F]
        out = {
            "participation": raw[..., 0],
            "arm_event": raw[..., 1:1 + e],
            "pos": grip_cur[..., :3] + raw[..., 1 + e:4 + e],
            "aperture": torch.sigmoid(raw[..., 4 + e]),
            "arm_pointer": self._pointer(arms, entities[:, None], movable[:, None])
            .masked_fill(~movable[:, None], NEG),
        }
        if self.rot:
            rot_cur = _take2(q_feats[..., N_FEATURES:N_FEATURES + S.ROT_DIM], cols)
            out["rot"] = rot_cur + raw[..., 5 + e:5 + e + S.ROT_DIM]
        group = labels[:, 2]
        out["group_event"] = self.group_event(group)
        out["group_pointer"] = self._pointer(group, entities, movable).masked_fill(~movable, NEG)
        obj = teacher_object if teacher_object is not None else out["group_pointer"].argmax(-1)
        obj_cell, obj_cur = _take(entities, obj), _take(cur, obj)
        out["goal"] = _pose_rotated(obj_cur, self.goal(torch.cat([group, obj_cell], -1)))
        return out


def _take2(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B, N, W], idx [B, k] -> [B, k, W]."""
    return x.gather(1, idx[..., None].expand(*idx.shape, x.shape[-1]))


class TimeRowBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.row = Attention(width, heads)
        self.down = Attention(width, heads)
        self.across = Attention(width, heads)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))

    def forward(self, xs, oks, xq, okq, scols, qcols):
        """xs [B, K, T, C, W] support cells, oks [B, K, T, C]; xq [B, Q, C, W], okq [B, Q, C].
        scols [B, K, 2], qcols [B, 2]: gripper columns."""
        B, K, T, C, W = xs.shape
        # 1. within each row
        xs = self.row(xs, xs, oks[..., :, None] & oks[..., None, :])
        xq = self.row(xq, xq, okq[..., :, None] & okq[..., None, :])
        # 2. down each column, within a demo (the query's rows are their own demo)
        st, okst = xs.transpose(2, 3), oks.transpose(2, 3)  # [B, K, C, T, W]
        xs = self.down(st, st, okst[..., :, None] & okst[..., None, :]).transpose(2, 3)
        qt, okqt = xq.transpose(1, 2), okq.transpose(1, 2)
        xq = self.down(qt, qt, okqt[..., :, None] & okqt[..., None, :]).transpose(1, 2)
        # 3. across rows, aligned columns: grippers, labels, time; keys = support rows
        n = C - LABEL_CELLS - 1
        fixed = torch.arange(n, C, device=xs.device)
        s_idx = torch.cat([scols[:, :, None, :].expand(B, K, T, 2),
                           fixed.expand(B, K, T, C - n)], -1)  # [B, K, T, A]
        q_idx = torch.cat([qcols[:, None, :].expand(B, xq.shape[1], 2),
                           fixed.expand(B, xq.shape[1], C - n)], -1)
        A = s_idx.shape[-1]
        sa = xs.gather(3, s_idx[..., None].expand(B, K, T, A, W)).reshape(B, K * T, A, W)
        qa = xq.gather(2, q_idx[..., None].expand(B, xq.shape[1], A, W))
        ok_sa = oks.gather(3, s_idx).reshape(B, K * T, A)
        keys = sa.transpose(1, 2)  # [B, A, R_s, W]
        key_ok = ok_sa.transpose(1, 2)  # [B, A, R_s]
        new_s = self.across(keys, keys, key_ok[..., None, :].expand(-1, -1, K * T, -1))
        new_q = self.across(qa.transpose(1, 2), keys,
                            key_ok[..., None, :].expand(-1, -1, xq.shape[1], -1))
        xs = xs.scatter(3, s_idx[..., None].expand(B, K, T, A, W),
                        new_s.transpose(1, 2).reshape(B, K, T, A, W))
        xq = xq.scatter(2, q_idx[..., None].expand(B, xq.shape[1], A, W), new_q.transpose(1, 2))
        # 4. feed-forward
        return xs + self.ffn(self.ffn_norm(xs)), xq + self.ffn(self.ffn_norm(xq))


class TimeRowPFN(nn.Module):
    def __init__(self, width: int = 128, heads: int = 4, layers: int = 4, grammar: str = "v1",
                 lang_dim: int = 0, query_initial: bool = True):
        super().__init__()
        rot = grammar == "v2"
        arm_events = len(S.ARM_EVENTS_V2 if rot else S.ARM_EVENTS)
        group_events = len(S.GROUP_EVENTS_V2 if rot else S.GROUP_EVENTS)
        n_features = N_FEATURES + (S.ROT_DIM if rot else 0)
        self.config = {"width": width, "heads": heads, "layers": layers, "grammar": grammar,
                       "lang_dim": lang_dim, "query_initial": query_initial,
                       "schema": S.SCHEMA_VERSION_V2 if rot else S.SCHEMA_VERSION}
        self.rot, self.width = rot, width
        self.cell = nn.Sequential(nn.Linear(n_features, width), nn.SiLU(), nn.Linear(width, width))
        self.entity_type = nn.Embedding(N_TYPES, width)
        self.labels = LabelEncoder(width, arm_events, group_events, rot)
        self.mask_labels = nn.Parameter(torch.randn(LABEL_CELLS, width) * 0.02)
        self.time = nn.Linear(width, width)
        self.query_time = nn.Parameter(torch.zeros(width))
        self.kind = nn.Parameter(torch.randn(2, width) * 0.02)  # support, query
        self.lang = nn.Linear(lang_dim, width) if lang_dim else None
        self.blocks = nn.ModuleList(TimeRowBlock(width, heads) for _ in range(layers))
        self.head = PairedHead(width, arm_events, group_events, rot)

    def _entities(self, feats, types):
        f = normalise(feats[..., :N_FEATURES])
        if self.rot:
            f = torch.cat([f, feats[..., N_FEATURES:]], -1)
        return self.cell(f) + self.entity_type(types.clamp_min(0).long())

    def forward(self, b: dict, teacher: bool | None = None) -> dict:
        teacher = self.training if teacher is None else teacher
        B, K, T, N, _ = b["s_feats"].shape
        s_types = b["s_types"][:, :, None, :].expand(B, K, T, N)
        ents = self._entities(b["s_feats"], s_types) + self.kind[0]
        lab = {k: b[f"s_{k}"] for k in ("y_pos", "y_aperture", "y_participation", "y_engaged",
                                        "y_event", "y_object", "g_event", "g_object", "g_goal")}
        if self.rot:
            lab["y_rot"] = b["s_y_rot"]
        labels = self.labels(lab, ents)  # [B, K, T, 3, W]
        time = self.time(_time_features(b["s_row_time"], self.width))[..., None, :]
        xs = torch.cat([ents, labels, time], -2)
        row_ok = b["s_row_valid"] & b["demo_valid"][..., None]
        ent_ok = (s_types >= 0) & row_ok[..., None]
        oks = torch.cat([ent_ok, row_ok[..., None].expand(B, K, T, LABEL_CELLS + 1)], -1)

        q_rows = b["q_feats"] if self.config["query_initial"] else b["q_feats"][:, 1:]
        Q = q_rows.shape[1]
        q_types = b["q_types"][:, None, :].expand(B, Q, N)
        qents = self._entities(q_rows, q_types) + self.kind[1]
        qlab = self.mask_labels.expand(B, Q, LABEL_CELLS, -1)
        qtime = self.query_time.expand(B, Q, 1, -1)
        xq = torch.cat([qents, qlab, qtime], -2)
        okq = torch.cat([q_types >= 0, torch.ones(B, Q, LABEL_CELLS + 1, dtype=torch.bool,
                                                    device=xq.device)], -1)
        if self.lang is not None and "lang" in b:
            on = (b["lang_kind"] > 0).float()[:, None]
            pooled = (self.lang(b["lang"]) * b["lang_mask"][..., None]).sum(1) / \
                b["lang_mask"].sum(1, keepdim=True).clamp_min(1)
            xq = xq + (pooled * on)[:, None, None, :]

        scols = _gripper_columns(b["s_types"].long())
        qcols = _gripper_columns(b["q_types"].long())
        for block in self.blocks:
            xs, xq = block(xs, oks, xq, okq, scols, qcols)
        cur = xq[:, -1]  # the query's current row
        tgt = b.get("t_g_object") if teacher else None
        return self.head(cur[:, N:N + LABEL_CELLS], cur[:, :N], b["q_types"].long(),
                         b["q_feats"][:, -1], tgt)


def paired_loss(out: dict, b: dict, rot: bool) -> tuple[torch.Tensor, dict]:
    """Masked paired-target loss; both arms' poses are supervised (design §8.2)."""
    valid = b["t_target_valid"] > 0.5
    v = valid.float()

    def mean(x, w):
        w = w.float()
        return (x * w).sum() / w.sum().clamp_min(1.0)

    part = b["t_y_participation"]
    terms = {
        "participation": mean(F.binary_cross_entropy_with_logits(
            out["participation"], part, reduction="none").mean(-1), v),
        "arm_event": mean(F.cross_entropy(out["arm_event"].flatten(0, 1),
                                          b["t_y_event"].long().flatten(),
                                          reduction="none").view(-1, 2).mean(-1), v),
    }
    acting = (part > 0.5) & (b["t_y_object"] >= 0) & valid[:, None]
    ptr = F.cross_entropy(out["arm_pointer"].flatten(0, 1),
                          b["t_y_object"].long().clamp_min(0).flatten(), reduction="none")
    terms["arm_pointer"] = mean(ptr, acting.flatten())
    pos_t = (b["t_y_pos"] - b["t_y_pos"].new_tensor(POS_CENTRE)) / POS_SCALE
    terms["pos"] = mean(F.smooth_l1_loss(out["pos"], pos_t, beta=0.05,
                                         reduction="none").mean((-1, -2)), v)
    terms["aperture"] = mean(F.smooth_l1_loss(out["aperture"], b["t_y_aperture"], beta=0.05,
                                              reduction="none").mean(-1), v)
    if rot:
        terms["rot"] = mean(F.smooth_l1_loss(out["rot"], b["t_y_rot"], beta=0.05,
                                             reduction="none").mean((-1, -2)), v)
    terms["group_event"] = mean(F.cross_entropy(out["group_event"], b["t_g_event"].long(),
                                                reduction="none"), v)
    terms["group_pointer"] = mean(F.cross_entropy(out["group_pointer"],
                                                  b["t_g_object"].long().clamp_min(0),
                                                  reduction="none"), v)
    from .model import target_pose

    terms["goal"] = mean(F.smooth_l1_loss(out["goal"], target_pose(b["t_g_goal"]), beta=0.05,
                                          reduction="none").mean(-1), v)
    total = sum(terms.values())
    return total, {k: float(t.detach()) for k, t in terms.items()}


def to_torch_time_row(inputs: dict, targets: dict | None, lang_table, device) -> dict:
    """numpy inputs (+ targets, prefixed `t_`) -> tensors; instruction tokens looked up."""
    b = {k: torch.as_tensor(np.asarray(v)).to(device) for k, v in inputs.items()}
    if targets is not None:
        b.update({f"t_{k}": torch.as_tensor(np.asarray(v)).to(device) for k, v in targets.items()})
    if lang_table is not None:
        tokens, mask = lang_table
        idx = b["lang_index"].clamp_min(0)
        b["lang"], b["lang_mask"] = tokens.to(device)[idx], mask.to(device)[idx]
    for k in list(b):
        if b[k].dtype == torch.float64:
            b[k] = b[k].float()
    return b
