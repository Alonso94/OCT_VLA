"""KeyframePFN: predict the next interaction keyframe from support demos, language and a query.

The architecture is ported from GEMIL's TaskPFN (alfa-d `gemil/pfn/traj_table.py`):
- a demonstration is a table, with rows = keyframes and columns = entities, plus one
  *event cell* per row;
- every block runs:
  1. row attention across a keyframe's cells, with attribute-typed difference edges;
  2. time attention down each column;
  3. query attention: the query's cells attend to every context cell of every demo and to
     the instruction token;
  4. a feed-forward layer.

There is no entity-index embedding, so predictions are equivariant to column order.

Heads read the query's current row:
- `pointer`: which movable entity the next event acts on, as a softmax over movables;
- `gripper`: which gripper acts (left / right);
- `event`: grasp / place / regrasp / release;
- `obj`: the acted object's pose after the event, as a residual from its current pose;
- `grip`: the acting gripper's position (relative to the predicted object pose) and its aperture;
- `final`: where the acted object ends up when it is finally placed.

`head="interaction"` trains every head. `head="goal"` is the goal-only ablation: the same
inputs and encoder, trained on `pointer` and `final` only. Regression heads are
teacher-forced on the true pointer during training and use the argmax at inference.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import (
    GRIPPER_TYPES,
    INITIAL,
    MOVABLE,
    N_EVENTS,
    N_FEATURES,
    N_TYPES,
    POS,
    SIZE,
)

NEG = -1e9
HEADS = ("interaction", "goal")
#: Fixed normalisation (workcell metres -> O(1)); no fitted statistics to leak or drift.
POS_CENTRE = (0.0, -0.15, 0.85)
POS_SCALE = 0.3
SIZE_SCALE = 0.05


def _attend(q, k, v, mask, heads, bias=None, edge_value=None):
    """q [..., Lq, W], k/v [..., Lk, W], mask [..., Lq, Lk] or [..., Lk]; rows with no valid
    key get a zero update (ported from GEMIL)."""
    *lead, lq, w = q.shape
    lk, d = k.shape[-2], w // heads
    qh = q.view(*lead, lq, heads, d).transpose(-2, -3)
    kh = k.view(*lead, lk, heads, d).transpose(-2, -3)
    vh = v.view(*lead, lk, heads, d).transpose(-2, -3)
    logits = qh @ kh.transpose(-1, -2) / math.sqrt(d)
    if bias is not None:
        logits = logits + bias.movedim(-1, -3)
    m = mask if mask.dim() == q.dim() else mask[..., None, :]
    m = m[..., None, :, :]
    attn = logits.masked_fill(~m, NEG).softmax(-1) * m.any(-1, keepdim=True).float()
    out = (attn @ vh).transpose(-2, -3).reshape(*lead, lq, w)
    if edge_value is not None:
        ev = edge_value.view(*lead, lq, lk, heads, d)
        out = out + torch.einsum("...hqk,...qkhd->...qhd", attn, ev).reshape(*lead, lq, w)
    return out


class Attention(nn.Module):
    def __init__(self, width, heads, edges=False):
        super().__init__()
        self.heads = heads
        self.norm_q, self.norm_kv = nn.LayerNorm(width), nn.LayerNorm(width)
        self.q, self.kv, self.out = (
            nn.Linear(width, width),
            nn.Linear(width, 2 * width),
            nn.Linear(width, width),
        )
        if edges:
            self.edge_bias, self.edge_value = nn.Linear(width, heads), nn.Linear(width, width)

    def forward(self, x, ctx, mask, edges=None):
        k, v = self.kv(self.norm_kv(ctx)).chunk(2, -1)
        bias = value = None
        if edges is not None:
            bias, value = self.edge_bias(edges), self.edge_value(edges)
        return x + self.out(_attend(self.q(self.norm_q(x)), k, v, mask, self.heads, bias, value))


class DifferenceEdges(nn.Module):
    """Pairwise e_ij = sum_a phi_a(Delta_a(i, j)) over position, yaw, size and aperture.

    Only differences enter, so the edges are translation invariant (GEMIL `edges.py`,
    extended to 3-D and to the gripper aperture)."""

    DIMS = {"position": 4, "yaw": 2, "size": 3, "aperture": 1}

    def __init__(self, width):
        super().__init__()
        self.phi = nn.ModuleDict(
            {
                a: nn.Sequential(nn.Linear(d, width), nn.SiLU(), nn.Linear(width, width))
                for a, d in self.DIMS.items()
            }
        )

    def forward(self, f):
        """f [..., N, F] normalised features -> [..., N, N, W]."""

        def diff(x):
            return x[..., None, :, :] - x[..., :, None, :]

        dp = diff(f[..., POS])
        s, c = f[..., 3], f[..., 4]
        rel = torch.stack(
            [
                s[..., None, :] * c[..., :, None] - c[..., None, :] * s[..., :, None],
                c[..., None, :] * c[..., :, None] + s[..., None, :] * s[..., :, None],
            ],
            -1,
        )
        d = {
            "position": torch.cat([dp, dp.pow(2).sum(-1, keepdim=True)], -1),
            "yaw": rel,
            "size": diff(f[..., SIZE]),
            "aperture": diff(f[..., 8:9]),
        }
        return sum(self.phi[a](d[a]) for a in self.DIMS)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.row = Attention(width, heads, edges=True)
        self.time = Attention(width, heads)
        self.query = Attention(width, heads)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width)
        )

    def rows(self, x, ok, edges):
        c = x.shape[-2]
        e = F.pad(
            edges, (0, 0, 0, c - edges.shape[-2], 0, c - edges.shape[-3])
        )  # event cell: no edges
        return self.row(x, x, ok[..., :, None] & ok[..., None, :], e)

    def times(self, x, ok):
        """Attention down each column: x [..., S, C, W], ok [..., S, C]."""
        xt, okt = x.transpose(-2, -3), ok.transpose(-1, -2)
        return self.time(xt, xt, okt[..., :, None] & okt[..., None, :]).transpose(-2, -3)

    def ffn_step(self, x):
        return x + self.ffn(self.ffn_norm(x))


def normalise(feats: torch.Tensor) -> torch.Tensor:
    f = feats.clone()
    f[..., POS] = (f[..., POS] - f.new_tensor(POS_CENTRE)) / POS_SCALE
    f[..., SIZE] = f[..., SIZE] / SIZE_SCALE
    return f


class KeyframePFN(nn.Module):
    def __init__(
        self,
        width: int = 128,
        heads: int = 4,
        layers: int = 4,
        lang_dim: int = 0,
        head: str = "interaction",
        use_context: bool = True,
    ):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {head!r}")
        self.config = {
            "width": width,
            "heads": heads,
            "layers": layers,
            "lang_dim": lang_dim,
            "head": head,
            "use_context": use_context,
        }
        self.head, self.use_context, w = head, use_context, width
        self.cell = nn.Sequential(nn.Linear(N_FEATURES, w), nn.SiLU(), nn.Linear(w, w))
        self.entity_type = nn.Embedding(N_TYPES, w)
        self.flag = nn.Embedding(3, w)  # context: not involved / involved; query: unknown
        self.event = nn.Embedding(N_EVENTS, w)
        self.gripper = nn.Embedding(3, w)  # left, right, none
        self.unknown_event = nn.Parameter(torch.zeros(w))
        self.kind = nn.Parameter(torch.randn(2, w) * 0.1)  # context cell, query cell
        self.edges = DifferenceEdges(w)
        self.blocks = nn.ModuleList(Block(w, heads) for _ in range(layers))
        self.lang = nn.Linear(lang_dim, w) if lang_dim else None
        self.pointer = nn.Linear(w, 1)
        self.final = _mlp(2 * w, w, 5)
        if head == "interaction":
            self.grip_head = nn.Linear(w, 1)
            self.event_head = nn.Linear(w, N_EVENTS - 1)
            self.obj = _mlp(2 * w, w, 5)
            self.grip = _mlp(3 * w, w, 4)

    # ------------------------------------------------------------------ cells
    def _cells(self, feats, types, flag, event_cell, kind):
        """feats [..., N, F], types [..., N] (-1 = pad), flag [..., N] in {0,1,2},
        event_cell [..., W] -> (cells [..., N+1, W], valid [..., N+1], edges [..., N, N, W])."""
        f = normalise(feats)
        valid = types >= 0
        x = self.cell(f) + self.entity_type(types.clamp_min(0)) + self.flag(flag) + self.kind[kind]
        x = torch.cat([x, (event_cell + self.kind[kind])[..., None, :]], -2)
        ok = torch.cat([valid, torch.ones_like(valid[..., :1])], -1)
        return x, ok, self.edges(f)

    def encode_context(self, b: dict):
        """-> per-block keys [B, K*S*(N+1) (+1 lang), W] and their mask."""
        bsz = b["q_types"].shape[0]
        keys_lang, ok_lang = self._lang(b, bsz)
        if not self.use_context:
            return [keys_lang] * len(self.blocks), ok_lang
        types = b["ctx_types"].long()[:, :, None, :].expand(-1, -1, b["ctx_feats"].shape[2], -1)
        ev = self.event(b["ctx_event"].long()) + self.gripper(
            torch.where(b["ctx_gripper"] < 0, 2, b["ctx_gripper"]).long()
        )
        x, ok, edges = self._cells(b["ctx_feats"], types, b["ctx_flag"].long(), ev, 0)
        ok = ok & b["ctx_rows"][..., None] & b["demo_valid"][:, :, None, None]
        keys = []
        for block in self.blocks:
            x = block.rows(x, ok, edges)
            x = block.times(x, ok)
            keys.append(torch.cat([x.reshape(bsz, -1, x.shape[-1]), keys_lang], 1))
            x = block.ffn_step(x)
        return keys, torch.cat([ok.reshape(bsz, -1), ok_lang], 1)

    def _lang(self, b, bsz):
        """Instruction tokens as extra keys [B, L, W], masked out when there is no instruction."""
        w = self.kind.shape[-1]
        if self.lang is None:
            dev = self.kind.device
            return self.kind.new_zeros(bsz, 0, w), torch.zeros(bsz, 0, dtype=torch.bool, device=dev)
        return self.lang(b["lang"]), b["lang_mask"] & (b["lang_kind"] > 0)[:, None]

    def query_cells(self, b: dict, keys, key_ok):
        """The query's two rows (initial, current) -> final cells [B, 2, N+1, W]."""
        bsz, n = b["q_types"].shape
        types = b["q_types"].long()[:, None, :].expand(-1, 2, -1)
        init = (self.event.weight[INITIAL] + self.gripper.weight[2]).expand(bsz, -1)
        ev = torch.stack([init, self.unknown_event.expand(bsz, -1)], 1)
        flag = torch.full_like(types, 2)
        x, ok, edges = self._cells(b["q_feats"], types, flag, ev, 1)
        for block, key in zip(self.blocks, keys, strict=False):
            x = block.rows(x, ok, edges)
            x = block.times(x, ok)
            x = block.query(x.reshape(bsz, -1, x.shape[-1]), key, key_ok).view(bsz, 2, n + 1, -1)
            x = block.ffn_step(x)
        return x, ok

    # ------------------------------------------------------------------ heads
    def forward(self, b: dict, teacher: bool | None = None) -> dict:
        """Logits and regressions for the next keyframe. `teacher` (default: training mode)
        indexes the regression heads with the true pointer and gripper."""
        teacher = self.training if teacher is None else teacher
        keys, key_ok = self.encode_context(b)
        x, _ = self.query_cells(b, keys, key_ok)
        cur = x[:, 1]  # current row [B, N+1, W]
        n = b["q_types"].shape[1]
        ent, evc = cur[:, :n], cur[:, n]
        types = b["q_types"].long()
        movable = types == MOVABLE
        out = {"pointer": self.pointer(ent).squeeze(-1).masked_fill(~movable, NEG)}
        obj_col = b["t_obj"] if teacher else out["pointer"].argmax(-1)
        obj_cell = _take(ent, obj_col)
        cur_obj = _take(normalise(b["q_feats"][:, 1]), obj_col)
        fin = self.final(torch.cat([obj_cell, evc], -1))
        out["final"] = _pose(cur_obj, fin)
        if self.head == "goal":
            return out
        gcols = torch.stack([(types == t).float().argmax(-1) for t in GRIPPER_TYPES], -1)  # [B, 2]
        gcells = torch.stack([_take(ent, gcols[:, i]) for i in range(2)], 1)
        out["gripper"] = self.grip_head(gcells).squeeze(-1)
        out["event"] = self.event_head(evc)
        grip_idx = b["t_grip"] if teacher else out["gripper"].argmax(-1)
        grip_cell = _take(gcells, grip_idx)
        out["obj"] = _pose(cur_obj, self.obj(torch.cat([obj_cell, evc], -1)))
        g = self.grip(torch.cat([grip_cell, obj_cell, evc], -1))
        out["grip_pos"] = out["obj"][:, :3] + g[:, :3]
        out["aperture"] = torch.sigmoid(g[:, 3])
        return out


def _mlp(i, w, o):
    return nn.Sequential(nn.Linear(i, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(), nn.Linear(w, o))


def _take(x, idx):
    """x [B, N, W], idx [B] -> [B, W]."""
    return x.gather(1, idx[:, None, None].expand(-1, 1, x.shape[-1])).squeeze(1)


def _pose(current, raw):
    """Residual position in normalised units, and a unit (sin, cos) yaw -> [B, 5]."""
    pos = current[:, POS] + raw[:, :3]
    yaw = F.normalize(raw[:, 3:5], dim=-1, eps=1e-6)
    return torch.cat([pos, yaw], -1)


def target_pose(pose: torch.Tensor) -> torch.Tensor:
    """A metric target pose [..., 5] (x, y, z, sin, cos) in the heads' normalised units."""
    out = pose.clone()
    out[..., :3] = (pose[..., :3] - pose.new_tensor(POS_CENTRE)) / POS_SCALE
    return out


def loss(model: KeyframePFN, out: dict, b: dict) -> tuple[torch.Tensor, dict]:
    terms = {
        "pointer": F.cross_entropy(out["pointer"], b["t_obj"]),
        "final": F.smooth_l1_loss(out["final"], target_pose(b["t_final"]), beta=0.05),
    }
    if model.head == "interaction":
        terms["gripper"] = F.cross_entropy(out["gripper"], b["t_grip"])
        terms["event"] = F.cross_entropy(out["event"], b["t_event"] - 1)
        terms["obj"] = F.smooth_l1_loss(out["obj"], target_pose(b["t_obj_pose"]), beta=0.05)
        gp = (b["t_grip_pos"] - b["t_grip_pos"].new_tensor(POS_CENTRE)) / POS_SCALE
        terms["grip_pos"] = F.smooth_l1_loss(out["grip_pos"], gp, beta=0.05)
        terms["aperture"] = F.smooth_l1_loss(out["aperture"], b["t_aperture"], beta=0.05)
    total = sum(terms.values())
    return total, {k: float(v.detach()) for k, v in terms.items()}


def to_metric(out: dict) -> dict:
    """Head outputs in normalised units -> numpy, positions in workcell metres."""
    res = {}
    for k, v in out.items():
        v = v.detach().float().cpu().numpy()
        if k in ("obj", "final", "grip_pos"):
            v = v.copy()
            v[..., :3] = v[..., :3] * POS_SCALE + np.asarray(POS_CENTRE)
        res[k] = v
    return res


def to_torch(batch: dict, lang_table, device) -> dict:
    """A numpy batch -> tensors on `device`, with the instruction tokens looked up.

    `lang_table` is (tokens [n, L, D], mask [n, L]) from `data.load_language`, or None.
    `lang_index` = -1 (no instruction) must coincide with `lang_kind` = 0; anything else
    would hand the model the tokens of the wrong string."""
    t = {k: torch.as_tensor(v).to(device) for k, v in batch.items()}
    if not torch.equal(t["lang_index"] < 0, t["lang_kind"] == 0):
        raise ValueError("lang_index and lang_kind disagree about which items have an instruction")
    if lang_table is not None:
        tokens, mask = lang_table
        idx = t["lang_index"].clamp_min(0)
        t["lang"], t["lang_mask"] = tokens.to(device)[idx], mask.to(device)[idx]
    return t


@torch.no_grad()
def predict(
    model: KeyframePFN, batch: dict, lang_table=None, device="cpu", chunk: int = 256
) -> dict:
    """Inference (argmax pointer and gripper) over a numpy batch, in metric units."""
    model.eval()
    n = len(batch["t_obj"])
    parts = []
    for s in range(0, n, chunk):
        sub = {k: v[s : s + chunk] for k, v in batch.items()}
        parts.append(to_metric(model(to_torch(sub, lang_table, device), teacher=False)))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
