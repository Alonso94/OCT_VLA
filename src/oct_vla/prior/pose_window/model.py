"""PoseWindowPFN: one completed support + an observed query prefix -> [H, 2, 7] wrist poses.

Cells. Every pose cell (wrists, objects, fixtures, and the two goal columns) goes through
one shared pose encoder ``e_pose``; a header embedding (kind, colour, category, size), an
observed/age embedding and the row's relative time are added. Each row has ``C`` entity
cells plus two goal cells (left, right): a support's goal cells hold its wrists' next
poses, a query's are a learned mask embedding.

Blocks (per layer):
1. attention within each row, over its cells;
2. attention along each column's history (support rows; query rows separately);
3. query only: its wrist and goal cells read the support's wrist and goal cells (all
   rows) of the same layer. Object information reaches those cells through step 1.

The support path never reads the query, so ``encode_support`` runs it once per reset and
``predict`` reuses its per-layer states (``test_model`` checks cached == uncached).

Decoder. ``2H`` learned (arm, horizon) queries self-attend and read the query's last
row; each emits a translation offset in workcell axes and a 6D rotation increment in
that wrist's frame. Every horizon is anchored at the current measured wrist:
p(t+h) = p(t) + dp, R(t+h) = R(t) dR.

The ``scene_only`` arm sets ``use_support=False``: step 3 is skipped, everything else equal.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from oct_vla.bpfn.schema import CATEGORIES, COLOURS, KINDS, LEFT_WRIST, PAD, RIGHT_WRIST

POS_CENTRE = (0.0, -0.15, 1.0)
POS_SCALE = 0.3
DELTA_SCALE = 0.10      # m: translation offsets are predicted in these units
SIZE_SCALE = 0.1        # m: object extents enter as size / SIZE_SCALE
LOGVOL_CENTRE, LOGVOL_SCALE = -9.0, 1.5   # log(m^3): a 5 cm cube is about -9


def _quat_mul(a, b):
    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)
    return torch.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], -1)


def quat_from_6d(r6: torch.Tensor) -> torch.Tensor:
    """6D (two columns) -> unit xyzw quaternion, differentiable."""
    a, b = r6[..., :3], r6[..., 3:]
    c1 = F.normalize(a, dim=-1)
    c2 = F.normalize(b - (c1 * b).sum(-1, keepdim=True) * c1, dim=-1)
    c3 = torch.cross(c1, c2, dim=-1)
    m = torch.stack([c1, c2, c3], -1)
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    w = 0.5 * torch.sqrt(torch.clamp(1 + tr, min=1e-6))
    x = (m[..., 2, 1] - m[..., 1, 2]) / (4 * w)
    y = (m[..., 0, 2] - m[..., 2, 0]) / (4 * w)
    z = (m[..., 1, 0] - m[..., 0, 1]) / (4 * w)
    return F.normalize(torch.stack([x, y, z, w], -1), dim=-1)


class _Attn(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)

    def forward(self, x, kv=None, key_pad=None):
        h = self.norm(x)
        kv = h if kv is None else kv
        if key_pad is not None and key_pad.all(1).any():
            # A sequence with no valid key (an empty support): attend to its first entry
            # so the result is defined; its outputs are never read (memory stays padded).
            key_pad = key_pad.clone()
            key_pad[key_pad.all(1), 0] = False
        out, _ = self.attn(h, kv, kv, key_padding_mask=key_pad, need_weights=False)
        return x + out


class _Layer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.row = _Attn(width, heads)
        self.col = _Attn(width, heads)
        self.cross = _Attn(width, heads)
        self.cross_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(),
                                nn.Linear(4 * width, width))

    @staticmethod
    def _rows(attn, x, cell_pad):
        B, T, C, W = x.shape
        y = attn(x.reshape(B * T, C, W), key_pad=cell_pad.reshape(B * T, C))
        return y.reshape(B, T, C, W)

    @staticmethod
    def _cols(attn, x, row_pad):
        B, T, C, W = x.shape
        xt = x.permute(0, 2, 1, 3).reshape(B * C, T, W)
        pad = row_pad[:, None, :].expand(B, C, T).reshape(B * C, T)
        y = attn(xt, key_pad=pad)
        return y.reshape(B, C, T, W).permute(0, 2, 1, 3)

    def support(self, x, cell_pad, row_pad):
        x = self._rows(self.row, x, cell_pad)
        x = self._cols(self.col, x, row_pad)
        return x + self.ff(x)

    def query(self, x, cell_pad, row_pad, mem, mem_pad, role_idx):
        x = self._rows(self.row, x, cell_pad)
        x = self._cols(self.col, x, row_pad)
        if mem is not None:
            B, T, C, W = x.shape
            # The query's aligned cells (2 wrists + 2 goals) read the support's.
            cells = torch.gather(x, 2, role_idx[:, None, :, None].expand(B, T, 4, W))
            flat = cells.reshape(B, T * 4, W)
            upd = self.cross(flat, kv=self.cross_norm(mem), key_pad=mem_pad) - flat
            x = x.scatter_add(2, role_idx[:, None, :, None].expand(B, T, 4, W),
                              upd.reshape(B, T, 4, W))
        return x + self.ff(x)


class PoseWindowPFN(nn.Module):
    def __init__(self, width: int = 128, heads: int = 4, layers: int = 4, horizon: int = 16,
                 use_support: bool = True):
        super().__init__()
        self.config = dict(width=width, heads=heads, layers=layers, horizon=horizon,
                           use_support=use_support)
        self.horizon, self.use_support = horizon, use_support
        self.e_pose = nn.Sequential(nn.Linear(7, width), nn.GELU(), nn.Linear(width, width))
        self.kind = nn.Embedding(len(KINDS) + 2, width)        # + goal_left, goal_right
        self.colour = nn.Embedding(len(COLOURS), width)
        self.category = nn.Embedding(len(CATEGORIES), width)
        #: Size enters scaled (extents / SIZE_SCALE) plus a log-volume channel: raw metres
        #: (0.04-0.14) through one linear layer barely separated sizes, and size-order
        #: rules scored 0.00-0.33 paired on dev (recipe 1).
        self.size = nn.Linear(4, width)
        self.obs = nn.Linear(2, width)                          # observed flag, age
        self.time = nn.Linear(16, width)
        self.table = nn.Embedding(2, width)                     # support / query
        self.goal_mask = nn.Parameter(torch.zeros(width))
        #: One always-valid memory token per layer, so a support with every row masked
        #: (the "none" condition) is well defined instead of a NaN softmax.
        self.null_mem = nn.Parameter(torch.zeros(layers, width))
        self.layers = nn.ModuleList(_Layer(width, heads) for _ in range(layers))
        self.dec_query = nn.Parameter(torch.randn(2 * horizon, width) * 0.02)
        dec = nn.TransformerDecoderLayer(width, heads, 4 * width, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.decoder = nn.TransformerDecoder(dec, 2)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 9))
        nn.init.zeros_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)
        with torch.no_grad():   # zero output = identity rotation increment
            self.head[1].bias[3:9] = torch.tensor([1.0, 0, 0, 0, 1.0, 0])

    # --- embedding ---------------------------------------------------------------------
    def _pose(self, pose):
        centre = pose.new_tensor(POS_CENTRE)
        return self.e_pose(torch.cat([(pose[..., :3] - centre) / POS_SCALE, pose[..., 3:]], -1))

    def _time(self, t):
        freqs = t.new_tensor([2.0 ** k for k in range(8)]) * math.pi / 8
        a = t[..., None] * freqs
        return self.time(torch.cat([a.sin(), a.cos()], -1))

    def _cells(self, b, prefix, which):
        pose, obs, age = b[prefix + "pose"], b[prefix + "observed"], b[prefix + "age"]
        B, T, C, _ = pose.shape
        size = b[prefix + "size"]
        logvol = (torch.log(size.prod(-1, keepdim=True).clamp(min=1e-7)) - LOGVOL_CENTRE) / LOGVOL_SCALE
        header = (self.kind(b[prefix + "kind"].long()) + self.colour(b[prefix + "colour"].long())
                  + self.category(b[prefix + "category"].long())
                  + self.size(torch.cat([size / SIZE_SCALE, logvol], -1)))
        ents = self._pose(pose) + header[:, None] \
            + self.obs(torch.stack([obs.float(), age], -1))
        goal_kinds = self.kind(torch.tensor([len(KINDS), len(KINDS) + 1], device=pose.device))
        if which == 0:
            goal = self._pose(b[prefix + "goal"]) * b[prefix + "goal_valid"][..., None] \
                + self.goal_mask * (~b[prefix + "goal_valid"])[..., None]
        else:
            goal = self.goal_mask.expand(B, T, 2, -1)
        goal = goal + goal_kinds
        x = torch.cat([ents, goal], 2) + self._time(b[prefix + "time"])[:, :, None] \
            + self.table.weight[which]
        cell_pad = torch.cat([b[prefix + "kind"] == PAD,
                              torch.zeros(B, 2, dtype=torch.bool, device=pose.device)], 1)
        cell_pad = cell_pad[:, None].expand(B, T, C + 2)
        row_pad = ~b[prefix + "row_valid"]
        return x, cell_pad, row_pad

    @staticmethod
    def _role_idx(kind):
        """Column index of the left wrist, right wrist, goal L, goal R per item."""
        C = kind.shape[1]
        left = (kind == LEFT_WRIST).float().argmax(1)
        right = (kind == RIGHT_WRIST).float().argmax(1)
        if not ((kind == LEFT_WRIST).sum(1) == 1).all() or not ((kind == RIGHT_WRIST).sum(1) == 1).all():
            raise ValueError("every table needs exactly one left and one right wrist column")
        return torch.stack([left, right, torch.full_like(left, C), torch.full_like(left, C + 1)], 1)

    # --- support / query ---------------------------------------------------------------
    def encode_support(self, b: dict) -> dict:
        x, cell_pad, row_pad = self._cells(b, "s_", 0)
        idx = self._role_idx(b["s_kind"])
        B, T, C, W = x.shape
        mems, pads = [], []
        mem_pad = row_pad[:, :, None].expand(B, T, 4).reshape(B, T * 4)
        null_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        for k, layer in enumerate(self.layers):
            x = layer.support(x, cell_pad, row_pad)
            cells = torch.gather(x, 2, idx[:, None, :, None].expand(B, T, 4, W))
            null = self.null_mem[k].expand(B, 1, W)
            mems.append(torch.cat([cells.reshape(B, T * 4, W), null], 1))
            pads.append(torch.cat([mem_pad, null_pad], 1))
        return {"mem": mems, "pad": pads}

    def predict(self, memory: dict | None, b: dict) -> dict:
        x, cell_pad, row_pad = self._cells(b, "q_", 1)
        idx = self._role_idx(b["q_kind"])
        for k, layer in enumerate(self.layers):
            mem = memory["mem"][k] if (memory is not None and self.use_support) else None
            pad = memory["pad"][k] if mem is not None else None
            x = layer.query(x, cell_pad, row_pad, mem, pad, idx)
        last = b["q_row_valid"].long().cumsum(1).argmax(1)              # last valid row
        B, T, C, W = x.shape
        ctx = x[torch.arange(B), last]                                    # [B, C, W]
        q = self.dec_query.expand(B, -1, -1)
        h = self.decoder(q, ctx, memory_key_padding_mask=cell_pad[torch.arange(B), last])
        out = self.head(h).reshape(B, 2, self.horizon, 9).transpose(1, 2)  # [B, H, 2, 9]
        # Anchor: each wrist's measured pose at the last query row.
        wrists = torch.gather(b["q_pose"][torch.arange(B), last], 1,
                              idx[:, :2, None].expand(B, 2, 7))          # [B, 2, 7]
        pos = wrists[:, None, :, :3] + DELTA_SCALE * out[..., :3]
        dq = quat_from_6d(out[..., 3:9])
        rot = _quat_mul(wrists[:, None, :, 3:].expand_as(dq), dq)
        return {"window": torch.cat([pos, F.normalize(rot, dim=-1)], -1),
                "delta": out[..., :3] * DELTA_SCALE}

    def forward(self, b: dict) -> dict:
        memory = self.encode_support(b) if self.use_support else None
        return self.predict(memory, b)


#: Rotation weight. 1 - <q_hat, q>^2 is ~0.005 at 8 degrees against a translation term
#: near 1, so with weight 1 rotation got almost no gradient (stuck at ~8 deg, recipe 1).
LAMBDA_ROT = 10.0


def window_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                length: float = 0.05, lambda_rot: float = LAMBDA_ROT) -> tuple[torch.Tensor, dict]:
    """Huber on translation / ``length`` + lambda (1 - <q_hat, q>^2), mean over valid
    (arm, horizon) pairs; ``valid`` is ``[B, H]`` (both arms share a horizon's validity)."""
    v = valid[..., None].expand(pred.shape[:-1]).float()             # [B, H, 2]
    n = v.sum().clamp(min=1)
    trans = F.huber_loss(pred[..., :3] / length, target[..., :3] / length, reduction="none",
                         delta=1.0).sum(-1)
    dot = (pred[..., 3:] * target[..., 3:]).sum(-1)
    rot = 1 - dot**2
    loss_trans = (trans * v).sum() / n
    loss_rot = (rot * v).sum() / n
    loss = loss_trans + lambda_rot * loss_rot
    err = (pred[..., :3] - target[..., :3]).norm(dim=-1)
    ang = 2 * torch.acos(dot.abs().clamp(0, 1))
    loss_value = loss.detach()
    return loss, {"loss": float(loss_value), "loss_trans": float(loss_trans.detach()),
                  "loss_rot": float(loss_rot.detach()), "trans_mm": float((err * v).sum().detach() / n * 1000),
                  "rot_deg": float((ang * v).sum().detach() / n * 180 / math.pi)}


def to_torch(batch: dict, device="cpu") -> dict:
    out = {}
    for k, v in batch.items():
        t = torch.as_tensor(v)
        out[k] = t.to(device, torch.float32 if t.dtype in (torch.float64, torch.float32) else None)
    return out
