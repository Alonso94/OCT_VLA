"""TablePFN: the demonstration and the query as tables of pose cells (TabPFN / ConTextTab style).

Cells hold a value and a column identity, nothing else:
- an entity cell: its pose (position normalised around the workcell, xyzw quaternion) plus
  its header (kind, colour, category, size);
- the time column: one cell per row, seconds since the table's first row (each table starts
  at t = 0, whatever its position in the episode), as in time-series PFNs;
- a CLS cell per row, which row attention fills with a summary of the row;
- support only: the two goal columns (each wrist's pose at the next row).
No support/query marker, observation flag or measurement age enters.

Blocks alternate attention across the cells of a row and along each column over time.

1. Demonstration: ``support_layers`` blocks over the support table; its row CLS cells are
   pooled by attention (learned seeds) into ``task_tokens`` task tokens.
2. Query: ``query_layers`` blocks over the query history, reading the task by one of
   - ``cell``: the task tokens are extra cells of every query row (read by row attention);
   - ``film``: each block scales and shifts every query cell by the task;
   - ``cross``: before each block, the query row CLS cells attend to the support's row CLS
     cells and the task tokens (a row-level connection);
   - ``cross_all``: the same, for every query cell rather than the CLS cells only.
3. Output, the next ``horizon`` poses of both wrists as offsets from the current wrist:
   - ``head``: learned (arm, horizon) queries read the last query row's cells;
   - ``rows``: ``horizon`` future rows are appended to the query (time known, every value
     masked) and their wrist cells are read out, filling the missing cells as TabPFN fills
     a test row's target.

The support never reads the query, so its encoding is computed once per support and
shared by its queries (``support_index``).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from oct_vla.bpfn.models.pose_window_pfn import (DELTA_SCALE, LOGVOL_CENTRE, LOGVOL_SCALE,
                                                 POS_CENTRE, POS_SCALE, SIZE_SCALE, _quat_mul,
                                                 quat_from_6d)
from oct_vla.bpfn.data.batches import FACTORS
from oct_vla.bpfn.schema import CATEGORIES, COLOURS, DT, KINDS, LEFT_WRIST, MOVABLE, PAD, RIGHT_WRIST

ARCH = "table"
TASK_READS = ("cell", "film", "cross", "cross_all")
OUTPUTS = ("head", "rows")
FRAMES = ("workcell", "relative")
#: Column kinds beyond the schema's: goal left / right, time, row CLS, task.
GOAL_LEFT, GOAL_RIGHT, TIME, ROW_CLS, TASK = range(len(KINDS), len(KINDS) + 5)
#: Periods (s) of the time features: from a frame to a whole demonstration.
TIME_PERIODS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 90.0)


class _Attention(nn.Module):
    """Pre-norm multi-head attention with a residual; a sequence whose keys are all masked
    attends to its first key, so the result is defined (it is never read)."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)

    def forward(self, x, kv=None, key_pad=None):
        h = self.norm(x)
        kv = h if kv is None else kv
        if key_pad is not None:
            first = torch.zeros_like(key_pad)
            first[:, 0] = True
            key_pad = key_pad & ~(first & key_pad.all(1, keepdim=True))
        out, _ = self.attn(h, kv, kv, key_padding_mask=key_pad, need_weights=False)
        return x + out


class _AxialBlock(nn.Module):
    """Attention across the cells of each row, then along each column over time, then a
    feed-forward layer."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.row = _Attention(width, heads)
        self.col = _Attention(width, heads)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(),
                                nn.Linear(4 * width, width))

    def forward(self, x, cell_pad, row_pad):
        """x [B, T, C, W]; cell_pad [B, C] (a padded column); row_pad [B, T]."""
        B, T, C, W = x.shape
        pad = cell_pad[:, None, :].expand(B, T, C).reshape(B * T, C)
        x = self.row(x.reshape(B * T, C, W), key_pad=pad).reshape(B, T, C, W)
        xt = x.permute(0, 2, 1, 3).reshape(B * C, T, W)
        pad = row_pad[:, None, :].expand(B, C, T).reshape(B * C, T)
        x = self.col(xt, key_pad=pad).reshape(B, C, T, W).permute(0, 2, 1, 3)
        return x + self.ff(x)


class TablePFN(nn.Module):
    """See the module docstring.

    Args:
        width, heads: the model width and attention heads.
        support_layers, query_layers: axial blocks over the demonstration and the query.
        horizon: predicted steps.
        task_read: how the query reads the demonstration (``cell``, ``film``, ``cross``).
        output: ``head`` or ``rows``.
        task_tokens: pooled demonstration tokens.
        use_support: False for the scene-only arm (the task is never read).
        frame: ``workcell`` (positions around a fixed workcell centre) or ``relative`` (each
            table's positions around its own objects' centroid at its first row, so a layout
            shifted as a whole looks the same; like RevIN in forecasting).
        hypotheses: futures predicted (``rows`` output), each with a confidence logit;
            ``window`` is the most confident one.
        support_patch: consecutive demonstration rows merged into one token (PatchTST /
            TimesFM style); 1 keeps every row.
        bf16_trunk: run the axial blocks in bfloat16 on CUDA (cell embeddings, pooling and
            the output stay float32, so positions keep full precision).
        aux_factors: heads on the task vector that predict the demonstrated rule's factors
            (manipulation, eligibility, order, repeat, compaction), for an auxiliary loss.
    """

    def __init__(self, width: int = 128, heads: int = 4, support_layers: int = 4,
                 query_layers: int = 4, horizon: int = 16, task_read: str = "cell",
                 output: str = "head", task_tokens: int = 1, use_support: bool = True,
                 frame: str = "workcell", hypotheses: int = 1, support_patch: int = 1,
                 bf16_trunk: bool = False, aux_factors: bool = False) -> None:
        super().__init__()
        if task_read not in TASK_READS or output not in OUTPUTS:
            raise ValueError(f"task_read in {TASK_READS}, output in {OUTPUTS}")
        if frame not in FRAMES:
            raise ValueError(f"frame in {FRAMES}")
        if hypotheses > 1 and output != "rows":
            raise ValueError("several hypotheses need the rows output")
        self.config = dict(arch=ARCH, width=width, heads=heads, support_layers=support_layers,
                           query_layers=query_layers, horizon=horizon, task_read=task_read,
                           output=output, task_tokens=task_tokens, use_support=use_support,
                           frame=frame, hypotheses=hypotheses, support_patch=support_patch,
                           bf16_trunk=bf16_trunk, aux_factors=aux_factors)
        self.horizon, self.task_read, self.output = horizon, task_read, output
        self.use_support, self.frame, self.hypotheses = use_support, frame, hypotheses
        self.support_patch, self.bf16_trunk = support_patch, bf16_trunk

        # Cells: value + header.
        self.pose_value = nn.Sequential(nn.Linear(7, width), nn.GELU(), nn.Linear(width, width))
        self.time_value = nn.Linear(2 * len(TIME_PERIODS), width)
        self.kind = nn.Embedding(len(KINDS) + 5, width)
        self.colour = nn.Embedding(len(COLOURS), width)
        self.category = nn.Embedding(len(CATEGORIES), width)
        self.size = nn.Linear(4, width)
        self.masked_value = nn.Parameter(torch.zeros(width))      # an unknown cell's value
        self.cls_value = nn.Parameter(torch.zeros(width))

        # Demonstration and its pooling into task tokens.
        self.support_blocks = nn.ModuleList(_AxialBlock(width, heads) for _ in range(support_layers))
        self.task_seeds = nn.Parameter(torch.randn(task_tokens, width) * 0.02)
        self.null_row = nn.Parameter(torch.zeros(width))          # keeps an empty support defined
        self.pool = _Attention(width, heads)
        if support_patch > 1:
            self.patch = nn.Linear(support_patch * width, width)
        if aux_factors:
            self.factor_heads = nn.ModuleList(nn.Linear(width, len(vocab)) for _, vocab in FACTORS)

        # Query, and how it reads the task.
        self.query_blocks = nn.ModuleList(_AxialBlock(width, heads) for _ in range(query_layers))
        if task_read == "film":
            self.film = nn.ModuleList(nn.Linear(width, 2 * width) for _ in range(query_layers))
            for layer in self.film:                       # starts as the identity
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        if task_read in ("cross", "cross_all"):
            self.cross = nn.ModuleList(_Attention(width, heads) for _ in range(query_layers))

        # Output.
        if output == "head":
            self.out_queries = nn.Parameter(torch.randn(2 * horizon, width) * 0.02)
            layer = nn.TransformerDecoderLayer(width, heads, 4 * width, batch_first=True,
                                               norm_first=True, dropout=0.0)
            self.decoder = nn.TransformerDecoder(layer, 2)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 9 * hypotheses))
        nn.init.zeros_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)
        with torch.no_grad():                               # zero output = no motion, no turn
            identity = torch.tensor([1.0, 0, 0, 0, 1.0, 0])
            for k in range(hypotheses):
                self.head[1].bias[9 * k + 3:9 * k + 9] = identity
        if hypotheses > 1:
            # Which future is most likely, read from the last observed row's CLS cell. The
            # hypotheses start apart (a small random head), or they would all learn the same.
            self.confidence = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hypotheses))
            nn.init.normal_(self.head[1].weight, std=1e-3)

    # --- cells ------------------------------------------------------------------------------
    def _pose(self, pose, origin):
        """Pose cells around ``origin`` [B, 3] (per table)."""
        shape = (origin.shape[0],) + (1,) * (pose.dim() - 2) + (3,)
        position = (pose[..., :3] - origin.reshape(shape)) / POS_SCALE
        return self.pose_value(torch.cat([position, pose[..., 3:]], -1))

    def _origin(self, b, prefix):
        """The table's reference position [B, 3]: the workcell centre, or (``relative``) its
        movable objects' centroid at its first row (the workcell centre if it has none)."""
        pose = b[prefix + "pose"]
        centre = pose.new_tensor(POS_CENTRE).expand(pose.shape[0], 3)
        if self.frame == "workcell":
            return centre
        movable = (b[prefix + "kind"] == MOVABLE).float()                           # [B, C]
        count = movable.sum(1, keepdim=True)
        centroid = (pose[:, 0, :, :3] * movable[..., None]).sum(1) / count.clamp(min=1)
        return torch.where(count > 0, centroid, centre)

    def _time(self, seconds):
        periods = seconds.new_tensor(TIME_PERIODS)
        angle = 2 * math.pi * seconds[..., None] / periods
        return self.time_value(torch.cat([angle.sin(), angle.cos()], -1))

    def _header(self, b, prefix):
        size = b[prefix + "size"]
        logvol = (torch.log(size.prod(-1, keepdim=True).clamp(min=1e-7)) - LOGVOL_CENTRE) / LOGVOL_SCALE
        return (self.kind(b[prefix + "kind"].long()) + self.colour(b[prefix + "colour"].long())
                + self.category(b[prefix + "category"].long())
                + self.size(torch.cat([size / SIZE_SCALE, logvol], -1)))          # [B, C, W]

    def _since_first_row(self, b, prefix):
        """Each table's time starts at 0 on its first row (the pack stores time relative to
        the last row)."""
        time = b[prefix + "time"]
        return time - time[:, :1]

    def _support_cells(self, b):
        """[B, T, C + 4, W]: CLS, time, entities, goal left, goal right."""
        origin = self._origin(b, "s_")
        entities = self._pose(b["s_pose"], origin) + self._header(b, "s_")[:, None]
        B, T, C, W = entities.shape
        valid = b["s_goal_valid"][..., None]                                       # [B, T, 2, 1]
        goal_value = self._pose(b["s_goal"], origin) * valid + self.masked_value * (~valid)
        goal_kind = self.kind.weight[[GOAL_LEFT, GOAL_RIGHT]]                       # [2, W]
        goals = goal_value + goal_kind
        time = (self._time(self._since_first_row(b, "s_")) + self.kind.weight[TIME])[:, :, None]
        cls = (self.cls_value + self.kind.weight[ROW_CLS]).expand(B, T, 1, W)
        cells = torch.cat([cls, time, entities, goals], 2)
        pad = torch.zeros(B, C + 4, dtype=torch.bool, device=cells.device)
        pad[:, 2:2 + C] = b["s_kind"] == PAD
        return cells, pad

    def _query_cells(self, b):
        """[B, T(+H), C + 2, W]: CLS, time, entities; with ``rows`` output, ``horizon``
        future rows follow the history (time known, every value masked)."""
        header = self._header(b, "q_")
        entities = self._pose(b["q_pose"], self._origin(b, "q_")) + header[:, None]
        B, T, C, W = entities.shape
        seconds = self._since_first_row(b, "q_")
        row_valid = b["q_row_valid"]
        if self.output == "rows":
            now = (seconds * row_valid).max(1, keepdim=True).values                # last valid row
            steps = torch.arange(1, self.horizon + 1, device=seconds.device, dtype=seconds.dtype)
            seconds = torch.cat([seconds, now + DT * steps], 1)
            future = (self.masked_value + header)[:, None].expand(B, self.horizon, C, W)
            entities = torch.cat([entities, future], 1)
            row_valid = torch.cat([row_valid, torch.ones(B, self.horizon, dtype=torch.bool,
                                                         device=row_valid.device)], 1)
        rows = entities.shape[1]
        time = (self._time(seconds) + self.kind.weight[TIME])[:, :, None]
        cls = (self.cls_value + self.kind.weight[ROW_CLS]).expand(B, rows, 1, W)
        cells = torch.cat([cls, time, entities], 2)
        pad = torch.zeros(B, C + 2, dtype=torch.bool, device=cells.device)
        pad[:, 2:] = b["q_kind"] == PAD
        return cells, pad, ~row_valid

    @staticmethod
    def _wrist_columns(kind):
        """Entity-column index of the left and right wrist per item."""
        if not torch.compiler.is_compiling():
            if not ((kind == LEFT_WRIST).sum(1) == 1).all() or not ((kind == RIGHT_WRIST).sum(1) == 1).all():
                raise ValueError("every table needs exactly one left and one right wrist column")
        left = (kind == LEFT_WRIST).float().argmax(1)
        right = (kind == RIGHT_WRIST).float().argmax(1)
        return torch.stack([left, right], 1)                                       # [B, 2]

    # --- the demonstration -----------------------------------------------------------------
    def encode_support(self, b: dict) -> dict:
        """The demonstration as row tokens [B, T, W] (row CLS cells) and task tokens."""
        cells, cell_pad = self._support_cells(b)
        row_pad = ~b["s_row_valid"]
        if self.support_patch > 1:
            cells, row_pad = self._patch_rows(cells, row_pad)
        with self._trunk_precision(cells):
            for block in self.support_blocks:
                cells = block(cells, cell_pad, row_pad)
        rows = cells[:, :, 0].float()                                               # [B, T, W]
        B = rows.shape[0]
        keys = torch.cat([rows, self.null_row.expand(B, 1, -1)], 1)
        keys_pad = torch.cat([row_pad, torch.zeros(B, 1, dtype=torch.bool, device=rows.device)], 1)
        seeds = self.task_seeds.expand(B, -1, -1)
        task = self.pool(seeds, kv=keys, key_pad=keys_pad)                         # [B, K, W]
        return {"rows": keys, "rows_pad": keys_pad, "task": task}

    def factor_logits(self, memory: dict) -> list[torch.Tensor]:
        """Per factor, logits [G, vocabulary] of the demonstrated rule (``aux_factors``)."""
        pooled = memory["task"].mean(1)
        return [head(pooled) for head in self.factor_heads]

    def _patch_rows(self, cells, row_pad):
        """Merge ``support_patch`` consecutive rows of every column into one token."""
        B, T, C, W = cells.shape
        P = self.support_patch
        if T % P:
            raise ValueError(f"{T} support rows do not split into patches of {P}")
        merged = cells.reshape(B, T // P, P, C, W).permute(0, 1, 3, 2, 4).reshape(B, T // P, C, P * W)
        return self.patch(merged), row_pad.reshape(B, T // P, P).all(-1)

    def _trunk_precision(self, cells):
        enabled = self.bf16_trunk and cells.is_cuda
        return torch.autocast(device_type=cells.device.type, dtype=torch.bfloat16, enabled=enabled)

    @staticmethod
    def share(memory: dict, index: torch.Tensor, n_queries: int) -> dict:
        """Each query's demonstration encoding, from the G encoded supports."""
        n_supports = memory["task"].shape[0]
        if index.shape != (n_queries,):
            raise ValueError(f"support_index {tuple(index.shape)} does not map {n_queries} queries")
        if not torch.compiler.is_compiling():
            if int(index.min()) < 0 or int(index.max()) >= n_supports:
                raise ValueError(f"support_index {tuple(index.shape)} does not map {n_queries} "
                                 f"queries onto {n_supports} supports")
        return {key: value.index_select(0, index) for key, value in memory.items()}

    # --- the query ---------------------------------------------------------------------------
    def predict(self, memory: dict | None, b: dict) -> dict:
        cells, cell_pad, row_pad = self._query_cells(b)
        B, R, C, W = cells.shape
        read = memory is not None and self.use_support
        task = memory["task"] if read else None
        if read and self.task_read == "cell":
            K = task.shape[1]
            cells = torch.cat([cells, (task + self.kind.weight[TASK])[:, None].expand(B, R, K, W)], 2)
            cell_pad = torch.cat([cell_pad, torch.zeros(B, K, dtype=torch.bool, device=cells.device)], 1)
        with self._trunk_precision(cells):
            for k, block in enumerate(self.query_blocks):
                if read and self.task_read == "film":
                    gamma, beta = self.film[k](task.mean(1)).chunk(2, -1)
                    cells = cells * (1 + gamma[:, None, None]) + beta[:, None, None]
                if read and self.task_read in ("cross", "cross_all"):
                    # Before the block, so the row's other cells read the update in its row
                    # attention (after the last block nothing would read it).
                    keys = torch.cat([memory["rows"], task], 1)
                    keys_pad = torch.cat([memory["rows_pad"], torch.zeros(B, task.shape[1], dtype=torch.bool,
                                                                          device=cells.device)], 1)
                    if self.task_read == "cross":
                        row_cls = self.cross[k](cells[:, :, 0], kv=keys, key_pad=keys_pad)
                        cells = torch.cat([row_cls[:, :, None], cells[:, :, 1:]], 2)
                    else:
                        flat = self.cross[k](cells.reshape(B, R * cells.shape[2], -1), kv=keys,
                                             key_pad=keys_pad)
                        cells = flat.reshape(cells.shape)
                cells = block(cells, cell_pad, row_pad)
        cells = cells.float()

        history = b["q_row_valid"]
        last = history.long().cumsum(1).argmax(1)                                   # last observed row
        wrists = self._wrist_columns(b["q_kind"])                                   # [B, 2]
        if self.output == "head":
            row = cells[torch.arange(B), last]                                      # [B, C', W]
            queries = self.out_queries.expand(B, -1, -1)
            h = self.decoder(queries, row, memory_key_padding_mask=cell_pad)
            out = self.head(h).reshape(B, 2, self.horizon, 9).transpose(1, 2)        # [B, H, 2, 9]
        else:
            T = history.shape[1]
            future = cells[:, T:T + self.horizon]                                    # [B, H, C', W]
            index = (2 + wrists)[:, None, :, None].expand(B, self.horizon, 2, W)
            out = self.head(torch.gather(future, 2, index))                         # [B, H, 2, 9K]

        anchor = torch.gather(b["q_pose"][torch.arange(B), last], 1,
                              wrists[:, :, None].expand(B, 2, 7))                    # [B, 2, 7]
        if self.hypotheses == 1:
            return {"window": self._window(anchor, out), "delta": out[..., :3] * DELTA_SCALE}
        K = self.hypotheses
        out = out.reshape(B, self.horizon, 2, K, 9).permute(0, 3, 1, 2, 4)          # [B, K, H, 2, 9]
        windows = self._window(anchor[:, None], out)                                # [B, K, H, 2, 7]
        confidence = self.confidence(cells[torch.arange(B), last, 0])               # [B, K]
        best = confidence.argmax(1)
        return {"window": windows[torch.arange(B), best], "windows": windows,
                "confidence": confidence, "delta": out[torch.arange(B), best][..., :3] * DELTA_SCALE}

    @staticmethod
    def _window(anchor, out):
        """Offsets ``out`` [..., H, 2, 9] applied to the measured wrists ``anchor`` [..., 2, 7]."""
        pos = anchor[..., None, :, :3] + DELTA_SCALE * out[..., :3]
        turn = quat_from_6d(out[..., 3:9])
        rot = _quat_mul(anchor[..., None, :, 3:].expand_as(turn), turn)
        return torch.cat([pos, F.normalize(rot, dim=-1)], -1)

    def forward(self, b: dict) -> dict:
        memory = None
        factor_logits = None
        if self.use_support:
            memory = self.encode_support(b)
            if self.config["aux_factors"]:
                factor_logits = self.factor_logits(memory)               # one row per support
            n_queries = b["q_pose"].shape[0]
            if "support_index" in b:
                memory = self.share(memory, b["support_index"].long(), n_queries)
            elif b["s_pose"].shape[0] != n_queries:
                raise ValueError("supports and queries differ in number: pass support_index")
        out = self.predict(memory, b)
        if factor_logits is not None:
            out["factor_logits"] = factor_logits
        return out
