"""Time-row items from the unchanged v1 generator (schema `time_row_schema.TIME_ROW_V1`).

An item has three parts:
- **`inputs`**, everything a model may read:
  - K support demos, each a table of time rows `[T, N, F]` with their labelled
    left/right next-event targets;
  - the query's initial and current scene;
  - the instruction.
- **`targets`:** the query's hidden next-event labels. No model input is built from them.
- **`meta`:** provenance (program, keyframe index, alpha, permutations, conflict partner).
  It never enters `inputs`.

The sampling of programs, scenes, K, language and conflict pairs is v1's
(`batches._scene_for`, `_partner`, `_differ`, `_target`, `_instruction_index`), so the
time-row data and v1's items come from the same distribution. Only the packing is new.

**Rows.** Keyframe j, then `rows_per_segment` interpolated states between keyframes j and
j+1 (`generator.interpolate`), in time order. Every row before the last keyframe is
labelled with the *next* keyframe; rows at the last keyframe are terminal.

[limitation] Interpolated rows are linear blends, not physics. They add (x, y) pairs
with the same label, and no dynamics information.
"""

from __future__ import annotations

import dataclasses
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import programs as P
from . import time_row_schema as S
from .batches import ItemConfig, _differ, _instruction_index, _partner, _scene_for, _target
from .generator import K_MAX, N_MAX, S_MAX, Episode, Infeasible, interpolate
from .programs_v2 import ALL_INSTRUCTIONS_V2, ProgramV2
from .schema import APERTURE, N_FEATURES, OPEN, POS

#: Keyframes per v2 episode: v1's plus two push keyframes per object.
S_MAX_V2 = S_MAX + 2 * 4


@dataclass(frozen=True)
class TimeRowConfig:
    item: ItemConfig = ItemConfig()
    #: Interpolated rows per keyframe segment (0 = keyframes only).
    rows_per_segment: int = 1

    @property
    def v2(self) -> bool:
        """The grammar is the programs': v2 (push prior, gripper orientation) or v1."""
        return isinstance(self.item.programs[0], ProgramV2)

    @property
    def n_features(self) -> int:
        from .generator_v2 import N_FEATURES_V2

        return N_FEATURES_V2 if self.v2 else N_FEATURES

    @property
    def labels(self) -> dict:
        return S.ROW_LABELS_V2 if self.v2 else S.ROW_LABELS

    @property
    def schema(self) -> str:
        return S.SCHEMA_VERSION_V2 if self.v2 else S.SCHEMA_VERSION

    @property
    def t_max(self) -> int:
        s_max = S_MAX_V2 if self.v2 else S_MAX
        return s_max + (s_max - 1) * self.rows_per_segment


def row_labels(ep: Episode, next_index: int | None, cols=None) -> dict:
    """The paired labels of a row whose next keyframe is `next_index` (None: terminal).

    `cols` maps an entity index of the episode to its column in the packed table (the
    inverse permutation); identity if omitted."""
    out = {
        "y_pos": np.zeros((S.N_ARMS, 3), np.float32),
        "y_aperture": np.zeros(S.N_ARMS, np.float32),
        "y_participation": np.zeros(S.N_ARMS, np.float32),
        "y_engaged": np.zeros(S.N_ARMS, np.float32),
        "y_event": np.full(S.N_ARMS, S.HOLD, np.int64),
        "y_object": np.full(S.N_ARMS, -1, np.int64),
        "g_event": np.int64(S.NO_EVENT),
        "g_object": np.int64(-1),
        "g_goal": np.zeros(S.POSE_DIM, np.float32),
        "target_valid": np.float32(0.0),
    }
    v2 = ep.keyframes[0].feats.shape[1] > N_FEATURES
    if v2:
        out["y_rot"] = np.zeros((S.N_ARMS, S.ROT_DIM), np.float32)
    if next_index is None:
        return out
    kf = ep.keyframes[next_index]
    n = ep.scene.n
    col = (lambda i: int(i)) if cols is None else (lambda i: int(cols[i]))
    for a in range(S.N_ARMS):
        out["y_pos"][a] = kf.feats[n + a, POS]
        out["y_aperture"][a] = kf.feats[n + a, APERTURE]
        out["y_engaged"][a] = float(kf.feats[n + a, APERTURE] < OPEN - 1e-9)
        if v2:
            out["y_rot"][a] = kf.feats[n + a, N_FEATURES:N_FEATURES + S.ROT_DIM]
        if kf.gripper == a:
            out["y_participation"][a] = 1.0
            out["y_event"][a] = kf.event
            out["y_object"][a] = col(kf.obj)
    out["g_event"] = np.int64(kf.event)
    out["g_object"] = np.int64(col(kf.obj))
    fp = ep.final_pose[kf.obj]
    out["g_goal"] = np.array([*fp[:3], np.sin(fp[3]), np.cos(fp[3])], np.float32)
    out["target_valid"] = np.float32(1.0)
    return out


def episode_rows(ep: Episode, rows_per_segment: int, rng: np.random.Generator) -> list[dict]:
    """Time rows of one episode, in order: {feats, next, segment, alpha}."""
    kfs = ep.keyframes
    rows = []
    for j, kf in enumerate(kfs):
        last = j == len(kfs) - 1
        rows.append({"feats": kf.feats, "next": None if last else j + 1, "segment": j,
                     "alpha": 0.0})
        if last:
            continue
        for alpha in np.sort(rng.uniform(0.05, 0.9, rows_per_segment)):
            rows.append({"feats": interpolate(kf, kfs[j + 1], float(alpha)), "next": j + 1,
                         "segment": j, "alpha": float(alpha)})
    return rows


_INT_LABELS = ("y_event", "y_object", "g_event", "g_object")


def _empty_inputs(cfg: TimeRowConfig) -> dict:
    t = cfg.t_max
    inp = {
        "s_feats": np.zeros((K_MAX, t, N_MAX, cfg.n_features), np.float32),
        "s_types": np.full((K_MAX, N_MAX), -1, np.int8),
        "s_row_valid": np.zeros((K_MAX, t), bool),
        # Within-demo time: segment index + alpha. Supports are labelled, so their time
        # is not a leak; the query's progress is kept out of `inputs` (meta only).
        "s_row_time": np.zeros((K_MAX, t), np.float32),
        "demo_valid": np.zeros(K_MAX, bool),
        **{f"s_{k}": np.zeros((K_MAX, t, *shape), np.int64 if k in _INT_LABELS else np.float32)
           for k, shape in cfg.labels.items()},
    }
    # Padding must not look like a pointer to column 0.
    inp["s_y_object"].fill(-1)
    inp["s_g_object"].fill(-1)
    return inp


def _fill_labels(inp: dict, d: int, r: int, labels: dict) -> None:
    for k, v in labels.items():
        inp[f"s_{k}"][d, r] = v


def pack_time_row_item(
    rng: np.random.Generator, prog: P.Program, demos: list[Episode], q: Episode, j: int,
    alpha: float, kind: int, cfg: TimeRowConfig, perms: list | None = None,
) -> tuple[dict, dict, dict]:
    """One item. `perms` (demos..., query) fixes the column permutations, for tests;
    otherwise each table draws its own."""
    inp = _empty_inputs(cfg)
    perms = list(perms) if perms is not None else [None] * (len(demos) + 1)
    used = []
    for d, ep in enumerate(demos):
        perm = perms[d] if perms[d] is not None else rng.permutation(len(ep.types))
        inv = np.argsort(perm)
        used.append(perm)
        inp["s_types"][d, : len(perm)] = ep.types[perm]
        inp["demo_valid"][d] = True
        for r, row in enumerate(episode_rows(ep, cfg.rows_per_segment, rng)):
            inp["s_feats"][d, r, : len(perm)] = row["feats"][perm]
            inp["s_row_valid"][d, r] = True
            inp["s_row_time"][d, r] = row["segment"] + row["alpha"]
            _fill_labels(inp, d, r, row_labels(ep, row["next"], inv))
    qperm = perms[-1] if perms[-1] is not None else rng.permutation(len(q.types))
    qinv = np.argsort(qperm)
    kf = q.keyframes
    current = kf[j].feats if alpha == 0 else interpolate(kf[j], kf[j + 1], alpha)
    q_feats = np.zeros((2, N_MAX, cfg.n_features), np.float32)
    q_feats[0, : len(qperm)] = kf[0].feats[qperm]
    q_feats[1, : len(qperm)] = current[qperm]
    q_types = np.full(N_MAX, -1, np.int8)
    q_types[: len(qperm)] = q.types[qperm]
    inp.update({
        "q_feats": q_feats, "q_types": q_types,
        "lang_kind": np.int64(kind), "lang_index": np.int64(_language(rng, kind, prog)),
    })
    targets = row_labels(q, j + 1, qinv)
    meta = {"program": int(prog.id), "j": int(j), "alpha": float(alpha),
            "n_objects": int(q.scene.n), "support_perms": [p.tolist() for p in used],
            "query_perm": qperm.tolist(), "conflict": False, "pair": -1}
    return inp, targets, meta


def _scenes(rng, progs):
    """v1's `_scene_for`, or the v2 generator for v2 programs."""
    if isinstance(progs[0], ProgramV2):
        from .generator_v2 import scene_for_v2

        return scene_for_v2(rng, progs)
    return _scene_for(rng, progs)


def _language(rng, kind: int, prog) -> int:
    """v1's instruction index, against the v2 instruction table for v2 programs."""
    if not isinstance(prog, ProgramV2):
        return _instruction_index(rng, kind, prog)
    if kind == 0:
        return -1
    if kind == 1:
        return ALL_INSTRUCTIONS_V2.index(
            P.AMBIGUOUS_INSTRUCTIONS[rng.integers(len(P.AMBIGUOUS_INSTRUCTIONS))])
    return ALL_INSTRUCTIONS_V2.index(prog.instructions()[rng.integers(2)])


def sample_time_row_items(rng: np.random.Generator, cfg: TimeRowConfig,
                          conflict: bool | None = None) -> list[tuple[dict, dict, dict]]:
    """One item or a conflict pair, sampled exactly as v1's `batches.sample_items`."""
    ic = cfg.item
    conflict = rng.random() < ic.conflict_fraction if conflict is None else conflict
    prog = ic.programs[rng.integers(len(ic.programs))]
    kind = int(rng.choice(3, p=ic.language))
    k = int(rng.integers(ic.k_range[0], ic.k_range[1] + 1))
    if kind == 2 and rng.random() < ic.support_dropout:  # explicit, language only
        k = 0
    if not conflict:
        demos = [_scenes(rng, [prog])[0] for _ in range(k)]
        (q,) = _scenes(rng, [prog])
        j = int(rng.integers(len(q.keyframes) - 1))
        alpha = 0.0 if rng.random() < ic.on_keyframe else float(rng.uniform(0, 0.9))
        return [pack_time_row_item(rng, prog, demos, q, j, alpha, kind, cfg)]
    other = _partner(rng, prog, ic.programs)
    if other is None:
        raise Infeasible("no partner program")
    for _ in range(20):
        qa, qb = _scenes(rng, [prog, other])
        common = [i for i in range(min(len(qa.keyframes), len(qb.keyframes)) - 1)
                  if np.allclose(qa.keyframes[i].feats, qb.keyframes[i].feats)
                  and _differ(_target(qa, i), _target(qb, i))]
        if common:
            break
    else:
        raise Infeasible("partners never diverge from a shared state")
    j = int(rng.choice(common))
    pairs = [_scenes(rng, [prog, other]) for _ in range(k)]
    # The partners share the query state and the support *scenes*; the same permutations
    # keep their columns aligned, so only the program differs.
    perms = [rng.permutation(len(d[0].types)) for d in pairs] + [rng.permutation(len(qa.types))]
    a = pack_time_row_item(rng, prog, [d[0] for d in pairs], qa, j, 0.0, kind, cfg, perms)
    b = pack_time_row_item(rng, other, [d[1] for d in pairs], qb, j, 0.0, kind, cfg, perms)
    a[2]["conflict"] = b[2]["conflict"] = True
    return [a, b]


def sample_time_row_batch(rng: np.random.Generator, cfg: TimeRowConfig, size: int):
    """`size` items stacked: (inputs, targets, meta list); conflict pairs adjacent."""
    items = []
    while len(items) < size:
        try:
            new = sample_time_row_items(rng, cfg, conflict=None if size - len(items) >= 2 else False)
        except Infeasible:
            continue
        if len(new) == 2:
            new[0][2]["pair"], new[1][2]["pair"] = len(items) + 1, len(items)
        items += new
    inputs = {k: np.stack([np.asarray(it[0][k]) for it in items]) for k in items[0][0]}
    targets = {k: np.stack([np.asarray(it[1][k]) for it in items]) for k in items[0][1]}
    return inputs, targets, [it[2] for it in items]


def _chunk(args):
    from .data import chunk_seed

    name, split, c, cfg, size = args
    return sample_time_row_batch(np.random.default_rng(chunk_seed(name, split, c)), cfg, size)


def build_time_row_eval_set(out: Path, name: str, split: str, items: int, chunk: int = 100,
                            workers: int = 8, cfg: TimeRowConfig | None = None) -> Path:
    """`items` items of program split `split` into `out/<split>.npz` (+ meta, manifest).

    Seeded per chunk by (name, split, chunk) as v1's `data.build_eval_set`, so the set does
    not depend on parallelism. Never overwrites."""
    from .data import _git_revision

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{split}.npz"
    if path.exists():
        raise FileExistsError(f"{path} exists; evaluation sets are never overwritten")
    cfg = cfg or TimeRowConfig(item=ItemConfig(programs=P.program_split(split)))
    jobs = [(name, split, c, cfg, chunk) for c in range(-(-items // chunk))]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            parts = list(pool.map(_chunk, jobs))
    else:
        parts = [_chunk(j) for j in jobs]
    offset, metas = 0, []
    for _, tg, meta in parts:
        for m in meta:
            m["pair"] = m["pair"] + offset if m["pair"] >= 0 else -1
        metas += meta
        offset += len(meta)
    arrays = {f"in/{k}": np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    arrays.update({f"tg/{k}": np.concatenate([p[1][k] for p in parts]) for k in parts[0][1]})
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    (out / f"{split}.meta.jsonl").write_text("\n".join(json.dumps(m) for m in metas) + "\n")
    manifest = {
        "schema": cfg.schema, "name": name, "split": split, "items": len(metas),
        "chunk": chunk, "rows_per_segment": cfg.rows_per_segment,
        "item_config": {f.name: [p.id for p in cfg.item.programs] if f.name == "programs"
                        else getattr(cfg.item, f.name) for f in dataclasses.fields(cfg.item)},
        "git": _git_revision(),
    }
    (out / f"{split}.json").write_text(json.dumps(manifest, indent=1, default=str))
    return path


def load_time_row_eval_set(path: Path) -> tuple[dict, dict, list[dict]]:
    with np.load(path) as z:
        inputs = {k[3:]: z[k] for k in z.files if k.startswith("in/")}
        targets = {k[3:]: z[k] for k in z.files if k.startswith("tg/")}
    metas = [json.loads(line) for line in path.with_suffix(".meta.jsonl").read_text().splitlines()
             if line]
    return inputs, targets, metas


__all__ = ["TimeRowConfig", "row_labels", "episode_rows", "pack_time_row_item",
           "sample_time_row_items", "sample_time_row_batch", "build_time_row_eval_set",
           "load_time_row_eval_set"]
