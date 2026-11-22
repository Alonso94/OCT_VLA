"""Items: (support table, query table, target window), and packing them into batches.

Support rows come from an observable sampler, query times are stratified over moving,
transition and idle rows, and conflict pairs share one query scene. Only
``Table.model_inputs()`` reaches the packed arrays. Details: ``docs/bpfn.md``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from oct_vla.bpfn.data.syn_data import generator
from oct_vla.bpfn.data.syn_data.observation import CLEAN, FaultConfig, observe
from oct_vla.bpfn.schema import (KINDS, PAD, POSE_DIM, Episode, Table, query_table, relabel,
                     semantic_relabeling, shuffle_columns, support_table, target_window)
from oct_vla.bpfn.data.splits import CountMatrix

S_ROWS = 128
C_MAX = 2 + generator.MAX_MOVABLES + 3      # wrists, movables, two decks, a marker
HISTORY = 16
#: 16 rows = 1.07 s at 15 Hz. 8 (0.53 s) was the first configuration; on the recorded
#: v4 runs the next object is identifiable from the expert's own window in 82 % of
#: approach onsets at 8 rows and 100 % at 16 (real approaches are slower than 0.5 s).
HORIZON = 16
TRANSITION = 5
APERTURE_STEP = 0.02     # aperture change per row that counts as a contact change
MOVE_STEP = 1e-4         # m per row: an object moving


def support_rows(ep: Episode, budget: int = S_ROWS) -> np.ndarray:
    """Observable keyframes plus uniform fill, sorted, at most ``budget`` rows."""
    T = ep.T
    keep = {0, T - 1}
    ap = np.abs(np.diff(ep.aperture, axis=0)).max(1) > APERTURE_STEP * max(
        1e-6, float(np.ptp(ep.aperture)) or 1.0)
    moving = np.linalg.norm(np.diff(ep.objects[..., :3], axis=0), axis=-1) > MOVE_STEP
    starts_stops = np.flatnonzero(np.diff(moving.astype(np.int8), axis=0).any(1)) + 1
    keep.update(int(t) + 1 for t in np.flatnonzero(ap))
    keep.update(int(t) for t in starts_stops)
    keep = sorted(k for k in keep if 0 <= k < T)
    if len(keep) > budget:
        keep = sorted(set(np.array(keep)[np.linspace(0, len(keep) - 1, budget).round().astype(int)]))
    fill = budget - len(keep)
    if fill > 0 and T > len(keep):
        extra = np.setdiff1d(np.linspace(0, T - 1, min(T, budget)).round().astype(int), keep)
        keep = sorted(set(keep) | set(extra[:fill].tolist()))
    return np.asarray(keep[:budget])


def strata(ep: Episode) -> dict[str, np.ndarray]:
    """Query-time strata from observables (not from generator labels)."""
    T = ep.T
    speed = np.r_[np.linalg.norm(np.diff(ep.wrist[..., :3], axis=0), axis=-1).max(1), 0.0]
    ap = np.r_[np.abs(np.diff(ep.aperture, axis=0)).max(1) > 1e-3, False]
    change = np.flatnonzero(ap)
    near = np.zeros(T, bool)
    for c in change:
        near[max(0, c - TRANSITION):c + TRANSITION + 1] = True
    rows = np.arange(T - 1)                  # a query needs at least one future row
    return {"transition": rows[near[:-1]],
            "moving": rows[(speed[:-1] > 0.01 / 15) & ~near[:-1]],
            "idle": rows[(speed[:-1] <= 0.01 / 15) & ~near[:-1]]}


def query_time(ep: Episode, rng, before: int | None = None) -> tuple[int, str]:
    groups = {k: (v if before is None else v[v < before]) for k, v in strata(ep).items()}
    groups = {k: v for k, v in groups.items() if len(v)}
    if not groups:
        return 0, "initial"
    name = list(groups)[int(rng.integers(len(groups)))]
    return int(rng.choice(groups[name])), name


@dataclass
class ItemConfig:
    programs: tuple
    counts: CountMatrix = field(default_factory=CountMatrix)
    conflict_fraction: float = 0.5
    branch_fraction: float = 0.3      # handover queries drawn from a delayed branch
    faults: FaultConfig = CLEAN       # query observation faults
    relabel: bool = False             # one shared semantic relabeling per pair
    history: int = HISTORY
    horizon: int = HORIZON
    s_rows: int = S_ROWS              # support row budget (observable sampler)
    #: Items per generated support/query pair, at independently drawn query times (and
    #: fresh column shuffles). Episode generation dominates the cost of an item (~0.1 s
    #: against ~0.2 ms to pack), so reuse multiplies throughput; the items of one group
    #: are correlated, which a batch of 64 dilutes.
    reuse: int = 1
    #: Extra items per conflict-pair member at other times. Kept small so the pure pairs
    #: (identical inputs, support decides) stay a large share of each batch.
    pair_extra: int = 0


def _distractors(program: generator.Program) -> int:
    return 0 if program.eligibility_kind == "all" else 1


def first_divergence(a: Episode, b: Episode) -> int:
    n = min(a.T, b.T)
    flat = lambda e: np.concatenate([e.wrist[:n].reshape(n, -1), e.objects[:n].reshape(n, -1),  # noqa: E731
                                     e.aperture[:n]], 1)
    diff = np.flatnonzero(~np.all(np.isclose(flat(a), flat(b), atol=1e-9), axis=1))
    return int(diff[0]) if len(diff) else n


def _item(rng, cfg: ItemConfig, support: Episode, query: Episode, t: int, stratum: str,
          cue: int | None, meta: dict, mapping=None) -> dict:
    s_tab = support_table(support, support_rows(support, cfg.s_rows))
    q_tab = query_table(query, t, cfg.history, obs=observe(query, rng, cfg.faults))
    s_tab, _ = shuffle_columns(s_tab, rng)
    q_tab, _ = shuffle_columns(q_tab, rng)
    if mapping is not None:
        s_tab, q_tab = relabel(s_tab, mapping), relabel(q_tab, mapping)
    window, valid = target_window(query, t, cfg.horizon, cue=cue)
    return {"support": s_tab, "query": q_tab, "target": window, "target_valid": valid,
            "meta": {**meta, "t": t, "stratum": stratum, "cue": cue,
                     "physical_support": support.annotations["physical"],
                     "eligible_support": support.annotations["eligible_count"],
                     "physical_query": query.annotations["physical"],
                     "eligible_query": query.annotations["eligible_count"],
                     "remaining_query": _remaining(query, t)}}


def handover_family(program: generator.Program, q_n: int, rng) -> tuple[Episode, Episode, int]:
    """Nominal and delayed-receiver handovers from one scene and one random stream, and
    the cue (first row at which they differ)."""
    for _ in range(50):
        try:
            scene = generator.sample_scene(program, q_n, _distractors(program), rng)
            state = copy.deepcopy(rng)
            nominal = generator.execute(program, scene, copy.deepcopy(state))
            k = int(rng.integers(len(nominal.annotations["selected"])))
            delayed = generator.execute(program, scene, copy.deepcopy(state),
                                 perturb={"transfer": k, "at": float(rng.uniform(0.2, 0.8)),
                                          "slow": float(rng.uniform(0.25, 0.5))})
            return nominal, delayed, first_divergence(nominal, delayed)
        except generator.Infeasible:
            continue
    raise generator.Infeasible("no feasible handover family")


def conflict_queries(program: generator.Program, partner: generator.Program, value: str | None, q_n: int,
                     rng) -> tuple[Episode, Episode, int]:
    """One query scene executed under two programs with different first picks, and the
    row at which the executions diverge."""
    for _ in range(50):
        try:
            d = _distractors(program) or 0
            scene = generator.sample_scene(program, q_n, max(d, 1 if value else 0), rng,
                                    distractor_value=value)
            state = copy.deepcopy(rng)
            qa = generator.execute(program, scene, copy.deepcopy(state))
            qb = generator.execute(partner, generator.rescope(scene, partner), copy.deepcopy(state))
            break
        except generator.Infeasible:
            continue
    else:
        raise generator.Infeasible("no feasible conflict scene")
    if qa.annotations["selected"][0] == qb.annotations["selected"][0]:
        raise generator.Infeasible("conflict pair with the same first pick")
    return qa, qb, first_divergence(qa, qb)


def _remaining(ep: Episode, t: int) -> int:
    """Selected objects not yet let go by time t (evaluator-only count)."""
    done = {e["obj"] for e in ep.annotations.get("events", [])
            if e["event"] in ("place", "push_end") and e["row"] <= t}
    return len([o for o in ep.annotations.get("selected", []) if o not in done])


def sample_item(rng, cfg: ItemConfig, program: generator.Program | None = None,
                reuse: int | None = None):
    """One support--query item of one program (independent scenes); with ``cfg.reuse`` > 1
    (or ``reuse``), a list of that many items from the same pair at different times."""
    program = program or cfg.programs[int(rng.integers(len(cfg.programs)))]
    s_n, q_n = cfg.counts.draw(rng)
    support = generator.sample_episode(program, s_n, _distractors(program), rng)
    cue, branch = None, "nominal"
    if program.manipulation == "handover" and rng.random() < cfg.branch_fraction:
        nominal, delayed, cue = handover_family(program, q_n, rng)
        branch = "delayed" if rng.random() < 0.5 else "nominal"
        query = delayed if branch == "delayed" else nominal
    else:
        query = generator.sample_episode(program, q_n, _distractors(program), rng)
    group = []
    for _ in range(max(1, reuse if reuse is not None else cfg.reuse)):
        t, stratum = query_time(query, rng)
        mapping = semantic_relabeling(rng) if cfg.relabel else None
        group.append(_item(rng, cfg, support, query, t, stratum, cue,
                           {"program": program.key, "family": program.family,
                            "branch": branch, "pair": None}, mapping))
    return group[0] if reuse is None and cfg.reuse == 1 else group


def conflict_partner(program: generator.Program, rng) -> tuple[generator.Program, str | None]:
    """A program whose first picks on a shared scene are disjoint from ``program``'s, and
    the distractor value the shared scene needs."""
    kind = program.eligibility_kind
    if kind == "colour":
        other = str(rng.choice([c for c in generator.GEN_COLOURS if c != program.eligibility[7:]]))
        return generator.Program(program.manipulation, f"colour:{other}", program.order,
                          program.repeat, program.compaction), other
    if kind == "category":
        other = [c for c in generator.GEN_CATEGORIES if c != program.eligibility[9:]][0]
        return generator.Program(program.manipulation, f"category:{other}", program.order,
                          program.repeat, program.compaction), other
    flip = {"left_to_right": "right_to_left", "right_to_left": "left_to_right",
            "small_to_large": "large_to_small", "large_to_small": "small_to_large",
            "nearest_marker": "farthest_marker", "farthest_marker": "nearest_marker"}
    return generator.Program(program.manipulation, program.eligibility, flip[program.order],
                      program.repeat, program.compaction), None


def sample_pair(rng, cfg: ItemConfig, program: generator.Program | None = None) -> list[dict]:
    """Two items sharing one query scene; programs with disjoint first picks."""
    program = program or cfg.programs[int(rng.integers(len(cfg.programs)))]
    partner, value = conflict_partner(program, rng)
    if partner not in cfg.programs and program in cfg.programs:
        raise generator.Infeasible(f"partner {partner.key} is outside this arm's programs")
    s_n, q_n = cfg.counts.draw(rng)
    qa, qb, split = conflict_queries(program, partner, value, q_n, rng)
    t, stratum = query_time(qa, rng, before=split)
    mapping = semantic_relabeling(rng) if cfg.relabel else None
    pair = int(rng.integers(1 << 62))
    items, extra = [], []
    for prog, q in ((program, qa), (partner, qb)):
        sup = generator.sample_episode(prog, s_n, _distractors(prog), rng)
        items.append(_item(rng, cfg, sup, q, t, stratum, None,
                           {"program": prog.key, "family": prog.family, "branch": "nominal",
                            "pair": pair, "first_pick": q.annotations["selected"][0],
                            "diverge": split}, mapping))
        # Reuse: the same support and query at other times are ordinary supervised items
        # (only the pair itself needs identical inputs).
        for _ in range(cfg.pair_extra):
            tt, st = query_time(q, rng)
            extra.append(_item(rng, cfg, sup, q, tt, st, None,
                               {"program": prog.key, "family": prog.family,
                                "branch": "nominal", "pair": None}, None))
    return items + extra


def sample_items(rng, cfg: ItemConfig, n: int) -> list[dict]:
    out = []
    while len(out) < n:
        try:
            if rng.random() < cfg.conflict_fraction:
                out.extend(sample_pair(rng, cfg))
            else:
                got = sample_item(rng, cfg, reuse=cfg.reuse)
                out.extend(got)
        except generator.Infeasible:
            continue
    return out   # may exceed n by one so a conflict pair is never split


def _pad_table(tab: Table, rows: int, cols: int) -> dict[str, np.ndarray]:
    T, C = tab.pose.shape[:2]
    if T > rows or C > cols:
        raise ValueError(f"table {T}x{C} exceeds the {rows}x{cols} pack")
    out = {
        "pose": np.zeros((rows, cols, POSE_DIM), np.float32),
        "observed": np.zeros((rows, cols), bool),
        "age": np.zeros((rows, cols), np.float32),
        "row_valid": np.zeros(rows, bool),
        "time": np.zeros(rows, np.float32),
        "kind": np.full(cols, PAD, np.int8),
        "colour": np.zeros(cols, np.int8),
        "category": np.zeros(cols, np.int8),
        "size": np.zeros((cols, 3), np.float32),
        "goal": np.zeros((rows, 2, POSE_DIM), np.float32),
        "goal_valid": np.zeros((rows, 2), bool),
    }
    # Unused pose cells hold the identity quaternion: an all-zero cell is never a pose.
    out["pose"][..., 6] = 1.0
    out["goal"][..., 6] = 1.0
    m = tab.model_inputs()
    out["pose"][:T, :C] = m["pose"]
    out["observed"][:T, :C] = m["observed"]
    out["age"][:T, :C] = m["age"]
    out["row_valid"][:T] = True
    if T:
        out["time"][:T] = m["time"] - m["time"][-1]   # relative to the table's last row
    for k in ("kind", "colour", "category"):
        out[k][:C] = m[k]
    out["size"][:C] = m["size"]
    out["goal"][:T] = m["goal"]
    out["goal_valid"][:T] = m["goal_valid"]
    return out


#: The rule's factors, each a small vocabulary (``s_factors`` in a packed batch).
FACTORS = (("manipulation", generator.MANIPULATIONS), ("eligibility", generator.ELIGIBILITY_KINDS),
           ("order", generator.ORDERS), ("repeat", generator.REPEATS),
           ("compaction", generator.COMPACTIONS))


def rule_factors(meta: dict) -> np.ndarray:
    """The demonstrated program's factor ids, or -1 where unknown (a recording has none).
    Training labels for an auxiliary loss only; never a model input."""
    out = np.full(len(FACTORS), -1, np.int64)
    program = meta.get("program")
    if program:
        manipulation, eligibility, order, repeat, compaction = program.split("|")
        values = (manipulation, eligibility.split(":")[0], order, repeat, compaction)
        for k, ((_, vocab), value) in enumerate(zip(FACTORS, values)):
            out[k] = vocab.index(value)
    return out


def pack(items: list[dict], s_rows: int = S_ROWS, cols: int = C_MAX) -> dict[str, np.ndarray]:
    """Stack items into a batch; keys ``s_*``, ``q_*``, ``target``, ``target_valid``."""
    batch: dict[str, list] = {}
    for it in items:
        for prefix, tab, rows in (("s_", it["support"], s_rows), ("q_", it["query"], HISTORY)):
            for k, v in _pad_table(tab, rows, cols).items():
                batch.setdefault(prefix + k, []).append(v)
        batch.setdefault("target", []).append(it["target"])
        batch.setdefault("target_valid", []).append(it["target_valid"])
        batch.setdefault("s_factors", []).append(rule_factors(it.get("meta", {})))
    out = {k: np.stack(v) for k, v in batch.items()}
    # The query's goal columns are masked placeholders by construction.
    if out["q_goal_valid"].any():
        raise AssertionError("a query table carries a valid goal: future leaked into the input")
    return out



# ---- groups: one support, several queries (the support is encoded once per batch) -------

def group(rng, cfg: ItemConfig, support: Episode, entries: list[tuple], mapping=None) -> dict:
    """One support table shared by several query items.

    ``entries``: (query episode, t, stratum, cue, meta) per query. The support is sampled
    and column-shuffled once; each query gets its own observation draw and shuffle; one
    relabeling (``mapping``) covers the support and every query, as for a single pair.
    """
    s_tab = support_table(support, support_rows(support, cfg.s_rows))
    s_tab, _ = shuffle_columns(s_tab, rng)
    if mapping is not None:
        s_tab = relabel(s_tab, mapping)
    items = []
    for query, t, stratum, cue, meta in entries:
        q_tab = query_table(query, t, cfg.history, obs=observe(query, rng, cfg.faults))
        q_tab, _ = shuffle_columns(q_tab, rng)
        if mapping is not None:
            q_tab = relabel(q_tab, mapping)
        window, valid = target_window(query, t, cfg.horizon, cue=cue)
        meta = {**meta, "t": t, "stratum": stratum, "cue": cue,
                "physical_support": support.annotations["physical"],
                "eligible_support": support.annotations["eligible_count"],
                "physical_query": query.annotations["physical"],
                "eligible_query": query.annotations["eligible_count"],
                "remaining_query": _remaining(query, t)}
        items.append({"query": q_tab, "target": window, "target_valid": valid, "meta": meta})
    return {"support": s_tab, "items": items}


def ungroup(groups: list[dict]) -> list[dict]:
    """The same items, each with its group's support attached."""
    items = []
    for g in groups:
        for item in g["items"]:
            items.append({"support": g["support"], **item})
    return items


def pack_groups(groups: list[dict], s_rows: int = S_ROWS, cols: int = C_MAX) -> dict[str, np.ndarray]:
    """Like ``pack``, but ``s_*`` has one entry per group and ``q_*`` / ``target`` one per
    query; ``support_index`` maps each query to its group's support."""
    batch: dict[str, list] = {}
    support_index = []
    for g_index, g in enumerate(groups):
        if not g["items"]:
            raise AssertionError("a group without queries: its support would be encoded for nothing")
        for key, value in _pad_table(g["support"], s_rows, cols).items():
            batch.setdefault("s_" + key, []).append(value)
        batch.setdefault("s_factors", []).append(rule_factors(g["items"][0].get("meta", {})))
        for item in g["items"]:
            for key, value in _pad_table(item["query"], HISTORY, cols).items():
                batch.setdefault("q_" + key, []).append(value)
            batch.setdefault("target", []).append(item["target"])
            batch.setdefault("target_valid", []).append(item["target_valid"])
            support_index.append(g_index)
    out = {key: np.stack(values) for key, values in batch.items()}
    out["support_index"] = np.asarray(support_index, np.int64)
    if out["q_goal_valid"].any():
        raise AssertionError("a query table carries a valid goal: future leaked into the input")
    return out

__all__ = ["C_MAX", "HISTORY", "HORIZON", "ItemConfig", "KINDS", "S_ROWS", "group", "pack",
           "pack_groups", "sample_item", "sample_items", "sample_pair", "support_rows", "ungroup"]
