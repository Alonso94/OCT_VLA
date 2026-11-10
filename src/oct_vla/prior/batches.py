"""Training and evaluation items of the interaction prior, packed into fixed-shape arrays.

One item is:
- K support demonstrations of one program, each a keyframe table [S, N, F] with per-row
  event, acting gripper and "involved" flags (acted object and acting gripper(s));
- a query: the initial and the current state of another scene under the same program;
- an instruction: none, ambiguous, or explicit;
- the next keyframe as the target.

Entity columns are permuted per table (consistently across a demo's rows), so nothing
can be read from column order.

Conflict items come in adjacent pairs:
- they share the query state and the support scenes;
- their programs differ in exactly one component;
- their next keyframes differ.

Only the support, or an explicit instruction, can tell the two apart, which is what makes
`contrast accuracy` (both of a pair right) a test of context use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import programs as P
from .generator import K_MAX, N_MAX, S_MAX, Episode, Infeasible, execute, interpolate, sample_scene
from .schema import GRIPPER_TYPES, N_FEATURES, POS, YAW

LANG_NONE, LANG_AMBIGUOUS, LANG_EXPLICIT = range(3)
#: A next keyframe "differs" between conflict partners if any of these change.
DIFFERENT_POSITION = 0.02


@dataclass(frozen=True)
class ItemConfig:
    programs: tuple[P.Program, ...] = P.TRAIN_PROGRAMS
    conflict_fraction: float = 0.5
    #: Probabilities of (none, ambiguous, explicit) instructions.
    language: tuple[float, float, float] = (0.2, 0.4, 0.4)
    #: With an explicit instruction, how often the support is dropped (language only).
    support_dropout: float = 0.25
    k_range: tuple[int, int] = (1, K_MAX)
    #: Probability that the query sits exactly on a keyframe (alpha = 0).
    on_keyframe: float = 0.3


def _query(ep: Episode, j: int, alpha: float):
    kf = ep.keyframes
    current = kf[j].feats if alpha == 0 else interpolate(kf[j], kf[j + 1], alpha)
    return np.stack([kf[0].feats, current])


def _target(ep: Episode, j: int) -> dict:
    nxt = ep.keyframes[j + 1]
    n = ep.scene.n
    obj = nxt.feats[nxt.obj]
    grip = nxt.feats[n + nxt.gripper]
    fp = ep.final_pose[nxt.obj]
    return {
        "obj": nxt.obj,
        "grip": nxt.gripper,
        "event": nxt.event,
        "obj_pose": np.concatenate([obj[POS], obj[YAW]]),
        "grip_pos": grip[POS],
        "aperture": grip[8],
        "final": np.concatenate([fp[:3], [np.sin(fp[3]), np.cos(fp[3])]]),
    }


def _differ(a: dict, b: dict) -> bool:
    return (
        a["obj"] != b["obj"]
        or a["grip"] != b["grip"]
        or a["event"] != b["event"]
        or np.linalg.norm(a["obj_pose"][:3] - b["obj_pose"][:3]) > DIFFERENT_POSITION
        or np.linalg.norm(a["grip_pos"] - b["grip_pos"]) > DIFFERENT_POSITION
    )


def _partner(rng, program: P.Program, programs) -> P.Program | None:
    pool = [p for p in programs if len(p.differs_in(program)) == 1]
    return pool[rng.integers(len(pool))] if pool else None


#: Object-count distribution of a scene. The count is drawn *before* the feasibility
#: retries and held fixed through them; redrawing it per retry over-represents one-object
#: scenes (the easiest to make feasible), where the pointer is trivial.
N_OBJECTS_P = (0.1, 0.3, 0.3, 0.3)


def _scene_for(rng, progs, tries=60):
    """A scene on which every program in `progs` is feasible, and their episodes."""
    n = int(rng.choice(len(N_OBJECTS_P), p=N_OBJECTS_P)) + 1
    for _ in range(tries):
        try:
            scene = sample_scene(rng, n=n)
            return [execute(scene, p, rng) for p in progs]
        except Infeasible:
            continue
    raise Infeasible(f"no common feasible scene for {progs}")


def _instruction_index(rng, kind: int, program: P.Program) -> int:
    if kind == LANG_NONE:
        return -1
    if kind == LANG_AMBIGUOUS:
        s = P.AMBIGUOUS_INSTRUCTIONS[rng.integers(len(P.AMBIGUOUS_INSTRUCTIONS))]
    else:
        s = program.instructions()[rng.integers(2)]
    return P.ALL_INSTRUCTIONS.index(s)


def sample_items(
    rng: np.random.Generator, cfg: ItemConfig, conflict: bool | None = None
) -> list[dict]:
    """One item, or a conflict pair of two items. Raises Infeasible rarely; callers retry."""
    conflict = rng.random() < cfg.conflict_fraction if conflict is None else conflict
    prog = cfg.programs[rng.integers(len(cfg.programs))]
    kind = int(rng.choice(3, p=cfg.language))
    k = int(rng.integers(cfg.k_range[0], cfg.k_range[1] + 1))
    if kind == LANG_EXPLICIT and rng.random() < cfg.support_dropout:
        k = 0
    if not conflict:
        demos = [_scene_for(rng, [prog])[0] for _ in range(k)]
        (q,) = _scene_for(rng, [prog])
        j = int(rng.integers(len(q.keyframes) - 1))
        alpha = 0.0 if rng.random() < cfg.on_keyframe else float(rng.uniform(0, 0.9))
        return [_pack(rng, prog, demos, q, j, alpha, kind)]

    other = _partner(rng, prog, cfg.programs)
    if other is None:
        raise Infeasible("no partner program")
    for _ in range(20):
        qa, qb = _scene_for(rng, [prog, other])
        common = [
            j
            for j in range(min(len(qa.keyframes), len(qb.keyframes)) - 1)
            if np.allclose(qa.keyframes[j].feats, qb.keyframes[j].feats)
            and _differ(_target(qa, j), _target(qb, j))
        ]
        if common:
            break
    else:
        raise Infeasible("partners never diverge from a shared state")
    j = int(rng.choice(common))
    pairs = [_scene_for(rng, [prog, other]) for _ in range(k)]
    a = _pack(rng, prog, [d[0] for d in pairs], qa, j, 0.0, kind)
    b = _pack(rng, other, [d[1] for d in pairs], qb, j, 0.0, kind)
    a["conflict"] = b["conflict"] = True
    return [a, b]


def _perm(rng, n_entities):
    return rng.permutation(n_entities)


def _pack(
    rng, prog: P.Program, demos: list[Episode], q: Episode, j: int, alpha: float, kind: int
) -> dict:
    item = {
        "ctx_feats": np.zeros((K_MAX, S_MAX, N_MAX, N_FEATURES), np.float32),
        "ctx_types": np.full((K_MAX, N_MAX), -1, np.int8),
        "ctx_rows": np.zeros((K_MAX, S_MAX), bool),
        "ctx_event": np.zeros((K_MAX, S_MAX), np.int8),
        "ctx_gripper": np.full((K_MAX, S_MAX), -1, np.int8),
        "ctx_flag": np.zeros((K_MAX, S_MAX, N_MAX), bool),
        "demo_valid": np.zeros(K_MAX, bool),
    }
    for d, ep in enumerate(demos):
        perm = _perm(rng, len(ep.types))  # new column c holds old entity perm[c]
        inv = np.argsort(perm)
        n = ep.scene.n
        item["ctx_types"][d, : len(perm)] = ep.types[perm]
        item["demo_valid"][d] = True
        for r, kf in enumerate(ep.keyframes):
            item["ctx_feats"][d, r, : len(perm)] = kf.feats[perm]
            item["ctx_rows"][d, r] = True
            item["ctx_event"][d, r] = kf.event
            item["ctx_gripper"][d, r] = kf.gripper
            if kf.obj >= 0:
                item["ctx_flag"][d, r, inv[kf.obj]] = True
                item["ctx_flag"][d, r, inv[n + kf.gripper]] = True  # acting gripper only
    perm = _perm(rng, len(q.types))
    inv = np.argsort(perm)
    query = _query(q, j, alpha)
    t = _target(q, j)
    item.update(
        {
            "q_feats": np.zeros((2, N_MAX, N_FEATURES), np.float32),
            "q_types": np.full(N_MAX, -1, np.int8),
            "t_obj": np.int64(inv[t["obj"]]),
            "t_grip": np.int64(t["grip"]),
            "t_event": np.int64(t["event"]),
            "t_obj_pose": t["obj_pose"].astype(np.float32),
            "t_grip_pos": t["grip_pos"].astype(np.float32),
            "t_aperture": np.float32(t["aperture"]),
            "t_final": t["final"].astype(np.float32),
            "program": np.int64(prog.id),
            "lang_kind": np.int64(kind),
            "lang_index": np.int64(_instruction_index(rng, kind, prog)),
            "events_done": np.int64(j),
            "alpha": np.float32(alpha),
            "n_objects": np.int64(q.scene.n),
            "conflict": False,
        }
    )
    item["q_feats"][:, : len(perm)] = query[:, perm]
    item["q_types"][: len(perm)] = q.types[perm]
    return item


def sample_batch(rng: np.random.Generator, cfg: ItemConfig, size: int) -> dict[str, np.ndarray]:
    """`size` items stacked; conflict pairs are adjacent and `pair` points at the partner."""
    items: list[dict] = []
    while len(items) < size:
        try:
            new = sample_items(rng, cfg, conflict=None if size - len(items) >= 2 else False)
        except Infeasible:
            continue
        if len(new) == 2:
            new[0]["pair"], new[1]["pair"] = np.int64(len(items) + 1), np.int64(len(items))
        else:
            new[0]["pair"] = np.int64(-1)
        items += new
    return {k: np.stack([np.asarray(it[k]) for it in items]) for k in items[0]}


def gripper_column(types: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """Column of gripper `grip` (0 left / 1 right) in each table: types [..., N], grip [...]."""
    want = np.asarray(GRIPPER_TYPES)[grip]
    hit = types == want[..., None]
    if not hit.any(-1).all():
        raise ValueError("a table is missing the requested gripper")
    return hit.argmax(-1)
