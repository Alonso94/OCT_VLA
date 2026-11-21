"""A pre-generated episode bank: generate once, sample items from it in training.

Recipe 1 spent its time generating episodes (~40 ms each, two or more per item) and kept
the GPU 36 % busy. Episodes are the expensive, reusable part of an item; everything that
makes an item distinct is cheap and stays on the fly:

=================================  ===============================================
stored once (the bank)             drawn per item (``BankSampler``)
=================================  ===============================================
pool episodes per (program, n)     which support and which query (independent
                                   scenes, so any support pairs with any query)
conflict query families: one       which family, its two supports, the query time
scene run under two programs with  before the divergence
different first picks
handover families: nominal +       which branch
delayed receiver from one scene
                                   query time, column shuffles, relabeling,
                                   observation faults, support row sampling
=================================  ===============================================

**One bank serves every arm.** Units cover the full arm's programs (every other arm's
programs are a subset) at each eligible count; a sampler keeps the units its arm may see
(``no_handover`` drops handover programs and families, ``equal_count`` draws matched
counts, ``nominal_only`` sets no branch fraction).

**Determinism.** Every unit draws from its own hashed stream, named by (bank name, kind,
program, count, slot). The bank's content therefore does not depend on how many shards,
processes or jobs produced it, and a lost shard regenerates bit-identically.

**Layout, sized for the machine.** A shard is one pickle of flat float32 arrays (all
episodes' rows concatenated, with offsets) plus small per-episode records, written
atomically (temporary name, then rename) so a killed job never leaves a half shard and a
rerun skips finished ones. Training loads every shard into RAM in the main process before
the data-loader workers fork; the array buffers are never written, so the workers share
one physical copy (``gc.freeze`` keeps the collector from touching the small Python
objects). ~70 KB per episode; the default spec is ~145 k episodes, ~10 GB, against the
60 GB a 16-CPU job gets. ``Bank`` refuses to load past half the job's memory limit.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import batches as Bt
from . import generator as Gn
from . import splits as Sp
from .schema import Episode, semantic_relabeling

BANK_VERSION = 1
POOL, CONFLICT, HANDOVER = 0, 1, 2
KIND_NAMES = ("pool", "conflict", "handover")
#: record columns: kind, program, partner, count, episode a, episode b, aux
#: (aux: the divergence row of a conflict family, the cue of a handover family).
REC = ("kind", "program", "partner", "count", "ep_a", "ep_b", "aux")
UNIT_TRIES = 20


@dataclass(frozen=True)
class BankSpec:
    name: str = "v1"
    pool: int = 24        # independent episodes per (program, eligible count)
    conflict: int = 24    # conflict query families per (program, eligible count)
    handover: int = 48    # nominal/delayed families per (handover program, count)
    counts: tuple[int, ...] = Sp.COUNTS
    shards: int = 32


def bank_programs() -> tuple[Gn.Program, ...]:
    """The union of every training arm's programs (all are subsets of ``full``)."""
    return Sp.programs_for("full")


def units(spec: BankSpec, programs=None) -> list[tuple[int, str, int, int]]:
    """(kind, program key, eligible count, slot) in a fixed order."""
    out = []
    for p in programs or bank_programs():
        for n in spec.counts:
            out += [(POOL, p.key, n, j) for j in range(spec.pool)]
            out += [(CONFLICT, p.key, n, j) for j in range(spec.conflict)]
            if p.manipulation == "handover":
                out += [(HANDOVER, p.key, n, j) for j in range(spec.handover)]
    return out


def unit_rng(spec: BankSpec, unit) -> np.random.Generator:
    kind, key, n, j = unit
    digest = hashlib.sha256(
        f"{Sp.NAMESPACE}/bank/{spec.name}/{KIND_NAMES[kind]}/{key}/{n}/{j}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def generate_unit(spec: BankSpec, unit, by_key: dict) -> tuple[list[Episode], str | None, int]:
    """The unit's episodes, the partner program key (conflict) and aux; [] if infeasible."""
    kind, key, n, _ = unit
    program, rng = by_key[key], unit_rng(spec, unit)
    for _ in range(UNIT_TRIES):
        try:
            if kind == POOL:
                return [Gn.sample_episode(program, n, Bt._distractors(program), rng)], None, -1
            if kind == CONFLICT:
                partner, value = Bt.conflict_partner(program, rng)
                qa, qb, split = Bt.conflict_queries(program, partner, value, n, rng)
                return [qa, qb], partner.key, split
            nominal, delayed, cue = Bt.handover_family(program, n, rng)
            return [nominal, delayed], None, cue
        except Gn.Infeasible:
            continue
    return [], None, -1


# ---- flat episode storage --------------------------------------------------------------

def pack_episodes(eps: list[Episode]) -> dict:
    T = np.array([e.T for e in eps], np.int64)
    N = np.array([len(e.track_ids) for e in eps], np.int64)
    z = lambda a: np.concatenate([[0], np.cumsum(a)]).astype(np.int64)  # noqa: E731
    cat = lambda xs, shape, dt=np.float32: (np.concatenate(xs).astype(dt) if xs  # noqa: E731
                                            else np.zeros(shape, dt))
    return {
        "row_off": z(T), "obj_off": z(N), "cell_off": z(T * N),
        "time": cat([e.time for e in eps], (0,)),
        "wrist": cat([e.wrist for e in eps], (0, 2, 7)),
        "aperture": cat([e.aperture for e in eps], (0, 2)),
        "objects": cat([e.objects.reshape(-1, 7) for e in eps], (0, 7)),
        "kind": cat([e.kind for e in eps], (0,), np.int8),
        "colour": cat([e.colour for e in eps], (0,), np.int8),
        "category": cat([e.category for e in eps], (0,), np.int8),
        "size": cat([e.size for e in eps], (0, 3)),
        "tracks": [e.track_ids for e in eps],
        "annotations": [{k: e.annotations[k] for k in
                         ("program", "family", "selected", "eligible_count", "physical",
                          "distractors", "events") if k in e.annotations} for e in eps],
    }


def unpack_episode(a: dict, i: int) -> Episode:
    r0, r1 = a["row_off"][i], a["row_off"][i + 1]
    o0, o1 = a["obj_off"][i], a["obj_off"][i + 1]
    c0, c1 = a["cell_off"][i], a["cell_off"][i + 1]
    return Episode(time=a["time"][r0:r1], wrist=a["wrist"][r0:r1], aperture=a["aperture"][r0:r1],
                   objects=a["objects"][c0:c1].reshape(r1 - r0, o1 - o0, 7),
                   kind=a["kind"][o0:o1], colour=a["colour"][o0:o1],
                   category=a["category"][o0:o1], size=a["size"][o0:o1],
                   track_ids=a["tracks"][i], annotations=a["annotations"][i])


# ---- generation ------------------------------------------------------------------------

def manifest(spec: BankSpec, programs=None) -> dict:
    progs = programs or bank_programs()
    return {"version": BANK_VERSION, "schema": "pose-window-v1", "spec": asdict(spec),
            "programs": [p.key for p in progs], "units": len(units(spec, progs))}


def shard_path(root: Path, k: int) -> Path:
    return Path(root) / f"shard_{k:04d}.pkl"


def build_shard(root: Path, spec: BankSpec, k: int, programs=None) -> dict:
    """Generate shard ``k`` (every ``spec.shards``-th unit). Skips a finished shard."""
    path = shard_path(root, k)
    if path.exists():
        return {"shard": k, "skipped": True}
    progs = programs or bank_programs()
    by_key = {p.key: p for p in progs}
    index = {key: i for i, key in enumerate(by_key)}
    t0 = time.time()
    eps, recs, missing = [], [], []
    for unit in units(spec, progs)[k::spec.shards]:
        got, partner, aux = generate_unit(spec, unit, by_key)
        if not got:
            missing.append(unit)
            continue
        ids = [len(eps) + m for m in range(len(got))]
        eps += got
        recs.append((unit[0], index[unit[1]], index.get(partner, -1), unit[2], ids[0],
                     ids[1] if len(ids) > 1 else -1, aux))
    shard = {"version": BANK_VERSION, "spec": asdict(spec), "shard": k,
             "programs": list(by_key), "records": np.array(recs, np.int64).reshape(-1, len(REC)),
             "episodes": pack_episodes(eps), "missing": missing,
             "seconds": round(time.time() - t0, 1)}
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with tmp.open("wb") as f:
        pickle.dump(shard, f, protocol=5)
    tmp.rename(path)
    return {"shard": k, "episodes": len(eps), "infeasible": len(missing),
            "seconds": shard["seconds"], "bytes": path.stat().st_size}


def _build_one(args) -> dict:
    root, spec, k, programs = args
    return build_shard(root, spec, k, programs)


def build(root: Path, spec: BankSpec, shards: list[int] | None = None,
          workers: int | None = None, programs=None, log=print) -> dict:
    """Generate the bank's shards in parallel processes (one shard per process at a time);
    ``workers`` defaults to the CPUs this process may use (the job's allocation)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    man = json.loads(json.dumps(manifest(spec, programs)))   # as stored
    mpath = root / "bank.json"
    if mpath.exists():
        old = json.loads(mpath.read_text())
        if old != man:
            raise SystemExit(f"{mpath} describes a different bank; use a new directory")
    else:
        mpath.write_text(json.dumps(man, indent=1) + "\n")
    todo = [k for k in (range(spec.shards) if shards is None else shards)
            if not shard_path(root, k).exists()]
    workers = min(workers or len(os.sched_getaffinity(0)), max(1, len(todo)))
    log(f"bank {root}: {len(todo)} shards to build on {workers} processes")
    if not todo:
        return man
    import multiprocessing as mp

    # forkserver: a clean child even when the caller has threads (torch, a test runner).
    with mp.get_context("forkserver").Pool(workers) as pool:
        for r in pool.imap_unordered(_build_one, [(root, spec, k, programs) for k in todo]):
            log(json.dumps(r))
    return man


# ---- loading and sampling --------------------------------------------------------------

def _memory_limit() -> float | None:
    for p in ("/sys/fs/cgroup/memory.max",):
        try:
            v = Path(p).read_text().strip()
            if v != "max":
                return float(v)
        except OSError:
            pass
    per_cpu, cpus = os.environ.get("SLURM_MEM_PER_CPU"), os.environ.get("SLURM_CPUS_PER_TASK")
    if per_cpu and cpus:
        return float(per_cpu) * float(cpus) * 2 ** 20
    return None


class Bank:
    """Every shard in RAM. Episode id ``g`` lives in shard ``searchsorted(base, g)``."""

    def __init__(self, root: Path, log=print):
        root = Path(root)
        self.root = root
        self.manifest = json.loads((root / "bank.json").read_text())
        spec = BankSpec(**{k: tuple(v) if isinstance(v, list) else v
                           for k, v in self.manifest["spec"].items()})
        self.spec = spec
        paths = [shard_path(root, k) for k in range(spec.shards)]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"bank {root} is incomplete: {len(missing)} shards missing "
                             f"({missing[:3]}...)")
        size = sum(p.stat().st_size for p in paths)
        limit = _memory_limit()
        if limit is not None and size > 0.5 * limit:
            raise SystemExit(f"bank {root} is {size / 2**30:.1f} GiB, over half this job's "
                             f"{limit / 2**30:.1f} GiB; ask for more memory or a smaller bank")
        t0 = time.time()
        self.shards, recs, base = [], [], 0
        self.base = []
        for p in paths:
            with p.open("rb") as f:
                s = pickle.load(f)
            if s["version"] != BANK_VERSION or s["programs"] != self.manifest["programs"]:
                raise SystemExit(f"{p} does not belong to this bank")
            r = s["records"].copy()
            r[:, 4] += base
            r[:, 5] = np.where(r[:, 5] >= 0, r[:, 5] + base, -1)
            recs.append(r)
            self.shards.append(s["episodes"])
            self.base.append(base)
            base += len(s["episodes"]["tracks"])
        self.base = np.array(self.base)
        self.records = np.concatenate(recs)
        self.episodes = base
        self.nbytes = size
        self.programs = self.manifest["programs"]
        gc.collect()
        gc.freeze()   # forked workers then never write the pages of these objects
        log(f"bank {root}: {base} episodes, {len(self.records)} units, "
            f"{size / 2**30:.2f} GiB, loaded in {time.time() - t0:.0f} s")

    def episode(self, g: int) -> Episode:
        k = int(np.searchsorted(self.base, g, side="right")) - 1
        return unpack_episode(self.shards[k], int(g - self.base[k]))


class BankSampler:
    """Items for one arm's ``ItemConfig`` from a bank, with ``batches.sample_items``'
    mixture: a conflict pair with probability ``conflict_fraction``, else an ordinary
    item (from a handover family with probability ``branch_fraction``)."""

    def __init__(self, bank: Bank, cfg: Bt.ItemConfig):
        self.bank, self.cfg = bank, cfg
        allowed = {p.key for p in cfg.programs}
        missing = allowed - set(bank.programs)
        if missing:
            raise SystemExit(f"{len(missing)} of this arm's programs are not in the bank")
        self.programs = {k: Gn.Program(*k.split("|")) for k in bank.programs}
        ok = np.array([k in allowed for k in bank.programs])
        R = bank.records
        self.pool: dict[tuple[int, int], np.ndarray] = {}
        self.handover: dict[tuple[int, int], np.ndarray] = {}
        for kind, store, col in ((POOL, self.pool, 4), (HANDOVER, self.handover, None)):
            rows = R[(R[:, 0] == kind) & ok[R[:, 1]]]
            for (p, n) in {(int(a), int(b)) for a, b in rows[:, [1, 3]]}:
                sel = rows[(rows[:, 1] == p) & (rows[:, 3] == n)]
                store[(p, n)] = sel[:, col] if col is not None else sel
        conf = R[(R[:, 0] == CONFLICT) & ok[R[:, 1]] & (R[:, 2] >= 0)]
        conf = conf[ok[conf[:, 2]]]
        self.conflict = {n: conf[conf[:, 3] == n] for n in cfg.counts.counts}
        self.arm_programs = sorted({p for p, _ in self.pool})
        for n in cfg.counts.counts:
            if not len(self.conflict[n]):
                raise SystemExit(f"bank has no conflict family at count {n} for this arm")

    def _support(self, rng, p: int, n: int, avoid: int = -1) -> int:
        """A pool episode id of program ``p`` at count ``n`` (never ``avoid`` if another
        exists: a support and its query are independent scenes)."""
        ids = self.pool[(p, n)]
        if avoid >= 0 and len(ids) > 1:
            ids = ids[ids != avoid]
        return int(ids[int(rng.integers(len(ids)))])

    def single(self, rng) -> list[dict]:
        cfg = self.cfg
        p = self.arm_programs[int(rng.integers(len(self.arm_programs)))]
        program = self.programs[self.bank.programs[p]]
        s_n, q_n = cfg.counts.draw(rng)
        s_id = self._support(rng, p, s_n)
        cue, branch = None, "nominal"
        fams = self.handover.get((p, q_n))
        if (program.manipulation == "handover" and fams is not None and len(fams)
                and rng.random() < cfg.branch_fraction):
            fam = fams[int(rng.integers(len(fams)))]
            cue = int(fam[6])
            branch = "delayed" if rng.random() < 0.5 else "nominal"
            query = self.bank.episode(int(fam[5] if branch == "delayed" else fam[4]))
        else:
            query = self.bank.episode(self._support(rng, p, q_n, avoid=s_id))
        support = self.bank.episode(s_id)
        out = []
        for _ in range(max(1, cfg.reuse)):
            t, stratum = Bt.query_time(query, rng)
            mapping = semantic_relabeling(rng) if cfg.relabel else None
            out.append(Bt._item(rng, cfg, support, query, t, stratum, cue,
                                {"program": program.key, "family": program.family,
                                 "branch": branch, "pair": None}, mapping))
        return out

    def pair(self, rng) -> list[dict]:
        cfg = self.cfg
        s_n, q_n = cfg.counts.draw(rng)
        fams = self.conflict[q_n]
        fam = fams[int(rng.integers(len(fams)))]
        qa, qb = self.bank.episode(int(fam[4])), self.bank.episode(int(fam[5]))
        split = int(fam[6])
        t, stratum = Bt.query_time(qa, rng, before=split)
        mapping = semantic_relabeling(rng) if cfg.relabel else None
        pair = int(rng.integers(1 << 62))
        items, extra = [], []
        for p, q in ((int(fam[1]), qa), (int(fam[2]), qb)):
            prog = self.programs[self.bank.programs[p]]
            sup = self.bank.episode(self._support(rng, p, s_n))
            items.append(Bt._item(rng, cfg, sup, q, t, stratum, None,
                                  {"program": prog.key, "family": prog.family,
                                   "branch": "nominal", "pair": pair,
                                   "first_pick": q.annotations["selected"][0],
                                   "diverge": split}, mapping))
            for _ in range(cfg.pair_extra):
                tt, st = Bt.query_time(q, rng)
                extra.append(Bt._item(rng, cfg, sup, q, tt, st, None,
                                      {"program": prog.key, "family": prog.family,
                                       "branch": "nominal", "pair": None}, None))
        return items + extra

    def sample_items(self, rng, n: int) -> list[dict]:
        out = []
        while len(out) < n:
            out += self.pair(rng) if rng.random() < self.cfg.conflict_fraction else self.single(rng)
        return out
