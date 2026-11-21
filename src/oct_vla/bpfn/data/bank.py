"""The pre-generated episode bank: episodes generated once, items drawn from them in training.

Stored per unit: pool episodes per (program, eligible count), conflict families and handover
families. Every unit has its own hashed random stream, so the content does not depend on
sharding. Why and how: ``docs/bpfn.md``.
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

from oct_vla.bpfn.data import batches, splits
from oct_vla.bpfn.data.syn_data import generator
from oct_vla.bpfn.schema import Episode

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
    counts: tuple[int, ...] = splits.COUNTS
    shards: int = 32


def bank_programs() -> tuple[generator.Program, ...]:
    """The union of every training arm's programs (all are subsets of ``full``)."""
    return splits.programs_for("full")


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
        f"{splits.NAMESPACE}/bank/{spec.name}/{KIND_NAMES[kind]}/{key}/{n}/{j}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def generate_unit(spec: BankSpec, unit, by_key: dict) -> tuple[list[Episode], str | None, int]:
    """The unit's episodes, the partner program key (conflict) and aux; [] if infeasible."""
    kind, key, n, _ = unit
    program, rng = by_key[key], unit_rng(spec, unit)
    for _ in range(UNIT_TRIES):
        try:
            if kind == POOL:
                return [generator.sample_episode(program, n, batches._distractors(program), rng)], None, -1
            if kind == CONFLICT:
                partner, value = batches.conflict_partner(program, rng)
                qa, qb, split = batches.conflict_queries(program, partner, value, n, rng)
                return [qa, qb], partner.key, split
            nominal, delayed, cue = batches.handover_family(program, n, rng)
            return [nominal, delayed], None, cue
        except generator.Infeasible:
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

