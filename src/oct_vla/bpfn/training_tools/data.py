"""Training data: item samplers (from the episode bank or generated on the fly), the
item stream that packs them into batches, and the data loader."""

from __future__ import annotations

import numpy as np
import torch

from oct_vla.bpfn.data import batches
from oct_vla.bpfn.data.bank import CONFLICT, HANDOVER, POOL, Bank
from oct_vla.bpfn.data.syn_data import generator
from oct_vla.bpfn.schema import Episode, semantic_relabeling


class OnTheFlySampler:
    """Items whose episodes are generated at every draw (``batches.sample_items``)."""

    def __init__(self, cfg: batches.ItemConfig) -> None:
        self.cfg = cfg

    def sample_items(self, rng: np.random.Generator, n: int) -> list[dict]:
        return batches.sample_items(rng, self.cfg, n)


class BankSampler:
    """Items for one arm, drawn from a pre-generated episode bank.

    The mixture is that of ``batches.sample_items``: a conflict pair with probability
    ``cfg.conflict_fraction``, else an ordinary item, whose query comes from a handover
    family with probability ``cfg.branch_fraction``.

    Args:
        bank: the loaded bank.
        cfg: the arm's item configuration; only its programs and counts are used.
        fresh_supports: generate every support episode instead of drawing it from the bank
            (queries and conflict families still come from the bank). With bank supports
            the model can recognise a support episode instead of reading its rule: held-out
            compositions then fail (pilots C and D).
    """

    def __init__(self, bank: Bank, cfg: batches.ItemConfig, fresh_supports: bool = False) -> None:
        self.bank = bank
        self.cfg = cfg
        self.fresh_supports = fresh_supports
        allowed = {program.key for program in cfg.programs}
        missing = allowed - set(bank.programs)
        if missing:
            raise SystemExit(f"{len(missing)} of this arm's programs are not in the bank")
        self.programs = {key: generator.Program(*key.split("|")) for key in bank.programs}
        self.allowed = np.array([key in allowed for key in bank.programs])

        self.pool: dict[tuple[int, int], np.ndarray] = {}      # (program, count) -> episode ids
        self.handover: dict[tuple[int, int], np.ndarray] = {}  # (program, count) -> family records
        self._index_pool_and_handover()
        self.conflict: dict[int, np.ndarray] = {}              # count -> family records
        self.by_pair: dict[tuple[int, int], np.ndarray] = {}   # program pair -> family records
        self._index_conflicts()
        self.arm_programs = sorted({program for program, _ in self.pool})

    def _index_pool_and_handover(self) -> None:
        records = self.bank.records
        for kind, store in ((POOL, self.pool), (HANDOVER, self.handover)):
            rows = records[(records[:, 0] == kind) & self.allowed[records[:, 1]]]
            for program, count in {(int(a), int(b)) for a, b in rows[:, [1, 3]]}:
                selected = rows[(rows[:, 1] == program) & (rows[:, 3] == count)]
                store[(program, count)] = selected[:, 4] if kind == POOL else selected

    def _index_conflicts(self) -> None:
        records = self.bank.records
        rows = records[(records[:, 0] == CONFLICT) & self.allowed[records[:, 1]] & (records[:, 2] >= 0)]
        rows = rows[self.allowed[rows[:, 2]]]          # the partner program must be allowed too
        # A family of (a, b) also serves as one of (b, a) with its queries swapped.
        low = np.minimum(rows[:, 1], rows[:, 2])
        high = np.maximum(rows[:, 1], rows[:, 2])
        for a, b in {(int(x), int(y)) for x, y in zip(low, high)}:
            self.by_pair[(a, b)] = rows[(low == a) & (high == b)]
        for count in self.cfg.counts.counts:
            self.conflict[count] = rows[rows[:, 3] == count]
            if not len(self.conflict[count]):
                raise SystemExit(f"bank has no conflict family at count {count} for this arm")

    def _support(self, rng: np.random.Generator, program: int, count: int, avoid: int = -1) -> int:
        """A pool episode id of ``program`` at ``count``; never ``avoid`` when another exists
        (a support and its query are independent scenes)."""
        ids = self.pool[(program, count)]
        if avoid >= 0 and len(ids) > 1:
            ids = ids[ids != avoid]
        return int(ids[int(rng.integers(len(ids)))])

    def _support_episode(self, rng: np.random.Generator, program: int, count: int) -> tuple[int, Episode]:
        """A support of ``program`` at ``count``: (bank id, episode), or (-1, a new episode)."""
        if not self.fresh_supports:
            support_id = self._support(rng, program, count)
            return support_id, self.bank.episode(support_id)
        prog = self.programs[self.bank.programs[program]]
        while True:
            try:
                return -1, generator.sample_episode(prog, count, batches._distractors(prog), rng)
            except generator.Infeasible:
                continue

    def single(self, rng: np.random.Generator) -> list[dict]:
        """One support and one query of the same program (``cfg.reuse`` items at other times)."""
        cfg = self.cfg
        p = self.arm_programs[int(rng.integers(len(self.arm_programs)))]
        program = self.programs[self.bank.programs[p]]
        s_count, q_count = cfg.counts.draw(rng)
        support_id, support = self._support_episode(rng, p, s_count)

        cue, branch = None, "nominal"
        families = self.handover.get((p, q_count))
        if (program.manipulation == "handover" and families is not None and len(families)
                and rng.random() < cfg.branch_fraction):
            family = families[int(rng.integers(len(families)))]
            cue = int(family[6])
            branch = "delayed" if rng.random() < 0.5 else "nominal"
            query = self.bank.episode(int(family[5] if branch == "delayed" else family[4]))
        else:
            query = self.bank.episode(self._support(rng, p, q_count, avoid=support_id))

        items = []
        for _ in range(max(1, cfg.reuse)):
            t, stratum = batches.query_time(query, rng)
            mapping = semantic_relabeling(rng) if cfg.relabel else None
            meta = {"program": program.key, "family": program.family, "branch": branch, "pair": None}
            items.append(batches._item(rng, cfg, support, query, t, stratum, cue, meta, mapping))
        return items

    def pair(self, rng: np.random.Generator) -> list[dict]:
        """A conflict pair: one query scene under two programs, each with its own support."""
        cfg = self.cfg
        s_count, q_count = cfg.counts.draw(rng)
        families = self.conflict[q_count]
        family = families[int(rng.integers(len(families)))]
        query_a = self.bank.episode(int(family[4]))
        query_b = self.bank.episode(int(family[5]))
        split = int(family[6])
        t, stratum = batches.query_time(query_a, rng, before=split)
        mapping = semantic_relabeling(rng) if cfg.relabel else None
        pair = int(rng.integers(1 << 62))

        items, extra = [], []
        for p, query in ((int(family[1]), query_a), (int(family[2]), query_b)):
            program = self.programs[self.bank.programs[p]]
            _, support = self._support_episode(rng, p, s_count)
            meta = {"program": program.key, "family": program.family, "branch": "nominal",
                    "pair": pair, "first_pick": query.annotations["selected"][0], "diverge": split}
            items.append(batches._item(rng, cfg, support, query, t, stratum, None, meta, mapping))
            for _ in range(cfg.pair_extra):
                other_t, other_stratum = batches.query_time(query, rng)
                meta = {"program": program.key, "family": program.family, "branch": "nominal",
                        "pair": None}
                extra.append(batches._item(rng, cfg, support, query, other_t, other_stratum, None,
                                           meta, None))
        return items + extra

    def sample_items(self, rng: np.random.Generator, n: int) -> list[dict]:
        items = []
        while len(items) < n:
            if rng.random() < self.cfg.conflict_fraction:
                items += self.pair(rng)
            else:
                items += self.single(rng)
        return items

    # ---- groups: one support and k queries (``batches.group``) ----------------------------

    def _query_count(self, rng: np.random.Generator, s_count: int) -> int:
        if self.cfg.counts.kind == "equal":
            return s_count
        return self.cfg.counts.draw(rng)[1]

    def group_single(self, rng: np.random.Generator, k: int) -> list[dict]:
        """One support of a program and ``k`` independent queries of the same program."""
        cfg = self.cfg
        p = self.arm_programs[int(rng.integers(len(self.arm_programs)))]
        program = self.programs[self.bank.programs[p]]
        s_count = cfg.counts.draw(rng)[0]
        support_id, support = self._support_episode(rng, p, s_count)
        entries = []
        for _ in range(k):
            q_count = self._query_count(rng, s_count)
            cue, branch = None, "nominal"
            families = self.handover.get((p, q_count))
            if (program.manipulation == "handover" and families is not None and len(families)
                    and rng.random() < cfg.branch_fraction):
                family = families[int(rng.integers(len(families)))]
                cue = int(family[6])
                branch = "delayed" if rng.random() < 0.5 else "nominal"
                query = self.bank.episode(int(family[5] if branch == "delayed" else family[4]))
            else:
                query = self.bank.episode(self._support(rng, p, q_count, avoid=support_id))
            t, stratum = batches.query_time(query, rng)
            meta = {"program": program.key, "family": program.family, "branch": branch, "pair": None}
            entries.append((query, t, stratum, cue, meta))
        mapping = semantic_relabeling(rng) if cfg.relabel else None
        return [batches.group(rng, cfg, support, entries, mapping)]

    def group_pair(self, rng: np.random.Generator, k: int) -> list[dict]:
        """Two groups (supports of programs a and b) over ``k`` conflict families of the
        pair (a, b): query j of each group shares a scene and a query time before the two
        executions diverge, so every item is a paired item."""
        cfg = self.cfg
        s_count, q_count = cfg.counts.draw(rng)
        first = self.conflict[q_count][int(rng.integers(len(self.conflict[q_count])))]
        a, b = int(first[1]), int(first[2])
        families = self.by_pair[(min(a, b), max(a, b))]
        if cfg.counts.kind == "equal":
            families = families[families[:, 3] == q_count]
        picks = [first]
        for _ in range(k - 1):
            picks.append(families[int(rng.integers(len(families)))])

        entries = {a: [], b: []}
        for family in picks:
            if int(family[1]) == a:
                id_a, id_b = int(family[4]), int(family[5])
            else:
                id_a, id_b = int(family[5]), int(family[4])
            query_a, query_b = self.bank.episode(id_a), self.bank.episode(id_b)
            split = int(family[6])
            t, stratum = batches.query_time(query_a, rng, before=split)
            pair = int(rng.integers(1 << 62))
            for p, query in ((a, query_a), (b, query_b)):
                program = self.programs[self.bank.programs[p]]
                meta = {"program": program.key, "family": program.family, "branch": "nominal",
                        "pair": pair, "first_pick": query.annotations["selected"][0],
                        "diverge": split}
                entries[p].append((query, t, stratum, None, meta))
        mapping = semantic_relabeling(rng) if cfg.relabel else None
        groups = []
        for p in (a, b):
            _, support = self._support_episode(rng, p, s_count)
            groups.append(batches.group(rng, cfg, support, entries[p], mapping))
        return groups

    def sample_groups(self, rng: np.random.Generator, n: int, k: int) -> list[dict]:
        """At least ``n`` groups of ``k`` queries (a conflict draw adds two groups)."""
        groups = []
        while len(groups) < n:
            if rng.random() < self.cfg.conflict_fraction:
                groups += self.group_pair(rng, k)
            else:
                groups += self.group_single(rng, k)
        return groups


class ItemStream(torch.utils.data.IterableDataset):
    """An endless stream of packed batches; each loader worker has its own random stream.

    Args:
        sampler: an ``OnTheFlySampler`` or a ``BankSampler``.
        batch: items per batch.
        seed: the run's seed (worker k draws from ``[seed, k]``).
        s_rows: support row budget of the packed batch.
    """

    def __init__(self, sampler, batch: int, seed: int, s_rows: int) -> None:
        self.sampler = sampler
        self.batch = batch
        self.seed = seed
        self.s_rows = s_rows

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker = 0 if info is None else info.id
        rng = np.random.default_rng([self.seed, worker])
        pending: list[dict] = []
        while True:
            # A sampler may return more than asked (a pair, reused items): the rest starts
            # the next batch.
            if len(pending) < self.batch:
                pending += self.sampler.sample_items(rng, self.batch - len(pending))
            items, pending = pending[:self.batch], pending[self.batch:]
            packed = batches.pack(items, s_rows=self.s_rows)
            yield {key: torch.as_tensor(value) for key, value in packed.items()}


class GroupStream(ItemStream):
    """Like ``ItemStream``, but a batch is ``batch / per_support`` groups of one support
    and ``per_support`` queries (``BankSampler.sample_groups``), packed with
    ``support_index`` so each support is encoded once."""

    def __init__(self, sampler: BankSampler, batch: int, seed: int, s_rows: int,
                 per_support: int) -> None:
        super().__init__(sampler, batch, seed, s_rows)
        self.per_support = per_support

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker = 0 if info is None else info.id
        rng = np.random.default_rng([self.seed, worker])
        n_groups = self.batch // self.per_support
        pending: list[dict] = []
        while True:
            # A conflict draw adds two groups: the extra one starts the next batch.
            if len(pending) < n_groups:
                pending += self.sampler.sample_groups(rng, n_groups - len(pending), self.per_support)
            groups, pending = pending[:n_groups], pending[n_groups:]
            packed = batches.pack_groups(groups, s_rows=self.s_rows)
            yield {key: torch.as_tensor(value) for key, value in packed.items()}


def build_dataloader(sampler, batch: int, seed: int, s_rows: int, workers: int,
                     per_support: int = 1) -> torch.utils.data.DataLoader:
    """Batches come packed from the stream (``batch_size=None``)."""
    if per_support > 1:
        stream = GroupStream(sampler, batch, seed, s_rows, per_support)
    else:
        stream = ItemStream(sampler, batch, seed, s_rows)
    return torch.utils.data.DataLoader(stream, batch_size=None, num_workers=workers,
                                       persistent_workers=workers > 0)
