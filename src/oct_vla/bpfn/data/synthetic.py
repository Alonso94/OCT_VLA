"""Stage-1 synthetic data: the training episode bank and the fixed evaluation packs."""

from __future__ import annotations

from pathlib import Path

from oct_vla.bpfn.data import bank, packs


class SyntheticData:
    """Generates the synthetic data of stage 1.

    Args:
        bank_root: directory of the episode bank (None: no bank).
        bank_spec: what the bank holds (units per program and count, shards).
        packs_root: directory of the evaluation packs (None: no packs).
        pack_names: which packs to build (``packs.PACKS``).
        pack_items: items per pack.
        s_rows: support row budget of the packs.
        workers: processes for the bank (None: the CPUs this job owns).
        programs: the bank's programs (None: every training program; tests pass a subset).
    """

    def __init__(self, bank_root: Path | None = None, bank_spec: bank.BankSpec | None = None,
                 packs_root: Path | None = None, pack_names: tuple[str, ...] = (),
                 pack_items: int = 512, s_rows: int = 128, workers: int | None = None,
                 programs: tuple | None = None) -> None:
        self.bank_root = bank_root
        self.bank_spec = bank_spec or bank.BankSpec()
        self.packs_root = packs_root
        self.pack_names = pack_names
        self.pack_items = pack_items
        self.s_rows = s_rows
        self.workers = workers
        self.programs = programs

    def generate_data(self, shard_ids: list[int] | None = None) -> None:
        """Builds whatever is missing; finished shards and existing packs are kept."""
        if self.bank_root is not None:
            bank.build(self.bank_root, self.bank_spec, shards=shard_ids, workers=self.workers,
                       programs=self.programs)
        for name in self.pack_names:
            path = Path(self.packs_root) / f"{name}.pkl"
            if path.exists():
                print(f"{path} exists; kept")
                continue
            packs.build(name, self.packs_root, self.pack_items, s_rows=self.s_rows)
            print(f"{path} written")
