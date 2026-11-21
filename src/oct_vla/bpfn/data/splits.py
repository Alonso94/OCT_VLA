"""Split registry and count matrices (plan stage 0.2).

- **Seeds.** Every split draws from its own hashed stream (``item_rng``), so train, dev
  and test items never share a scene; dev and test are fixed packs.
- **Held-out composition.** ``handover`` x ``colour`` selection is sealed: no training arm
  sees it; ``heldout_handover_colour`` evaluates it on mixed-colour multi-object scenes
  while both constituents (carry x colour, handover x all/category) stay in training.
- **Sealed count.** Five eligible objects appear only in the sealed extrapolation split.
- **Count matrices.** ``full``: support and query eligible counts drawn independently from
  {2, 3, 4}; ``equal``: query = support, same marginal. Final count evaluation fixes the
  support at 3.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from oct_vla.bpfn.data.syn_data.generator import Program, all_programs

NAMESPACE = "pose-window-v1"
SPLITS = ("train", "dev", "test")
HELD_OUT = {("handover", "colour"): "heldout_handover_colour"}
COUNTS = (2, 3, 4)
SEALED_COUNT = 5
#: Training arms (plan 1b). Each differs from ``full`` in exactly one respect.
ARMS = ("full", "equal_count", "scene_only", "no_handover", "nominal_only")


def item_rng(split: str, index: int, namespace: str = NAMESPACE) -> np.random.Generator:
    if split not in SPLITS and split not in HELD_OUT.values() and split != "sealed_count":
        raise ValueError(f"unknown split {split!r}")
    digest = hashlib.sha256(f"{namespace}/{split}/{index}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def held_out_name(program: Program) -> str | None:
    return HELD_OUT.get(program.family)


def programs_for(arm: str = "full") -> tuple[Program, ...]:
    """Training programs of an arm: never a held-out family; ``no_handover`` also drops
    every handover program (its per-arm motion is covered by carries)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    progs = [p for p in all_programs() if held_out_name(p) is None]
    if arm == "no_handover":
        progs = [p for p in progs if p.manipulation != "handover"]
    return tuple(progs)


def heldout_programs(name: str = "heldout_handover_colour") -> tuple[Program, ...]:
    return tuple(p for p in all_programs() if held_out_name(p) == name)


@dataclass(frozen=True)
class CountMatrix:
    kind: str = "full"           # "full" | "equal" | "eval3"
    counts: tuple[int, ...] = COUNTS

    def draw(self, rng: np.random.Generator) -> tuple[int, int]:
        """(support eligible, query eligible)."""
        if self.kind == "eval3":
            return 3, int(rng.choice(self.counts))
        s = int(rng.choice(self.counts))
        if self.kind == "equal":
            return s, s
        if self.kind == "full":
            return s, int(rng.choice(self.counts))
        raise ValueError(f"unknown count matrix {self.kind!r}")


def count_matrix_for(arm: str) -> CountMatrix:
    return CountMatrix("equal" if arm == "equal_count" else "full")
