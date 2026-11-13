"""v2 task programs: v1's selector × destination × arrangement × arm, plus compaction.

The v1 grammar (`programs.py`) and its program ids are untouched. v2 programs are enumerated
separately, so a v1 checkpoint, evaluation set or language table never sees a changed id.

`compact` is the push prior (paper/5_time_row_pfn_design.md §10):
- `none`: v1 behaviour;
- `toward_neighbour`: after each row placement next to an earlier object, the *other* arm
  pushes the placed object along the row until the gap to its neighbour is `COMPACT_GAP`.

Compaction applies to row arrangements only; a stack has nothing to compact.

The scene does not reveal whether a task compacts. Only the demonstrations (or an explicit
instruction) do, so conflict pairs that differ only in `compact` test in-context task
inference. The exact shelf program (shelf-family selectors × higher × row × left/nearest ×
compaction) is held out together with v1's shelf family; its components appear with other
partners.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from . import programs as P

COMPACTION = ("none", "toward_neighbour")

_COMPACT_TEXT = {"none": "", "toward_neighbour": ", then push each one against the previous"}


@dataclass(frozen=True)
class ProgramV2:
    selector: str
    destination: str
    arrangement: str
    arm: str
    compact: str = "none"

    def __post_init__(self) -> None:
        P.Program(self.selector, self.destination, self.arrangement, self.arm)  # validates
        if self.compact not in COMPACTION:
            raise ValueError(f"{self.compact!r} is not one of {COMPACTION}")
        if self.compact != "none" and self.arrangement == "stack":
            raise ValueError("a stack has nothing to compact")

    @property
    def v1(self) -> P.Program:
        return P.Program(self.selector, self.destination, self.arrangement, self.arm)

    @property
    def id(self) -> int:
        return ALL_PROGRAMS_V2.index(self)

    def instructions(self) -> tuple[str, str]:
        tail = _COMPACT_TEXT[self.compact]
        a, b = self.v1.instructions()
        return (a[:-1] + tail + ".", b[:-1] + tail + ".")

    def differs_in(self, other: "ProgramV2") -> tuple[str, ...]:
        return tuple(f for f in ("selector", "destination", "arrangement", "arm", "compact")
                     if getattr(self, f) != getattr(other, f))


ALL_PROGRAMS_V2: tuple[ProgramV2, ...] = tuple(
    ProgramV2(p.selector, p.destination, p.arrangement, p.arm, c)
    for p, c in itertools.product(P.ALL_PROGRAMS, COMPACTION)
    if not (c != "none" and p.arrangement == "stack")
)

SHELF_FAMILY_V2 = frozenset(p for p in ALL_PROGRAMS_V2 if p.v1 in P.SHELF_FAMILY)
COMPOSITION_HOLDOUT_V2 = frozenset(p for p in ALL_PROGRAMS_V2 if p.v1 in P.COMPOSITION_HOLDOUT)
TRAIN_PROGRAMS_V2: tuple[ProgramV2, ...] = tuple(
    p for p in ALL_PROGRAMS_V2 if p not in SHELF_FAMILY_V2 and p not in COMPOSITION_HOLDOUT_V2)

ALL_INSTRUCTIONS_V2: tuple[str, ...] = tuple(dict.fromkeys(
    [*P.AMBIGUOUS_INSTRUCTIONS, *(s for p in ALL_PROGRAMS_V2 for s in p.instructions())]))


def program_split_v2(name: str) -> tuple[ProgramV2, ...]:
    if name == "train":
        return TRAIN_PROGRAMS_V2
    if name == "composition":
        return tuple(sorted(COMPOSITION_HOLDOUT_V2, key=ALL_PROGRAMS_V2.index))
    if name == "shelf_family":
        return tuple(sorted(SHELF_FAMILY_V2, key=ALL_PROGRAMS_V2.index))
    raise ValueError(f"unknown program split {name!r}")
